"""Model loader."""

from __future__ import annotations

import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from safetensors import safe_open
from diffulex.config import Config
from diffulex.logger import get_logger
from diffulex.utils.quantization.loader_adapter import load_offline_quantized_weight

logger = get_logger(__name__)


def _read_quantize_config(model_dir: str) -> dict:
    """Read vLLM-style quantization metadata."""
    cfg_path = os.path.join(model_dir, "quantize_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _find_offline_capable_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Find modules that support offline quantized weights."""
    return {
        name: m for name, m in model.named_modules()
        if hasattr(m, "set_delegate")
    }


def _find_module(name: str, candidates: dict[str, nn.Module]) -> nn.Module | None:
    """Find module by name with fallback strategies."""
    if name in candidates:
        return candidates[name]
    leaf = name.split(".")[-1]
    for n, m in candidates.items():
        if n.endswith("." + name) or n.split(".")[-1] == leaf:
            return m
    return None


def _apply_tp_sharding_to_tensors(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
    tp_dim: int,
    format: str,
    bits: int,
    is_marlin: bool,
) -> dict[str, torch.Tensor]:
    """Apply tensor parallelism sharding to quantized tensors."""
    qweight, scales = tensors["qweight"], tensors["scales"]
    qzeros = tensors.get("qzeros")
    g_idx = tensors.get("g_idx")
    
    # Infer shapes (GPTQ: qweight[K/pack, N], AWQ: qweight[K, N/pack])
    if format == "gptq":
        out_features = int(qweight.shape[1])
        pack_factor = (32 // bits) if bits else 8
        in_features = int(qweight.shape[0]) * (16 if is_marlin else pack_factor)
    else:  # awq
        out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
        pack_factor = 8
        in_features = int(qweight.shape[0])
    
    if tp_dim == 0:  # Column parallel: shard output features
        out_per = out_features // tp_size
        out_start = tp_rank * out_per
        out_end = out_start + out_per
        out_packed_start, out_packed_end = out_start // pack_factor, out_end // pack_factor
        
        if format == "gptq":
            if is_marlin:
                n_factor = bits // 2
                qweight = qweight[:, (out_start * n_factor):(out_end * n_factor)]
            else:
                qweight = qweight[:, out_start:out_end]
                if qzeros is not None and qzeros.numel() > 0:
                    qzeros = qzeros[:, out_packed_start:out_packed_end]
        else:  # awq
            qweight = qweight[:, out_packed_start:out_packed_end]
            if qzeros is not None:
                qzeros = qzeros[:, out_packed_start:out_packed_end]
        
        scales = scales[:, out_start:out_end] if scales.ndim == 2 else scales
        
    else:  # Row parallel: shard input features
        in_per = in_features // tp_size
        in_start = tp_rank * in_per
        in_end = in_start + in_per
        
        if format == "gptq":
            if is_marlin:
                q_start, q_end = in_start // 16, in_end // 16
                qweight = qweight[q_start:q_end, :]
                num_groups = int(scales.shape[0])
                expected_groups = in_features // 128
                if num_groups == expected_groups:
                    g_start, g_end = in_start // 128, in_end // 128
                    scales = scales[g_start:g_end, :]
                elif num_groups == 2 * expected_groups:
                    g_start, g_end = 2 * in_start // 128, 2 * in_end // 128
                    scales = scales[g_start:g_end, :]
            else:
                q_start, q_end = in_start // pack_factor, in_end // pack_factor
                g_start, g_end = in_start // 128, in_end // 128
                qweight = qweight[q_start:q_end, :]
                if qzeros is not None and qzeros.numel() > 0:
                    qzeros = qzeros[g_start:g_end, :]
                scales = scales[g_start:g_end, :]
        else:  # awq
            g_start, g_end = in_start // 128, in_end // 128
            qweight = qweight[in_start:in_end, :]
            if qzeros is not None:
                qzeros = qzeros[g_start:g_end, :]
            scales = scales[g_start:g_end, :]
        
        if g_idx is not None and g_idx.numel() > 0:
            g_idx = g_idx[in_start:in_end]
    
    result = {"qweight": qweight, "scales": scales}
    if qzeros is not None:
        result["qzeros"] = qzeros
    if g_idx is not None:
        result["g_idx"] = g_idx
    return result


def _load_gptq_awq_weights(model: nn.Module, config: Config) -> tuple[int, int, int]:
    """Load GPTQ/AWQ offline quantized weights."""
    quant_cfg = _read_quantize_config(config.model)
    ckpt_bits = quant_cfg.get("bits", 4)
    ckpt_group_size = quant_cfg.get("group_size", 128)
    is_marlin = quant_cfg.get("checkpoint_format", "").lower() == "gptq_marlin"
    
    if not (config.load_gptq or config.load_awq):
        return 0, 0, 0
    
    # Build key->file mapping
    key_to_file = {}
    for file in glob(os.path.join(config.model, "*.safetensors")):
        try:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    key_to_file[key] = file
        except Exception as e:
            logger.warning(f"Failed to read {file}: {e}")
    
    # Group keys by module prefix
    module_keys = {}
    for key in key_to_file:
        if ".qweight" in key:
            prefix = key[:-8]
            module_keys.setdefault(prefix, {})["qweight"] = key
        elif ".qzeros" in key:
            module_keys.setdefault(key[:-7], {})["qzeros"] = key
        elif ".scales" in key:
            module_keys.setdefault(key[:-7], {})["scales"] = key
        elif ".g_idx" in key:
            module_keys.setdefault(key[:-6], {})["g_idx"] = key
    
    offline_modules = _find_offline_capable_modules(model)
    loaded_gptq, loaded_awq, skipped = 0, 0, 0
    
    for module_name, key_dict in module_keys.items():
        try:
            module = _find_module(module_name, offline_modules)
            if module is None:
                skipped += 1
                continue
            
            # Determine format
            has_g_idx = "g_idx" in key_dict
            is_gptq = has_g_idx or is_marlin
            
            if is_gptq and not config.load_gptq:
                skipped += 1
                continue
            if not is_gptq and not config.load_awq:
                skipped += 1
                continue
            
            format = "gptq" if is_gptq else "awq"
            
            # Load tensors
            files = {key_to_file.get(k) for k in key_dict.values() if k}
            files.discard(None)
            tensors = {}
            for file in files:
                with safe_open(file, "pt", "cpu") as f:
                    for k, v in key_dict.items():
                        if k not in tensors and v in f.keys():
                            tensors[k] = f.get_tensor(v)
            
            if "qweight" not in tensors or "scales" not in tensors:
                skipped += 1
                continue
            
            # Apply TP sharding if needed
            tp_size = getattr(module, "tp_size", 1) or 1
            tp_rank = getattr(module, "tp_rank", 0) or 0
            tp_dim = getattr(module, "tp_dim", None)
            
            if tp_size > 1 and tp_dim in (0, 1):
                tensors = _apply_tp_sharding_to_tensors(
                    tensors, tp_rank, tp_size, tp_dim,
                    format, ckpt_bits, is_marlin
                )
            
            load_offline_quantized_weight(
                module=module,
                qweight=tensors["qweight"],
                qzeros=tensors.get("qzeros"),
                scales=tensors["scales"],
                g_idx=tensors.get("g_idx"),
                format=format,
                bits=ckpt_bits,
                group_size=ckpt_group_size,
                is_marlin=is_marlin,
            )
            
            if format == "gptq":
                loaded_gptq += 1
            else:
                loaded_awq += 1
                
        except Exception as e:
            logger.exception(f"Failed to load offline quant weights for {module_name}: {e}")
            skipped += 1
    
    return loaded_gptq, loaded_awq, skipped


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def enable_lora_for_model(model: nn.Module, lora_config: dict) -> nn.Module:
    """Enable LoRA for linear layers in the model."""
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 32.0)
    lora_dropout = lora_config.get('lora_dropout', 0.0)
    target_modules = lora_config.get('target_modules', [])
    
    for name, module in model.named_modules():
        if hasattr(module, '__init_lora__'):
            if not target_modules or any(t in name for t in target_modules):
                module.__init_lora__(r, lora_alpha, lora_dropout)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader: simple copy."""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, config: Config):
    """Load model weights."""
    # Enable LoRA if configured
    if config.use_lora and config.lora_path:
        lora_config = load_lora_config(config.lora_path)
        model = enable_lora_for_model(model, lora_config or {'r': 16, 'lora_alpha': 32.0, 'lora_dropout': 0.0})
    
    # Load offline quantized weights
    loaded_gptq, loaded_awq, skipped = _load_gptq_awq_weights(model, config)
    if loaded_gptq or loaded_awq:
        print(f"Loaded offline quantized weights: GPTQ={loaded_gptq}, AWQ={loaded_awq}, skipped={skipped}")
    
    # Load base model weights
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if any(k in weight_name for k in (".qweight", ".qzeros", ".scales", ".g_idx")):
                    continue  # Skip quantized weights
                
                # Map weight name to model parameter
                param_name = weight_name.replace(".weight", "").replace(".", "_")
                if param_name in dict(model.named_parameters()):
                    param = dict(model.named_parameters())[param_name]
                    if hasattr(param, 'weight_loader'):
                        param.weight_loader(param, f.get_tensor(weight_name))
                    else:
                        default_weight_loader(param, f.get_tensor(weight_name))


def load_lora_weights(model: nn.Module, lora_path: str, packed_modules_mapping: dict | None = None):
    """Load LoRA weights from a checkpoint."""
    if not os.path.exists(lora_path):
        return
    
    lora_weights = {}
    for file in glob(os.path.join(lora_path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for key in f.keys():
                lora_weights[key] = f.get_tensor(key)
    
    # Apply LoRA weights to model
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"
            if lora_a_key in lora_weights and lora_b_key in lora_weights:
                module.lora_A.data.copy_(lora_weights[lora_a_key])
                module.lora_B.data.copy_(lora_weights[lora_b_key])
