"""Load HuggingFace models into Diffulex Edge models.

Supports 4 model types: FastdLLM V2, Dream, LLaDA, SDAR
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from safetensors import safe_open

from .fast_dllm_v2_edge import FastDLLMv2Edge, FastDLLMv2EdgeConfig
from .dream_edge import DreamEdge, DreamEdgeConfig
from .llada_edge import LLaDAEdge, LLaDAEdgeConfig
from .sdar_edge import SDAREdge, SDAREdgeConfig


def get_safetensor_files(model_path: Path) -> List[Path]:
    """Get list of safetensor files in model directory.
    
    Handles both single-file and multi-file (sharded) models.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        List of paths to safetensor files
    """
    # Check for single file first
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return [single_file]
    
    # Check for index file (sharded model)
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        
        # Get unique weight files from index
        weight_files = set()
        for key, filename in index.get("weight_map", {}).items():
            weight_files.add(model_path / filename)
        
        # Sort for consistent ordering
        return sorted(weight_files)
    
    # Check for pattern-matched files
    pattern_files = sorted(model_path.glob("model-*.safetensors"))
    if pattern_files:
        return pattern_files
    
    return []


MODEL_REGISTRY = {
    "fast_dllm_v2": (FastDLLMv2Edge, FastDLLMv2EdgeConfig),
    "dream": (DreamEdge, DreamEdgeConfig),
    "llada": (LLaDAEdge, LLaDAEdgeConfig),
    "sdar": (SDAREdge, SDAREdgeConfig),
}


def detect_model_type(config: Dict[str, Any]) -> str:
    """Detect model type from HF config.
    
    Args:
        config: HuggingFace config dict
        
    Returns:
        Model type string
    """
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")
    
    # Check architectures
    for arch in architectures:
        arch_lower = arch.lower()
        if "fast" in arch_lower and "dllm" in arch_lower:
            return "fast_dllm_v2"
        elif "dream" in arch_lower:
            return "dream"
        elif "llada" in arch_lower:
            return "llada"
        elif "sdar" in arch_lower:
            return "sdar"
    
    # Check model_type
    if "fast" in model_type or "dllm" in model_type:
        return "fast_dllm_v2"
    elif "dream" in model_type:
        return "dream"
    elif "llada" in model_type:
        return "llada"
    elif "sdar" in model_type:
        return "sdar"
    
    # Default
    return "fast_dllm_v2"


def create_edge_config(model_type: str, hf_config: Dict[str, Any]) -> Any:
    """Create Edge model config from HF config.
    
    Args:
        model_type: Model type string
        hf_config: HF config dict
        
    Returns:
        Edge config object
    """
    _, ConfigClass = MODEL_REGISTRY[model_type]
    
    # Common parameters
    common_params = {
        "vocab_size": hf_config.get("vocab_size", 32000),
        "hidden_size": hf_config.get("hidden_size", 2048),
        "intermediate_size": hf_config.get("intermediate_size", 5504),
        "max_position_embeddings": hf_config.get("max_position_embeddings", 32768),
    }
    
    # Model-specific parameters
    if model_type == "fast_dllm_v2":
        return ConfigClass(
            vocab_size=common_params["vocab_size"],
            hidden_size=common_params["hidden_size"],
            num_hidden_layers=hf_config.get("num_hidden_layers", 28),
            num_attention_heads=hf_config.get("num_attention_heads", 12),
            num_key_value_heads=hf_config.get("num_key_value_heads", 2),
            intermediate_size=common_params["intermediate_size"],
            max_position_embeddings=common_params["max_position_embeddings"],
            head_dim=hf_config.get("head_dim", None),  # Will be calculated
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
            attention_bias=hf_config.get("attention_bias", True),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
            bd_size=hf_config.get("bd_size", 32),
        )
    elif model_type == "dream":
        return ConfigClass(
            vocab_size=common_params["vocab_size"],
            hidden_size=common_params["hidden_size"],
            num_hidden_layers=hf_config.get("num_hidden_layers", 28),
            num_attention_heads=hf_config.get("num_attention_heads", 28),
            num_key_value_heads=hf_config.get("num_key_value_heads", 4),
            intermediate_size=common_params["intermediate_size"],
            max_position_embeddings=common_params["max_position_embeddings"],
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
            attention_bias=hf_config.get("attention_bias", True),
            attention_dropout=hf_config.get("attention_dropout", 0.0),
            head_dim=common_params["hidden_size"] // hf_config.get("num_attention_heads", 28),
            mask_token_id=hf_config.get("mask_token_id", 151666),
            pad_token_id=hf_config.get("pad_token_id", 151643),
            bos_token_id=hf_config.get("bos_token_id", 151643),
            eos_token_id=hf_config.get("eos_token_id", 151643),
        )
    elif model_type == "llada":
        return ConfigClass(
            vocab_size=common_params["vocab_size"],
            hidden_size=common_params["hidden_size"],
            num_hidden_layers=hf_config.get("num_hidden_layers", 28),
            num_attention_heads=hf_config.get("num_attention_heads", 16),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            intermediate_size=common_params["intermediate_size"],
            max_position_embeddings=common_params["max_position_embeddings"],
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
            attention_bias=hf_config.get("attention_bias", False),
            head_dim=hf_config.get("head_dim", 128),
            mask_token_id=hf_config.get("mask_token_id", 126336),
        )
    elif model_type == "sdar":
        return ConfigClass(
            vocab_size=common_params["vocab_size"],
            hidden_size=common_params["hidden_size"],
            num_hidden_layers=hf_config.get("num_hidden_layers", 28),
            num_attention_heads=hf_config.get("num_attention_heads", 16),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            intermediate_size=common_params["intermediate_size"],
            max_position_embeddings=common_params["max_position_embeddings"],
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
            attention_bias=hf_config.get("attention_bias", False),
            head_dim=hf_config.get("head_dim", 128),
            diffusion_block_size=hf_config.get("diffusion_block_size", 4),  # SDAR default is 4
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_hf_weights_to_edge(
    model_path: str,
    edge_model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, Dict[str, Any]]:
    """Load HuggingFace weights into Edge model.
    
    Supports both single-file and multi-file (sharded) safetensors.
    
    Args:
        model_path: Path to HF model directory
        edge_model: Edge model instance
        dtype: Target dtype
        
    Returns:
        (success, config) tuple
    """
    model_path = Path(model_path)
    
    # Load config
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    
    # Get safetensor files
    safetensor_files = get_safetensor_files(model_path)
    if not safetensor_files:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    print(f"Loading weights from {len(safetensor_files)} file(s)...")
    loaded = 0
    mismatched = []
    missing = []
    
    # Load all weights into memory first (for multi-file support)
    state_dict = {}
    for file_path in safetensor_files:
        print(f"  Reading {file_path.name}...")
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    print(f"  Total tensors in checkpoint: {len(state_dict)}")
    
    # Map and load weights
    for hf_key, tensor in state_dict.items():
        # Map HF key to Edge key
        # HF Dream uses "model.xxx" prefix
        edge_key = hf_key[6:] if hf_key.startswith("model.") else hf_key
        
        # Get tensor with target dtype
        tensor = tensor.to(dtype)
        
        # Set in model
        try:
            parts = edge_key.split(".")
            module = edge_model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            param_name = parts[-1]
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if isinstance(param, nn.Parameter):
                    if param.shape == tensor.shape:
                        param.data.copy_(tensor)
                        loaded += 1
                    else:
                        mismatched.append((edge_key, param.shape, tensor.shape))
                elif hasattr(param, 'data'):
                    if param.shape == tensor.shape:
                        param.data.copy_(tensor)
                        loaded += 1
                    else:
                        mismatched.append((edge_key, param.shape, tensor.shape))
            else:
                missing.append(edge_key)
        except (AttributeError, IndexError) as e:
            missing.append(edge_key)
    
    print(f"  Loaded {loaded} tensors")
    if mismatched:
        print(f"  {len(mismatched)} shape mismatches")
        for key, model_shape, hf_shape in mismatched[:5]:
            print(f"    {key}: model{model_shape} vs hf{hf_shape}")
    if missing:
        print(f"  {len(missing)} keys not found in model (first 5): {missing[:5]}")
    
    return loaded > 0, config


def optimize_for_cpu():
    """Configure PyTorch for optimal CPU performance.
    
    Sets thread count and enables optimizations for CPU inference.
    """
    import os
    
    # Get available CPU cores
    cpu_count = os.cpu_count() or 1
    
    # Set PyTorch thread settings
    torch.set_num_threads(min(cpu_count, 64))
    torch.set_num_interop_threads(min(cpu_count // 2, 32))
    
    # Enable OpenMP if available
    if hasattr(torch.backends, 'openmp'):
        torch.backends.openmp.enable = True
    
    # Enable MKL-DNN if available
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    # Set environment variables for OpenMP and MKL
    os.environ.setdefault('OMP_NUM_THREADS', str(min(cpu_count, 64)))
    os.environ.setdefault('MKL_NUM_THREADS', str(min(cpu_count, 64)))
    
    return {
        'num_threads': torch.get_num_threads(),
        'num_interop_threads': torch.get_num_interop_threads(),
    }


def load_hf_model(
    model_path: str, 
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    optimize_cpu: bool = True,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """Load HuggingFace model into Edge model.
    
    Args:
        model_path: Path to HF model directory
        dtype: Target dtype
        device: Device to load model on ("cpu" or "cuda")
        optimize_cpu: Whether to enable CPU optimizations
        
    Returns:
        (model, model_type, config) tuple
    """
    # Optimize CPU settings if requested
    if device == "cpu" and optimize_cpu:
        settings = optimize_for_cpu()
        print(f"CPU optimized: {settings}")
    
    # Load config first
    with open(Path(model_path) / "config.json", "r") as f:
        hf_config = json.load(f)
    
    # Detect model type
    model_type = detect_model_type(hf_config)
    print(f"Detected model type: {model_type}")
    
    # Create Edge config
    edge_config = create_edge_config(model_type, hf_config)
    
    # Create model
    ModelClass, _ = MODEL_REGISTRY[model_type]
    model = ModelClass(edge_config)
    
    # Load weights
    success, _ = load_hf_weights_to_edge(model_path, model, dtype)
    if not success:
        raise RuntimeError("Failed to load weights")
    
    # Move to device
    model = model.to(device)
    
    # Enable inference optimizations
    model.eval()
    
    # TorchScript or compile for better performance (optional)
    # model = torch.compile(model, mode="reduce-overhead")
    
    return model, model_type, hf_config
