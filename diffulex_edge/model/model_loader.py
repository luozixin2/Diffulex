"""Load HuggingFace models into Diffulex Edge models.

Supports 4 model types: FastdLLM V2, Dream, LLaDA, SDAR
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from safetensors import safe_open

from .fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from .dream_edge import DreamEdge, DreamEdgeConfig
from .llada_edge import LLaDAEdge, LLaDAEdgeConfig
from .sdar_edge import SDAREdge, SDAREdgeConfig


MODEL_REGISTRY = {
    "fast_dllm_v2": (FastdLLMV2Edge, FastdLLMV2EdgeConfig),
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
            num_layers=hf_config.get("num_hidden_layers", 28),
            num_heads=hf_config.get("num_attention_heads", 16),
            num_kv_heads=hf_config.get("num_key_value_heads", 8),
            intermediate_size=common_params["intermediate_size"],
            max_seq_len=common_params["max_position_embeddings"],
            head_dim=hf_config.get("head_dim", 128),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
        )
    elif model_type == "dream":
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
            attention_bias=hf_config.get("attention_bias", True),
            head_dim=hf_config.get("head_dim", 128),
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
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_hf_weights_to_edge(
    model_path: str,
    edge_model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, Dict[str, Any]]:
    """Load HuggingFace weights into Edge model.
    
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
    
    # Load weights
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        # Try alternative names
        for alt_name in ["model.safetensors.index.json", "pytorch_model.bin"]:
            alt_path = model_path / alt_name
            if alt_path.exists():
                raise NotImplementedError(f"Alternative weight format not supported: {alt_name}")
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    print(f"Loading weights from {safetensors_path}...")
    loaded = 0
    mismatched = []
    
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for hf_key in f.keys():
            # Map HF key to Edge key
            edge_key = hf_key[6:] if hf_key.startswith("model.") else hf_key
            
            # Get tensor
            tensor = f.get_tensor(hf_key).to(dtype)
            
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
                    if param.shape == tensor.shape:
                        param.data.copy_(tensor)
                        loaded += 1
                    else:
                        mismatched.append((edge_key, param.shape, tensor.shape))
            except (AttributeError, IndexError):
                pass
    
    print(f"  Loaded {loaded} tensors")
    if mismatched:
        print(f"  {len(mismatched)} shape mismatches")
        for key, model_shape, hf_shape in mismatched[:5]:
            print(f"    {key}: model{model_shape} vs hf{hf_shape}")
    
    return loaded > 0, config


def load_hf_model(model_path: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """Load HuggingFace model into Edge model.
    
    Args:
        model_path: Path to HF model directory
        dtype: Target dtype
        
    Returns:
        (model, model_type, config) tuple
    """
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
    
    model.eval()
    return model, model_type, hf_config
