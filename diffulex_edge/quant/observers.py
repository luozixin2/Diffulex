"""Observer configurations for quantization.

Observers collect statistics during calibration to determine
optimal quantization parameters (scale and zero_point).
"""

from typing import Optional
import torch
from torch.ao.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


def get_default_observers(dtype: torch.dtype = torch.qint8) -> dict:
    """Get default observer configuration for static quantization.
    
    Uses per-tensor quantization for activations and per-channel
    for weights. Good balance of accuracy and performance.
    
    Args:
        dtype: Quantization data type (qint8 or quint8)
        
    Returns:
        Dict with observer factory functions
    """
    return {
        "activation": lambda: MinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_tensor_affine,
            reduce_range=True,  # Use 127 instead of 128 for symmetry with backends
        ),
        "weight": lambda: PerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            reduce_range=True,
            ch_axis=0,  # Channel dimension for linear layers
        ),
    }


def get_per_channel_observers(dtype: torch.dtype = torch.qint8) -> dict:
    """Get per-channel observer configuration.
    
    Uses per-channel quantization for both activations and weights.
    Highest accuracy but may have lower performance on some backends.
    
    Args:
        dtype: Quantization data type
        
    Returns:
        Dict with observer factory functions
    """
    return {
        "activation": lambda: PerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            reduce_range=True,
            ch_axis=-1,  # Last dimension for activations
        ),
        "weight": lambda: PerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            reduce_range=True,
            ch_axis=0,
        ),
    }


def get_moving_average_observers(
    dtype: torch.dtype = torch.qint8,
    averaging_constant: float = 0.01,
) -> dict:
    """Get moving average observer configuration.
    
    Better for streaming/long sequences as it adapts to distribution changes.
    
    Args:
        dtype: Quantization data type
        averaging_constant: Weight for new values (0-1)
        
    Returns:
        Dict with observer factory functions
    """
    return {
        "activation": lambda: MovingAverageMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_tensor_affine,
            reduce_range=True,
            averaging_constant=averaging_constant,
        ),
        "weight": lambda: MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            reduce_range=True,
            ch_axis=0,
            averaging_constant=averaging_constant,
        ),
    }


def get_dynamic_quant_observers(dtype: torch.dtype = torch.qint8) -> dict:
    """Get observer configuration for dynamic quantization.
    
    Only observes weights (activations quantized dynamically at runtime).
    
    Args:
        dtype: Quantization data type
        
    Returns:
        Dict with observer factory functions
    """
    return {
        "weight": lambda: PerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            reduce_range=True,
            ch_axis=0,
        ),
        "activation": None,  # Not used for dynamic quantization
    }
