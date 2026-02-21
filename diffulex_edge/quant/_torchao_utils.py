"""Utilities for torchao integration.

Handles optional torchao dependency with graceful degradation.
"""

import functools
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

_TORCHAO_AVAILABLE: Optional[bool] = None
_TORCHAO_VERSION: Optional[str] = None


def check_torchao_available() -> bool:
    """Check if torchao is available without importing it.
    
    Returns:
        True if torchao is installed and available
    """
    global _TORCHAO_AVAILABLE, _TORCHAO_VERSION
    if _TORCHAO_AVAILABLE is None:
        try:
            import torchao
            _TORCHAO_AVAILABLE = True
            _TORCHAO_VERSION = getattr(torchao, "__version__", "unknown")
            logger.info(f"torchao {_TORCHAO_VERSION} is available")
        except ImportError:
            _TORCHAO_AVAILABLE = False
            _TORCHAO_VERSION = None
    return _TORCHAO_AVAILABLE


def get_torchao_version() -> Optional[str]:
    """Get torchao version if available.
    
    Returns:
        Version string or None if not available
    """
    global _TORCHAO_VERSION
    if _TORCHAO_AVAILABLE is None:
        check_torchao_available()
    return _TORCHAO_VERSION


def requires_torchao(min_version: str = "0.3.0"):
    """Decorator for functions requiring torchao.
    
    Args:
        min_version: Minimum required torchao version
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not check_torchao_available():
                raise RuntimeError(
                    f"{func.__name__} requires torchao. "
                    f"Install with: pip install torchao>=0.3.0 "
                    f"or pip install diffulex_edge[quant]"
                )
            
            # Version check
            if _TORCHAO_VERSION and _TORCHAO_VERSION != "unknown":
                try:
                    import packaging.version
                    if packaging.version.parse(_TORCHAO_VERSION) < packaging.version.parse(min_version):
                        raise RuntimeError(
                            f"{func.__name__} requires torchao >= {min_version}, "
                            f"found {_TORCHAO_VERSION}"
                        )
                except ImportError:
                    # packaging not available, skip version check
                    pass
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TorchAOWrapper:
    """Wrapper for torchao functionality with graceful degradation.
    
    This class provides a unified interface to torchao features,
    falling back gracefully when torchao is not available.
    """
    
    def __init__(self):
        self._available = check_torchao_available()
        self._version = get_torchao_version()
    
    @property
    def available(self) -> bool:
        """Check if torchao is available."""
        return self._available
    
    @property
    def version(self) -> Optional[str]:
        """Get torchao version."""
        return self._version
    
    def quantize_int4_weight_only(
        self,
        model: Any,
        group_size: int = 32,
    ) -> Any:
        """Apply INT4 weight-only quantization.
        
        Args:
            model: PyTorch model to quantize
            group_size: Group size for quantization (32, 64, 128, 256)
            
        Returns:
            Quantized model
            
        Raises:
            RuntimeError: If torchao is not available
            ValueError: If group_size is invalid
        """
        if not self._available:
            raise RuntimeError(
                "INT4 quantization requires torchao. "
                "Install with: pip install torchao>=0.3.0"
            )
        
        valid_group_sizes = [32, 64, 128, 256]
        if group_size not in valid_group_sizes:
            raise ValueError(
                f"Invalid group_size {group_size}. "
                f"Must be one of {valid_group_sizes}"
            )
        
        try:
            import torchao
            from torchao.quantization import quantize_, int4_weight_only
            
            # Apply INT4 weight-only quantization
            quantize_(model, int4_weight_only(group_size=group_size))
            return model
            
        except Exception as e:
            raise RuntimeError(f"INT4 quantization failed: {e}") from e
    
    def quantize_int8_dynamic(
        self,
        model: Any,
    ) -> Any:
        """Apply INT8 dynamic quantization using torchao.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model
        """
        if not self._available:
            raise RuntimeError(
                "torchao INT8 dynamic quantization requires torchao. "
                "Install with: pip install torchao>=0.3.0"
            )
        
        try:
            import torchao
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
            
            quantize_(model, int8_dynamic_activation_int8_weight())
            return model
            
        except Exception as e:
            raise RuntimeError(f"INT8 dynamic quantization failed: {e}") from e
    
    def quantize_int8_weight_only(
        self,
        model: Any,
    ) -> Any:
        """Apply INT8 weight-only quantization using torchao.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model
        """
        if not self._available:
            raise RuntimeError(
                "torchao INT8 weight-only quantization requires torchao. "
                "Install with: pip install torchao>=0.3.0"
            )
        
        try:
            import torchao
            from torchao.quantization import quantize_, int8_weight_only
            
            quantize_(model, int8_weight_only())
            return model
            
        except Exception as e:
            raise RuntimeError(f"INT8 weight-only quantization failed: {e}") from e


# Global wrapper instance
torchao_wrapper = TorchAOWrapper()


def get_torchao_wrapper() -> TorchAOWrapper:
    """Get the global torchao wrapper instance.
    
    Returns:
        TorchAOWrapper instance
    """
    return torchao_wrapper
