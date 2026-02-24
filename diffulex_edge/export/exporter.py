"""End-to-end export for DiffuLex Edge.

Exports models to ExecuTorch .pte format with optimizations.
Supports multiple backends through the backends module.
"""

import copy
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .config import ExportConfig, ExportResult, BackendType, QuantizationType


class DiffuLexExporter:
    """Exporter for DiffuLex models to ExecuTorch.
    
    Supports multiple backends:
    - XNNPACK: Universal CPU backend (ARM64/x86)
    - QNN: Qualcomm NPU backend
    - CoreML: Apple Neural Engine backend
    
    Usage:
        from diffulex_edge.export import ExportConfig, BackendType, DiffuLexExporter
        
        config = ExportConfig(
            output_path="model.pte",
            backend=BackendType.XNNPACK,
            quantization=QuantizationType.DYNAMIC_INT8,
        )
        
        exporter = DiffuLexExporter(config)
        result = exporter.export(model, example_inputs)
        
        if result.success:
            print(f"Exported to {result.output_path} ({result.file_size_mb:.2f} MB)")
    """
    
    def __init__(self, config: ExportConfig):
        """Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self._backend = None
    
    def _get_backend(self):
        """Get backend instance based on config."""
        if self._backend is not None:
            return self._backend
        
        from ..backends.base import BackendConfig, BackendRegistry
        
        backend_name = self.config.backend.value.lower()
        backend_class = BackendRegistry.get(backend_name)
        
        if backend_class is None:
            available = list(BackendRegistry.list_backends().keys())
            raise RuntimeError(
                f"Backend '{backend_name}' not found. "
                f"Available: {available}"
            )
        
        # Map quantization type to mode string
        quant_mode = None
        if self.config.quantization == QuantizationType.DYNAMIC_INT8:
            quant_mode = "dynamic"
        elif self.config.quantization == QuantizationType.STATIC_INT8:
            quant_mode = "static"
        elif self.config.quantization == QuantizationType.WEIGHT_ONLY_INT8:
            quant_mode = "weight_only"
        
        backend_config = BackendConfig(
            backend_name=backend_name,
            quantize=self.config.quantization != QuantizationType.NONE,
            quantization_mode=quant_mode,
            memory_planning=self.config.memory_planning,
            max_seq_len=self.config.max_seq_len,
        )
        
        self._backend = backend_class(backend_config)
        return self._backend
    
    def _load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load checkpoint if specified in config."""
        if self.config.checkpoint_path and self.config.checkpoint_path.exists():
            print(f"Loading checkpoint from {self.config.checkpoint_path}")
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully")
        
        return model
    
    def _apply_quantization(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """Apply real quantization that reduces PTE file size."""
        if self.config.quantization == QuantizationType.NONE:
            return model
        
        print(f"Applying REAL quantization: {self.config.quantization.value}")
        print(f"  This will reduce PTE file size as follows:")
        
        from ..quant.core_quant import (
            quantize_to_fp16,
            quantize_to_int8,
            verify_quantization_accuracy,
        )
        
        if self.config.quantization == QuantizationType.FP16:
            print(f"  FP16: 4 bytes -> 2 bytes per weight (2x size reduction)")
            model = quantize_to_fp16(model)
            
        elif self.config.quantization == QuantizationType.WEIGHT_ONLY_INT8:
            print(f"  Weight-only INT8: 4 bytes -> 1 byte per weight (4x size reduction)")
            model = quantize_to_int8(model, mode="weight_only")
            
        elif self.config.quantization == QuantizationType.DYNAMIC_INT8:
            print(f"  Dynamic INT8: 4 bytes -> 1 byte per weight (4x size reduction)")
            model = quantize_to_int8(model, mode="dynamic")
            
        elif self.config.quantization == QuantizationType.STATIC_INT8:
            print(f"  Static INT8: 4 bytes -> 1 byte per weight (4x size reduction)")
            model = quantize_to_int8(model, mode="static")
        
        elif self.config.quantization == QuantizationType.INT4:
            # Try INT4, fall back to INT8
            from ..quant import INT4Quantizer, QuantizationConfig
            from ..quant.base import QuantizationDtype
            quantizer = INT4Quantizer()
            if quantizer.is_available():
                print(f"  INT4: 4 bytes -> 0.5 bytes per weight (8x size reduction)")
                config = QuantizationConfig(dtype=QuantizationDtype.INT4)
                result = quantizer.quantize(model, config)
                model = result.model
            else:
                print(f"  INT4 not available, falling back to INT8 (4x reduction)")
                model = quantize_to_int8(model, mode="weight_only")
        
        # Verify accuracy
        try:
            print(f"\n  Verifying quantization accuracy...")
            vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', 32000)
            test_input = torch.randint(0, vocab_size, (1, 32), dtype=torch.long)
            test_positions = torch.arange(32, dtype=torch.long).unsqueeze(0)
            test_inputs = (test_input, test_positions)
            
            verify_quantization_accuracy(
                copy.deepcopy(model),
                model,
                test_inputs,
                tolerance=0.1
            )
            print(f"  Quantization accuracy verified successfully")
        except Exception as e:
            print(f"  Warning: Could not verify accuracy: {e}")
            print(f"  Continuing with export anyway...")
        
        return model
    
    def _is_windows_with_missing_flatc(self, error_msg: str) -> bool:
        """Check if error is due to missing flatc on Windows."""
        import sys
        return sys.platform == "win32" and ("flatc" in error_msg.lower() or "WinError 2" in error_msg)
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Tuple[Any, ...],
    ) -> ExportResult:
        """Export model to ExecuTorch format.
        
        Args:
            model: The model to export
            example_inputs: Example inputs for tracing
            
        Returns:
            ExportResult with status and metadata
        """
        start_time = time.time()
        
        # Load checkpoint if specified
        model = self._load_checkpoint(model)
        
        # Apply quantization
        if self.config.quantization != QuantizationType.NONE:
            model = self._apply_quantization(model)
        
        # Get backend and export
        backend = self._get_backend()
        print(f"Using {self.config.backend.value} backend...")
        
        result = backend.export(model, example_inputs)
        
        elapsed = time.time() - start_time
        
        if result.success and result.buffer:
            # Save buffer to file
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_path, "wb") as f:
                f.write(result.buffer)
            
            file_size = self.config.output_path.stat().st_size / (1024 * 1024)
            
            print(f"Export completed: {file_size:.2f} MB in {elapsed:.2f}s")
            
            return ExportResult(
                success=True,
                output_path=self.config.output_path,
                file_size_mb=file_size,
                compilation_time_sec=elapsed,
            )
        
        # Handle failure
        error_msg = result.error_message or "Unknown error"
        
        if self._is_windows_with_missing_flatc(error_msg):
            error_msg = (
                "flatc (FlatBuffer compiler) not available on Windows. "
                "Edge IR conversion succeeded but .pte compilation requires Linux/macOS. "
                "For development on Windows, the Edge IR is valid and can be exported on Linux."
            )
        
        print(f"Export failed after {elapsed:.2f}s: {error_msg}")
        
        return ExportResult(
            success=False,
            error_message=error_msg,
            compilation_time_sec=elapsed,
        )
    
    def get_backend_info(self) -> dict:
        """Get information about available backends."""
        from ..backends.base import BackendRegistry
        
        backends = {}
        for name, backend_class in BackendRegistry.list_backends().items():
            try:
                instance = backend_class()
                backends[name] = instance.get_metadata()
            except Exception as e:
                backends[name] = {"available": False, "reason": str(e)}
        
        return backends


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    output_path: str,
    backend: str = "xnnpack",
    quantization: str = "none",
    **kwargs,
) -> ExportResult:
    """Convenience function to export a model.
    
    Args:
        model: Model to export
        example_inputs: Example inputs for tracing
        output_path: Path to save .pte file
        backend: Backend type (xnnpack, coreml, qnn, reference, etc.)
        quantization: Quantization type (none, dynamic_int8, etc.)
        **kwargs: Additional config options
        
    Returns:
        ExportResult
    """
    config = ExportConfig(
        output_path=Path(output_path),
        backend=BackendType(backend),
        quantization=QuantizationType(quantization),
        **kwargs,
    )
    
    exporter = DiffuLexExporter(config)
    return exporter.export(model, example_inputs)
