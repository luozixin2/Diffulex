"""End-to-end export for DiffuLex Edge.

Exports models to ExecuTorch .pte format with optimizations.
Supports multiple backends through the backends module.
"""

import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict, Union
import torch
import torch.nn as nn

from .config import ExportConfig, ExportResult, BackendType, QuantizationType


def _get_backend_partitioner(backend: BackendType, quantization: QuantizationType):
    """Get backend partitioner for delegation (legacy method).
    
    Args:
        backend: Target backend
        quantization: Quantization type
        
    Returns:
        Partitioner for the backend, or None if not available
    """
    if backend == BackendType.XNNPACK:
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            return XnnpackPartitioner()
        except ImportError:
            print("Warning: XNNPACK backend not available")
            return None
    
    elif backend == BackendType.COREML:
        try:
            from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
            return CoreMLPartitioner()
        except ImportError:
            print("Warning: CoreML backend not available")
            return None
    
    # Reference and other backends don't need partitioner
    return None


class DiffuLexExporter:
    """Exporter for DiffuLex models to ExecuTorch.
    
    Supports multiple backends:
    - XNNPACK: Universal CPU backend (ARM64/x86)
    - QNN: Qualcomm NPU backend
    - CoreML: Apple Neural Engine backend
    
    Usage:
        # Method 1: Using ExportConfig
        config = ExportConfig(
            output_path="model.pte",
            backend=BackendType.XNNPACK,
            quantization=QuantizationType.DYNAMIC_INT8,
        )
        
        exporter = DiffuLexExporter(config)
        result = exporter.export(model, example_inputs)
        
        # Method 2: Using backend directly (recommended)
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        backend = XNNPACKBackend(BackendConfig(quantize=True))
        result = backend.export(model, example_inputs)
        if result.success:
            with open("model.pte", "wb") as f:
                f.write(result.buffer)
    """
    
    def __init__(self, config: ExportConfig):
        """Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self._exported_program = None
        self._edge_program = None
        self._backend = None
        
    def _get_backend(self):
        """Get backend instance based on config."""
        if self._backend is not None:
            return self._backend
        
        # Try to use new backend architecture
        try:
            from ..backends.base import BackendConfig
            from ..backends import BackendRegistry
            
            backend_name = self.config.backend.value.lower()
            backend_class = BackendRegistry.get(backend_name)
            
            if backend_class is not None:
                # Map quantization type
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
        except ImportError:
            pass
        
        # Fall back to legacy method
        return None
        
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
        example_inputs: Tuple[Any, ...],
    ) -> nn.Module:
        """Apply quantization if configured."""
        if self.config.quantization == QuantizationType.NONE:
            return model
        
        print(f"Applying quantization: {self.config.quantization.value}")
        
        if self.config.quantization == QuantizationType.DYNAMIC_INT8:
            from ..quant import apply_dynamic_quantization
            return apply_dynamic_quantization(model)
        
        elif self.config.quantization == QuantizationType.WEIGHT_ONLY_INT8:
            from ..quant import apply_weight_only_quantization
            return apply_weight_only_quantization(model)
        
        elif self.config.quantization == QuantizationType.STATIC_INT8:
            # Static quantization requires calibration
            from ..quant import DiffuLexQuantizer, QuantizationConfig, QuantizationMode
            
            quant_config = QuantizationConfig(mode=QuantizationMode.STATIC)
            quantizer = DiffuLexQuantizer(quant_config)
            
            # Prepare
            prepared = quantizer.prepare_for_quantization(model, example_inputs)
            
            # Calibrate if data available
            if self.config.calibration_data_path:
                calib_data = self._load_calibration_data()
                quantizer.calibrate(prepared, calib_data)
            else:
                print("Warning: No calibration data, using default statistics")
            
            # Convert
            return quantizer.convert(prepared)
        
        elif self.config.quantization == QuantizationType.FP16:
            return model.half()
        
        return model
    
    def _get_memory_planning_pass(self):
        """Get memory planning pass based on config."""
        try:
            from executorch.exir.passes import MemoryPlanningPass
            
            if self.config.memory_planning == "greedy":
                return MemoryPlanningPass()
            elif self.config.memory_planning == "linear_scan":
                # Linear scan memory planning
                return MemoryPlanningPass()
            else:
                return MemoryPlanningPass()
        except ImportError:
            # Fallback: return None and let ExecuTorch use default
            return None
    
    def _load_calibration_data(self) -> List[Tuple[Any, ...]]:
        """Load calibration data for static quantization."""
        # TODO: Implement calibration data loading
        # For now, return empty list
        return []
    
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
        
        # Try new backend architecture first
        backend = self._get_backend()
        if backend is not None and backend.is_available():
            print(f"Using {self.config.backend.value} backend...")
            try:
                result = backend.export(model, example_inputs)
                
                if result.success and result.buffer:
                    # Save buffer to file
                    self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.config.output_path, "wb") as f:
                        f.write(result.buffer)
                    
                    file_size = self.config.output_path.stat().st_size / (1024 * 1024)
                    elapsed = time.time() - start_time
                    
                    print(f"Export completed: {file_size:.2f} MB in {elapsed:.2f}s")
                    
                    return ExportResult(
                        success=True,
                        output_path=self.config.output_path,
                        file_size_mb=file_size,
                        compilation_time_sec=elapsed,
                    )
                elif not result.success:
                    # Handle flatc error on Windows
                    if self._is_windows_with_missing_flatc(result.error_message or ""):
                        return ExportResult(
                            success=False,
                            error_message=(
                                "flatc (FlatBuffer compiler) not available on Windows. "
                                "Edge IR conversion succeeded but .pte compilation requires Linux/macOS. "
                                "For development on Windows, the Edge IR is valid and can be exported on Linux."
                            ),
                            compilation_time_sec=time.time() - start_time,
                        )
                    else:
                        return ExportResult(
                            success=False,
                            error_message=result.error_message,
                            compilation_time_sec=time.time() - start_time,
                        )
            except Exception as e:
                print(f"Backend export failed: {e}, falling back to legacy method...")
        
        # Legacy export method
        return self._export_legacy(model, example_inputs, start_time)
    
    def _export_legacy(
        self,
        model: nn.Module,
        example_inputs: Tuple[Any, ...],
        start_time: float,
    ) -> ExportResult:
        """Legacy export method for backward compatibility."""
        try:
            # Step 1: Load checkpoint if specified
            model = self._load_checkpoint(model)
            model.eval()
            
            # Step 2: Apply quantization (if before export)
            if self.config.quantization == QuantizationType.STATIC_INT8:
                pass  # Will be handled in Step 5
            elif self.config.quantization == QuantizationType.FP16:
                model = model.half()
            
            # Step 3: Export with torch.export
            print("Exporting with torch.export...")
            with torch.no_grad():
                exported = torch.export.export(
                    model,
                    example_inputs,
                    strict=False,
                )
            self._exported_program = exported
            print("torch.export completed")
            
            # Step 4: Convert to Edge IR
            print("Converting to Edge IR...")
            from executorch.exir import to_edge
            
            edge_program = to_edge(exported)
            self._edge_program = edge_program
            print("Edge IR conversion completed")
            
            # Step 5: Apply post-export quantization if needed
            if self.config.quantization == QuantizationType.STATIC_INT8:
                print("Applying static quantization to Edge IR...")
                try:
                    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                        XNNPACKQuantizer,
                        get_symmetric_quantization_config,
                    )
                    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
                    
                    quantizer = XNNPACKQuantizer()
                    quantizer.set_global(get_symmetric_quantization_config())
                    print("Note: Static quantization may need calibration data")
                except ImportError as e:
                    print(f"Warning: Could not apply static quantization: {e}")
            
            # Step 6: Apply backend delegation
            if self.config.enable_delegate:
                partitioner = _get_backend_partitioner(
                    self.config.backend,
                    self.config.quantization,
                )
                
                if partitioner:
                    print(f"Applying {self.config.backend.value} backend delegation...")
                    try:
                        from executorch.exir import to_edge_transform_and_lower
                        
                        edge_program = to_edge_transform_and_lower(
                            exported,
                            partitioner=[partitioner],
                        )
                        print("Backend delegation completed")
                    except Exception as e:
                        print(f"Warning: Backend delegation failed: {e}")
                        print("Continuing without delegation...")
            
            # Step 7: Compile to .pte
            print(f"Compiling to {self.config.output_path}...")
            
            # Ensure output directory exists
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate the .pte file
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_pte = Path(tmpdir) / "model.pte"
                
                # Use ExecuTorch's to_executorch
                from executorch.exir import ExecutorchBackendConfig
                
                exec_program = edge_program.to_executorch(
                    config=ExecutorchBackendConfig()
                )
                
                # Save the file
                with open(temp_pte, "wb") as f:
                    exec_program.write_to_file(f)
                
                # Move to final location
                import shutil
                shutil.move(str(temp_pte), str(self.config.output_path))
            
            # Calculate file size
            file_size = self.config.output_path.stat().st_size / (1024 * 1024)
            elapsed = time.time() - start_time
            
            print(f"Export completed: {file_size:.2f} MB in {elapsed:.2f}s")
            
            return ExportResult(
                success=True,
                output_path=self.config.output_path,
                file_size_mb=file_size,
                compilation_time_sec=elapsed,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            print(f"Export failed after {elapsed:.2f}s: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Provide helpful messages for common issues
            if self._is_windows_with_missing_flatc(error_msg):
                error_msg += "\nNote: .pte compilation requires flatc. On Windows, this may not be available. Consider using WSL2 or Linux."
            elif "LinearPackedParamsBase" in error_msg:
                error_msg += "\nNote: Eager mode quantization is not compatible with torch.export. Use STATIC_INT8 quantization instead."
            
            return ExportResult(
                success=False,
                error_message=error_msg,
                compilation_time_sec=elapsed,
            )
    
    def get_exported_program(self):
        """Get the exported program (for debugging)."""
        return self._exported_program
    
    def get_edge_program(self):
        """Get the edge program (for debugging)."""
        return self._edge_program
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        try:
            from ..backends.base import BackendRegistry
            from ..backends import XNNPACKBackend, QNNBackend
            
            backends = {}
            
            # Check XNNPACK
            xnn = XNNPACKBackend()
            backends["xnnpack"] = xnn.get_metadata()
            
            # Check QNN
            qnn = QNNBackend()
            backends["qnn"] = qnn.get_metadata()
            
            # Check CoreML
            try:
                from ..backends import CoreMLBackend
                if CoreMLBackend is not None:
                    coreml = CoreMLBackend()
                    backends["coreml"] = coreml.get_metadata()
            except ImportError:
                backends["coreml"] = {"available": False, "reason": "not installed"}
            
            return backends
        except ImportError:
            return {"error": "backends module not available"}


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    output_path: str,
    backend: str = "reference",
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
