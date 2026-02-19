"""Model exporter for DiffuLex Edge.

Provides export capabilities for different dLLM model architectures
including Dream, LLaDA, SDAR, and FastdLLM V2.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json


@dataclass
class ModelExportConfig:
    """Configuration for model export.
    
    Attributes:
        model_type: Type of model (dream, llada, sdar, fast_dllm_v2)
        output_path: Path to save exported model
        quantization: Quantization mode (none, int8, int4)
        target_backend: Target backend (xnnpack, qnn, coreml, cpu)
        max_sequence_length: Maximum sequence length
        enable_kv_cache: Whether to enable KV cache
    """
    model_type: str = "fast_dllm_v2"
    output_path: Optional[Path] = None
    quantization: str = "none"
    target_backend: str = "cpu"
    max_sequence_length: int = 2048
    enable_kv_cache: bool = True


class BaseModelExporter(ABC):
    """Base class for model exporters.
    
    All model-specific exporters should inherit from this class.
    """
    
    def __init__(self, config: ModelExportConfig):
        self.config = config
        self.model_metadata = {}
    
    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for export.
        
        Args:
            model: Raw model instance
            
        Returns:
            Prepared model ready for export
        """
        pass
    
    @abstractmethod
    def get_sample_input(self) -> torch.Tensor:
        """Get sample input for tracing/export.
        
        Returns:
            Sample input tensor
        """
        pass
    
    def export_torchscript(self, model: nn.Module, output_path: Path) -> Path:
        """Export model to TorchScript format.
        
        Args:
            model: Prepared model
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        model.eval()
        sample_input = self.get_sample_input()
        
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(output_path))
            print(f"Exported TorchScript model to {output_path}")
            return output_path
        except Exception as e:
            print(f"TorchScript export failed: {e}")
            # Fallback to tracing
            traced = torch.jit.trace(model, sample_input)
            traced.save(str(output_path))
            print(f"Exported traced model to {output_path}")
            return output_path
    
    def export_onnx(self, model: nn.Module, output_path: Path) -> Path:
        """Export model to ONNX format.
        
        Args:
            model: Prepared model
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        model.eval()
        sample_input = self.get_sample_input()
        
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
        )
        print(f"Exported ONNX model to {output_path}")
        return output_path
    
    def export_metadata(self, output_path: Path) -> Path:
        """Export model metadata to JSON.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        metadata = {
            "model_type": self.config.model_type,
            "quantization": self.config.quantization,
            "target_backend": self.config.target_backend,
            "max_sequence_length": self.config.max_sequence_length,
            "enable_kv_cache": self.config.enable_kv_cache,
            **self.model_metadata,
        }
        
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Exported metadata to {metadata_path}")
        return metadata_path
    
    def export(self, model: nn.Module, output_dir: Path) -> Dict[str, Path]:
        """Export model in all supported formats.
        
        Args:
            model: Model to export
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prepared_model = self.prepare_model(model)
        
        results = {}
        
        # Export TorchScript
        try:
            ts_path = output_dir / f"{self.config.model_type}.pt"
            results["torchscript"] = self.export_torchscript(prepared_model, ts_path)
        except Exception as e:
            print(f"TorchScript export skipped: {e}")
        
        # Export ONNX
        try:
            onnx_path = output_dir / f"{self.config.model_type}.onnx"
            results["onnx"] = self.export_onnx(prepared_model, onnx_path)
        except Exception as e:
            print(f"ONNX export skipped: {e}")
        
        # Export metadata
        meta_path = output_dir / f"{self.config.model_type}"
        results["metadata"] = self.export_metadata(meta_path)
        
        return results


class FastdLLMV2Exporter(BaseModelExporter):
    """Exporter for FastdLLM V2 models."""
    
    def __init__(self, config: Optional[ModelExportConfig] = None):
        config = config or ModelExportConfig(model_type="fast_dllm_v2")
        super().__init__(config)
        self.model_metadata = {
            "supports_shift_logits": True,
            "supports_diffusion": True,
            "mask_token_id": 126336,
        }
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare FastdLLM V2 model for export."""
        model.eval()
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        """Get sample input for FastdLLM V2."""
        return torch.randint(0, 32000, (1, 32))  # [batch=1, seq_len=32]


class DreamModelExporter(BaseModelExporter):
    """Exporter for Dream models."""
    
    def __init__(self, config: Optional[ModelExportConfig] = None):
        config = config or ModelExportConfig(model_type="dream")
        super().__init__(config)
        self.model_metadata = {
            "supports_shift_logits": True,
            "supports_diffusion": True,
            "noise_schedule": "cosine",
            "num_diffusion_steps": 50,
        }
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare Dream model for export."""
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        """Get sample input for Dream model."""
        return torch.randint(0, 32000, (1, 32))


class LLaDAModelExporter(BaseModelExporter):
    """Exporter for LLaDA models."""
    
    def __init__(self, config: Optional[ModelExportConfig] = None):
        config = config or ModelExportConfig(model_type="llada")
        super().__init__(config)
        self.model_metadata = {
            "supports_shift_logits": True,
            "supports_diffusion": True,
            "mask_token_id": 126336,
            "confidence_threshold": 0.9,
        }
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare LLaDA model for export."""
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        """Get sample input for LLaDA model."""
        return torch.randint(0, 32000, (1, 32))


class SDARModelExporter(BaseModelExporter):
    """Exporter for SDAR models."""
    
    def __init__(self, config: Optional[ModelExportConfig] = None):
        config = config or ModelExportConfig(model_type="sdar")
        super().__init__(config)
        self.model_metadata = {
            "supports_shift_logits": True,
            "supports_diffusion": True,
            "diffusion_steps": 32,
            "schedule": "linear",
        }
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare SDAR model for export."""
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        """Get sample input for SDAR model."""
        return torch.randint(0, 50000, (1, 32))


class ModelExporterFactory:
    """Factory for creating model exporters.
    
    Usage:
        exporter = ModelExporterFactory.create("fast_dllm_v2")
        paths = exporter.export(model, output_dir)
    """
    
    _exporters = {
        "fast_dllm_v2": FastdLLMV2Exporter,
        "dream": DreamModelExporter,
        "llada": LLaDAModelExporter,
        "sdar": SDARModelExporter,
    }
    
    @classmethod
    def create(
        cls,
        model_type: str,
        config: Optional[ModelExportConfig] = None
    ) -> BaseModelExporter:
        """Create an exporter for the given model type.
        
        Args:
            model_type: Type of model to export
            config: Optional export configuration
            
        Returns:
            Model exporter instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._exporters:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls._exporters.keys())}"
            )
        
        if config is None:
            config = ModelExportConfig(model_type=model_type)
        
        return cls._exporters[model_type](config)
    
    @classmethod
    def register(
        cls,
        model_type: str,
        exporter_class: type
    ) -> None:
        """Register a new exporter type.
        
        Args:
            model_type: Model type identifier
            exporter_class: Exporter class to register
        """
        cls._exporters[model_type] = exporter_class
    
    @classmethod
    def list_supported_types(cls) -> List[str]:
        """List all supported model types.
        
        Returns:
            List of supported model type strings
        """
        return list(cls._exporters.keys())


def export_model(
    model: nn.Module,
    model_type: str,
    output_dir: Union[str, Path],
    config: Optional[ModelExportConfig] = None,
) -> Dict[str, Path]:
    """Convenience function to export a model.
    
    Args:
        model: Model to export
        model_type: Type of model
        output_dir: Output directory
        config: Optional export configuration
        
    Returns:
        Dictionary mapping format names to file paths
    """
    exporter = ModelExporterFactory.create(model_type, config)
    return exporter.export(model, Path(output_dir))


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Example: Export a mock model
    print("Model Exporter Example")
    print("=" * 50)
    
    # List supported types
    print("\nSupported model types:")
    for model_type in ModelExporterFactory.list_supported_types():
        print(f"  - {model_type}")
    
    # Create a simple mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32000, 256)
            self.head = nn.Linear(256, 32000)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            return self.head(x)
    
    model = MockModel()
    
    # Export
    print("\nExporting mock FastdLLM V2 model...")
    try:
        paths = export_model(
            model,
            model_type="fast_dllm_v2",
            output_dir="./test_export",
        )
        print(f"\nExported files:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
    except Exception as e:
        print(f"Export failed: {e}")
