"""Export configuration for DiffuLex Edge."""

import dataclasses
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import torch


class BackendType(Enum):
    """ExecuTorch backend types."""
    XNNPACK = "xnnpack"          # CPU backend, good for mobile/server
    QNN = "qnn"                  # Qualcomm backend
    COREML = "coreml"            # Apple CoreML
    MPS = "mps"                  # Apple Metal Performance Shaders
    VULKAN = "vulkan"            # Vulkan GPU backend
    REFERENCE = "reference"      # Reference implementation (slow, for testing)
    
    
class QuantizationType(Enum):
    """Quantization types for export."""
    NONE = "none"                # No quantization (FP32)
    DYNAMIC_INT8 = "dynamic_int8" # Dynamic INT8 quantization
    STATIC_INT8 = "static_int8"  # Static INT8 quantization
    WEIGHT_ONLY_INT8 = "weight_only_int8"  # Weight-only INT8
    FP16 = "fp16"                # FP16 (half precision)
    INT4 = "int4"                # INT4 weight-only quantization (via torchao)


@dataclasses.dataclass
class ExportConfig:
    """Configuration for model export.
    
    Attributes:
        # Model configuration
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Attention head dimension
        intermediate_size: FFN intermediate dimension
        
        # Export configuration
        backend: Target ExecuTorch backend
        quantization: Quantization type
        use_kv_cache: Enable KV cache optimization
        
        # IO configuration  
        output_path: Path to save .pte file
        calibration_data: Optional path to calibration data (for static quant)
        
        # Optimization flags
        enable_delegate: Use backend delegation
        memory_planning: Memory planning algorithm
        
        # Advanced options
        constant_methods: Dict of constant methods to preserve
        dynamic_shapes: Enable dynamic shape support
    """
    # Model config (will be inferred from checkpoint if not provided)
    max_seq_len: int = 2048
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # Defaults to num_heads
    head_dim: Optional[int] = None  # Defaults to hidden_size // num_heads
    intermediate_size: Optional[int] = None  # Defaults to 4 * hidden_size
    
    # Export config
    backend: BackendType = BackendType.REFERENCE  # Default for development
    quantization: QuantizationType = QuantizationType.NONE
    use_kv_cache: bool = True
    
    # Paths
    output_path: Path = Path("model.pte")
    checkpoint_path: Optional[Path] = None
    calibration_data_path: Optional[Path] = None
    
    # Optimization
    enable_delegate: bool = True
    memory_planning: str = "greedy"  # "greedy" or "linear_scan"
    
    # Advanced
    constant_methods: Optional[Dict[str, Any]] = None
    dynamic_shapes: bool = False  # Static shapes are more efficient for Edge
    
    def __post_init__(self):
        """Set derived values and validate."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
            
        # Ensure paths are Path objects
        self.output_path = Path(self.output_path)
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.calibration_data_path:
            self.calibration_data_path = Path(self.calibration_data_path)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dict."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        data = dataclasses.asdict(self)
        # Convert Path objects to strings
        for key in ['output_path', 'checkpoint_path', 'calibration_data_path']:
            if data[key] is not None:
                data[key] = str(data[key])
        # Convert Enum to string
        data['backend'] = self.backend.value
        data['quantization'] = self.quantization.value
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ExportConfig":
        """Deserialize from JSON string."""
        import json
        data = json.loads(json_str)
        # Convert strings back to Path/Enum
        for key in ['output_path', 'checkpoint_path', 'calibration_data_path']:
            if data.get(key) is not None:
                data[key] = Path(data[key])
        data['backend'] = BackendType(data['backend'])
        data['quantization'] = QuantizationType(data['quantization'])
        return cls(**data)


@dataclasses.dataclass
class ExportResult:
    """Result of model export."""
    success: bool
    output_path: Optional[Path] = None
    file_size_mb: Optional[float] = None
    compilation_time_sec: Optional[float] = None
    error_message: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.success:
            return f"ExportResult(success=True, path={self.output_path}, size={self.file_size_mb:.1f}MB)"
        else:
            return f"ExportResult(success=False, error={self.error_message})"
