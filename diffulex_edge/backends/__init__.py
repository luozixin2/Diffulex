"""DiffuLex Edge 后端支持模块.

提供多后端导出和推理支持:
- XNNPACK: 通用 CPU 后端 (ARM64/x86)
- QNN: Qualcomm NPU 后端
- CoreML: Apple Neural Engine 后端
"""

from .base import EdgeBackend, ExportResult, BackendConfig, BackendRegistry
from .xnnpack_backend import XNNPACKBackend
from .qnn_backend import QNNBackend

__all__ = [
    "EdgeBackend",
    "ExportResult",
    "BackendConfig",
    "BackendRegistry",
    "XNNPACKBackend",
    "QNNBackend",
]

# 可选后端 (需要额外依赖)
try:
    from .coreml_backend import CoreMLBackend
    __all__.append("CoreMLBackend")
except ImportError:
    CoreMLBackend = None
