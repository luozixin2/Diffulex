"""后端抽象基类."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class ExportResult:
    """导出结果."""
    
    success: bool
    buffer: Optional[bytes] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class BackendConfig:
    """后端配置."""
    
    # 通用配置
    backend_name: str = "xnnpack"
    quantize: bool = False
    quantization_mode: str = "weight_only"  # dynamic, static, weight_only, qat
    
    # 内存配置
    memory_planning: str = "greedy"  # greedy, sequential
    max_memory_mb: int = 2048
    
    # 序列长度配置
    max_seq_len: int = 2048
    batch_size: int = 1
    
    # 后端特定配置
    backend_options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.backend_options is None:
            self.backend_options = {}


class EdgeBackend(ABC):
    """端侧后端抽象基类."""
    
    name: str = "base"
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._validate_config()
    
    def _validate_config(self):
        """验证配置."""
        pass
    
    @abstractmethod
    def export(
        self,
        model: nn.Module,
        example_inputs: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为对应格式.
        
        Args:
            model: PyTorch 模型
            example_inputs: 示例输入
            config: 导出配置
            
        Returns:
            ExportResult: 导出结果
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用.
        
        Returns:
            bool: 是否可用
        """
        pass
    
    def get_partitioner(self):
        """获取分区器 (用于 ExecuTorch)."""
        raise NotImplementedError(f"{self.name} backend does not support partitioning")
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取后端元数据."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "config": {
                "quantize": self.config.quantize,
                "quantization_mode": self.config.quantization_mode,
                "max_seq_len": self.config.max_seq_len,
            }
        }


class BackendRegistry:
    """后端注册表."""
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """注册后端."""
        cls._backends[name.lower()] = backend_class
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取后端类."""
        return cls._backends.get(name.lower())
    
    @classmethod
    def list_backends(cls) -> Dict[str, type]:
        """列出所有注册的后端."""
        return cls._backends.copy()
    
    @classmethod
    def create(cls, name: str, config: Optional[BackendConfig] = None) -> Optional[EdgeBackend]:
        """创建后端实例."""
        backend_class = cls.get(name)
        if backend_class is None:
            return None
        return backend_class(config)
