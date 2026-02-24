"""后端抽象基类."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
    """端侧后端抽象基类.
    
    提供了通用的导出流程，子类只需实现特定的配置和分区器。
    """
    
    name: str = "base"
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._validate_config()
    
    def _validate_config(self):
        """验证配置."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用.
        
        Returns:
            bool: 是否可用
        """
        pass
    
    def get_partitioner(self):
        """获取分区器 (用于 ExecuTorch).
        
        子类可以覆盖此方法以提供特定的分区器。
        
        Returns:
            Partitioner 实例或 None
        """
        return None
    
    def _prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """准备模型用于导出.
        
        如果模型支持导出（有 get_export_wrapper 方法），使用包装器。
        否则直接使用原始模型。
        
        Args:
            model: 原始模型
            
        Returns:
            准备好用于导出的模型
        """
        # Check if model provides an export wrapper
        if hasattr(model, 'get_export_wrapper'):
            wrapper = model.get_export_wrapper()
            if wrapper is not None:
                logger.info(f"Using model's export wrapper: {type(wrapper).__name__}")
                return wrapper
        
        # Check for legacy forward_export (backward compatibility)
        if hasattr(model, 'forward_export'):
            logger.info("Detected forward_export method, using generic ExportWrapper")
            from ..model.wrapper import ExportWrapper
            return ExportWrapper(model)
        
        # Use model as-is
        return model
    
    def _torch_export(
        self,
        model: nn.Module,
        example_inputs: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    ) -> Any:
        """执行 torch.export.export.
        
        Args:
            model: 准备好的模型
            example_inputs: 示例输入
            
        Returns:
            ExportedProgram
        """
        from torch.export import export
        
        if isinstance(example_inputs, dict):
            return export(model, (), example_inputs)
        else:
            if not isinstance(example_inputs, tuple):
                example_inputs = (example_inputs,)
            return export(model, example_inputs)
    
    def _to_edge(
        self,
        exported_program: Any,
    ) -> Any:
        """转换为 Edge Dialect 并下放到后端.
        
        Args:
            exported_program: ExportedProgram
            
        Returns:
            EdgeProgram
        """
        from executorch.exir import to_edge_transform_and_lower
        
        partitioner = self.get_partitioner()
        if partitioner is not None:
            return to_edge_transform_and_lower(
                exported_program,
                partitioner=[partitioner],
            )
        else:
            # 没有分区器时，只转换为 Edge IR
            from executorch.exir import to_edge
            return to_edge(exported_program)
    
    def _to_executorch(
        self,
        edge_program: Any,
    ) -> Any:
        """生成 ExecuTorch 程序.
        
        Args:
            edge_program: EdgeProgram
            
        Returns:
            ExecuTorchProgram
        """
        from executorch.exir.program._program import ExecutorchBackendConfig
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
        
        # 配置内存规划算法
        if self.config.memory_planning == "greedy":
            algo = greedy
        else:
            algo = MemoryPlanningAlgorithmSuite()
        
        memory_planning_pass = MemoryPlanningPass(
            memory_planning_algo=algo,
            alloc_graph_input=False,
            alloc_graph_output=False,
        )
        
        exec_config = ExecutorchBackendConfig(
            memory_planning_pass=memory_planning_pass,
        )
        
        return edge_program.to_executorch(exec_config)
    
    def _pre_export(self, model: nn.Module) -> None:
        """导出前的准备工作.
        
        Args:
            model: 原始模型
        """
        model.eval()
        torch.set_grad_enabled(False)
        
        # 确保所有参数不需要梯度
        for param in model.parameters():
            param.requires_grad = False
    
    def _create_success_result(
        self,
        buffer: bytes,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """创建成功的导出结果.
        
        Args:
            buffer: 导出的字节缓冲区
            extra_metadata: 额外的元数据
            
        Returns:
            ExportResult
        """
        metadata = {
            "buffer_size_bytes": len(buffer),
            "backend": self.name,
            "quantized": self.config.quantize,
            "quantization_mode": self.config.quantization_mode if self.config.quantize else None,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return ExportResult(
            success=True,
            buffer=buffer,
            metadata=metadata,
        )
    
    def _create_error_result(
        self,
        error_message: str,
    ) -> ExportResult:
        """创建失败的导出结果.
        
        Args:
            error_message: 错误信息
            
        Returns:
            ExportResult
        """
        # 检查是否是常见的 flatc 错误
        if "flatc" in error_message.lower() or "WinError 2" in error_message:
            error_message = (
                "FlatBuffer compiler (flatc) not available. "
                "On Windows, this is expected. Use Linux/macOS for full .pte generation."
            )
        
        logger.error(f"{self.name} export failed: {error_message}")
        return ExportResult(
            success=False,
            error_message=error_message,
        )
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为后端特定格式.
        
        通用导出流程，子类可以覆盖以添加特定的前后处理。
        
        Args:
            model: PyTorch 模型
            example_inputs: 示例输入
            config: 导出配置 (可选，覆盖实例配置)
            
        Returns:
            ExportResult: 导出结果
        """
        if config is not None:
            self.config = config
        
        try:
            logger.info(f"Starting {self.name} export...")
            
            # 1. 准备模型
            self._pre_export(model)
            export_model = self._prepare_model_for_export(model)
            
            # 2. 导出为 ExportedProgram
            logger.info("Exporting to ExportedProgram...")
            ep = self._torch_export(export_model, example_inputs)
            logger.info("ExportedProgram created successfully")
            
            # 3. 转换为 Edge Dialect 并下放到后端
            logger.info(f"Converting to Edge Dialect and lowering to {self.name}...")
            edge = self._to_edge(ep)
            logger.info("Edge conversion successful")
            
            # 4. 生成 .pte 文件
            logger.info("Generating ExecuTorch program...")
            exec_prog = self._to_executorch(edge)
            buffer = exec_prog.buffer
            
            logger.info(f"{self.name} export successful, buffer size: {len(buffer)} bytes")
            
            return self._create_success_result(buffer)
            
        except Exception as e:
            return self._create_error_result(str(e))
    
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
