"""XNNPACK CPU 后端实现."""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import BackendConfig, EdgeBackend, ExportResult

logger = logging.getLogger(__name__)


class XNNPACKBackend(EdgeBackend):
    """XNNPACK CPU 后端.
    
    适用于通用 ARM64/x86 设备的 CPU 推理后端。
    通过 XNNPACK 库提供优化的神经网络算子。
    """
    
    name = "xnnpack"
    
    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._partitioner = None
    
    def _validate_config(self):
        """验证 XNNPACK 配置."""
        valid_modes = ["dynamic", "static", "weight_only", "qat", None, ""]
        if self.config.quantization_mode not in valid_modes:
            logger.warning(
                f"Unknown quantization mode: {self.config.quantization_mode}, "
                f"valid modes: {valid_modes}"
            )
    
    def is_available(self) -> bool:
        """检查 XNNPACK 后端是否可用."""
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            return True
        except ImportError:
            return False
    
    def get_partitioner(self):
        """获取 XNNPACK 分区器."""
        if self._partitioner is None:
            try:
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                    XnnpackPartitioner,
                )
                
                # 创建分区器，支持量化配置
                partitioner_kwargs = {}
                
                # 如果使用量化，配置相关选项
                if self.config.quantize:
                    partitioner_kwargs["quantization_mode"] = self.config.quantization_mode
                
                self._partitioner = XnnpackPartitioner(**partitioner_kwargs)
                logger.info(f"Created XNNPACK partitioner with config: {partitioner_kwargs}")
                
            except ImportError as e:
                logger.error(f"Failed to import XNNPACK partitioner: {e}")
                raise RuntimeError("XNNPACK partitioner not available") from e
        
        return self._partitioner
    
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
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为 XNNPACK 格式.
        
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
            # 导入 ExecuTorch 相关模块
            from torch.export import export
            from executorch.exir import to_edge_transform_and_lower
            from executorch.exir.passes import MemoryPlanningPass
            from executorch.exir.program._program import ExecutorchBackendConfig
            
            logger.info(f"Starting XNNPACK export with config: {self.config}")
            
            # 1. 确保模型处于评估模式，禁用所有梯度
            model.eval()
            torch.set_grad_enabled(False)
            
            # 确保所有参数不需要梯度
            for param in model.parameters():
                param.requires_grad = False
            
            # 2. 准备模型（使用包装器如果需要）
            export_model = self._prepare_model_for_export(model)
            
            # 3. 导出为 ExportedProgram
            logger.info("Exporting to ExportedProgram...")
            
            if isinstance(example_inputs, dict):
                ep = export(export_model, (), example_inputs)
            else:
                if not isinstance(example_inputs, tuple):
                    example_inputs = (example_inputs,)
                ep = export(export_model, example_inputs)
            
            logger.info("ExportedProgram created successfully")
            
            # 4. 使用新的工作流：to_edge_transform_and_lower
            logger.info("Converting to Edge Dialect and lowering to XNNPACK...")
            partitioner = self.get_partitioner()
            edge = to_edge_transform_and_lower(
                ep,
                partitioner=[partitioner],
            )
            logger.info("Edge conversion and XNNPACK lowering successful")
            
            # 5. 生成 .pte 文件
            logger.info("Generating ExecuTorch program...")
            
            # Use default ExecuTorch backend config
            exec_config = ExecutorchBackendConfig()
            
            exec_prog = edge.to_executorch(exec_config)
            buffer = exec_prog.buffer
            
            logger.info(f"XNNPACK export successful, buffer size: {len(buffer)} bytes")
            
            # 收集元数据
            metadata = {
                "buffer_size_bytes": len(buffer),
                "backend": "xnnpack",
                "quantized": self.config.quantize,
                "quantization_mode": self.config.quantization_mode if self.config.quantize else None,
            }
            
            return ExportResult(
                success=True,
                buffer=buffer,
                metadata=metadata,
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"XNNPACK export failed: {error_msg}")
            
            # 检查是否是常见的 flatc 错误
            if "flatc" in error_msg.lower() or "WinError 2" in error_msg:
                error_msg = (
                    "FlatBuffer compiler (flatc) not available. "
                    "On Windows, this is expected. Use Linux/macOS for full .pte generation."
                )
            
            return ExportResult(
                success=False,
                error_message=error_msg,
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取后端元数据."""
        metadata = super().get_metadata()
        metadata.update({
            "supported_platforms": ["arm64", "x86_64", "aarch64"],
            "supports_quantization": True,
            "recommended_for": ["general_cpu", "android", "linux_embedded"],
        })
        return metadata


# 注册后端
from .base import BackendRegistry
BackendRegistry.register("xnnpack", XNNPACKBackend)
