"""XNNPACK CPU 后端实现."""

import logging
from typing import Any, Dict, Optional

from .base import BackendConfig, BackendRegistry, EdgeBackend, ExportResult

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
    
    def export(
        self,
        model,
        example_inputs,
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为 XNNPACK 格式.
        
        使用基类的通用导出流程。
        
        Args:
            model: PyTorch 模型
            example_inputs: 示例输入
            config: 导出配置
            
        Returns:
            ExportResult: 导出结果
        """
        logger.info(f"Starting XNNPACK export with config: {self.config}")
        
        # 调用基类的通用导出流程
        result = super().export(model, example_inputs, config)
        
        # 添加 XNNPACK 特定的元数据
        if result.success:
            result.metadata.update({
                "supported_platforms": ["arm64", "x86_64", "aarch64"],
                "supports_quantization": True,
                "recommended_for": ["general_cpu", "android", "linux_embedded"],
            })
        
        return result
    
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
BackendRegistry.register("xnnpack", XNNPACKBackend)
