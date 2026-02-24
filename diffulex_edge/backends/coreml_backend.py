"""Apple CoreML / Neural Engine 后端实现."""

import logging
import sys
from typing import Any, Dict, Optional

from .base import BackendConfig, BackendRegistry, EdgeBackend, ExportResult

logger = logging.getLogger(__name__)


class CoreMLBackend(EdgeBackend):
    """Apple CoreML Neural Engine 后端.
    
    适用于 Apple 设备的 ANE (Apple Neural Engine) 推理后端。
    利用 Apple Silicon 的专用神经网络加速器。
    
    要求:
        - Apple Silicon 设备 (A12 Bionic 或更高)
        - macOS 14.0+ / iOS 17.0+
        - coremltools 已安装
    """
    
    name = "coreml"
    
    # CoreML 支持的计算单元
    COMPUTE_UNITS = {
        "all": "all",  # 使用所有可用单元
        "cpu_only": "cpuOnly",
        "cpu_and_gpu": "cpuAndGPU",
        "cpu_and_ane": "cpuAndNeuralEngine",
        "ane_only": "neuralEngineOnly",
    }
    
    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._partitioner = None
        self._compute_unit = "cpu_and_ane"
        self._minimum_deployment_target = None
    
    def _validate_config(self):
        """验证 CoreML 配置."""
        if self.config.backend_options:
            # 计算单元配置
            compute_unit = self.config.backend_options.get("compute_unit", "cpu_and_ane")
            if compute_unit in self.COMPUTE_UNITS:
                self._compute_unit = compute_unit
            else:
                logger.warning(f"Unknown compute_unit: {compute_unit}, using cpu_and_ane")
            
            # 最低部署目标
            self._minimum_deployment_target = self.config.backend_options.get(
                "minimum_deployment_target", "iOS17"
            )
    
    def is_available(self) -> bool:
        """检查 CoreML 后端是否可用."""
        try:
            # CoreML 只在 macOS/iOS 上可用
            if sys.platform != "darwin":
                return False
            
            from executorch.backends.apple.coreml.partition.coreml_partitioner import (
                CoreMLPartitioner,
            )
            return True
        except ImportError:
            return False
    
    def get_partitioner(self):
        """获取 CoreML 分区器."""
        if self._partitioner is None:
            try:
                from executorch.backends.apple.coreml.partition.coreml_partitioner import (
                    CoreMLPartitioner,
                )
                
                # 配置分区器
                partitioner_kwargs = {
                    "compute_unit": self._compute_unit,
                }
                
                if self._minimum_deployment_target:
                    partitioner_kwargs["minimum_deployment_target"] = self._minimum_deployment_target
                
                self._partitioner = CoreMLPartitioner(**partitioner_kwargs)
                logger.info(f"Created CoreML partitioner with compute_unit: {self._compute_unit}")
                
            except ImportError as e:
                logger.error(f"Failed to import CoreML partitioner: {e}")
                raise RuntimeError("CoreML partitioner not available") from e
        
        return self._partitioner
    
    def export(
        self,
        model,
        example_inputs,
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为 CoreML 格式.
        
        Args:
            model: PyTorch 模型
            example_inputs: 示例输入
            config: 导出配置
            
        Returns:
            ExportResult: 导出结果
        """
        # 检查平台
        if sys.platform != "darwin":
            return ExportResult(
                success=False,
                error_message="CoreML backend is only supported on macOS/iOS.",
            )
        
        logger.info(f"Starting CoreML export with compute_unit: {self._compute_unit}")
        
        # 调用基类的通用导出流程
        result = super().export(model, example_inputs, config)
        
        # 添加 CoreML 特定的元数据
        if result.success:
            result.metadata.update({
                "compute_unit": self._compute_unit,
                "supported_platforms": ["macos", "ios", "ipados", "watchos"],
                "supported_chips": ["A12+", "M1+", "A14+", "M2+"],
                "supports_quantization": True,
                "recommended_quantization": "static",
                "recommended_for": ["apple_ane", "ios", "macos"],
            })
        
        return result
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """获取执行指标 (仅在 Apple 设备上可用).
        
        Returns:
            Dict 包含 ANE 利用率等指标
        """
        # 这需要实际的 CoreML 运行时支持
        # 作为占位符返回空指标
        return {
            "ane_utilization": 0,
            "cpu_utilization": 0,
            "gpu_utilization": 0,
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取后端元数据."""
        metadata = super().get_metadata()
        metadata.update({
            "compute_units": list(self.COMPUTE_UNITS.keys()),
            "supported_platforms": ["macos", "ios", "ipados", "watchos"],
            "supported_chips": ["A12+", "M1+", "A14+", "M2+"],
            "supports_quantization": True,
            "recommended_quantization": "static",
            "recommended_for": ["apple_ane", "ios", "macos"],
        })
        return metadata


# 注册后端
BackendRegistry.register("coreml", CoreMLBackend)
