"""Qualcomm QNN NPU 后端实现."""

import logging
import sys
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .base import BackendConfig, EdgeBackend, ExportResult

logger = logging.getLogger(__name__)


class QNNBackend(EdgeBackend):
    """Qualcomm QNN NPU 后端.
    
    适用于 Qualcomm 设备的 NPU 推理后端。
    利用 Hexagon Tensor Processor (HTP) 或 DSP 进行加速。
    
    要求:
        - Qualcomm 设备 (Snapdragon 8xx/7xx/6xx 系列)
        - QNN SDK 已安装
        - Android/Linux 环境
    """
    
    name = "qnn"
    
    # QNN 支持的量化模式
    SUPPORTED_QUANT_MODES = ["static", "weight_only"]
    
    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._partitioner = None
        self._soc_model = None
    
    def _validate_config(self):
        """验证 QNN 配置."""
        # QNN 建议使用静态量化或权重量化
        if self.config.quantize and self.config.quantization_mode not in self.SUPPORTED_QUANT_MODES:
            logger.warning(
                f"QNN recommends static or weight_only quantization, "
                f"got: {self.config.quantization_mode}"
            )
        
        # 获取 SoC 模型配置
        if self.config.backend_options:
            self._soc_model = self.config.backend_options.get("soc_model")
    
    def is_available(self) -> bool:
        """检查 QNN 后端是否可用."""
        try:
            # QNN 只在 Linux/Android 上可用
            if sys.platform == "win32":
                return False
            
            from executorch.backends.qualcomm.partition.qnn_partitioner import (
                QnnPartitioner,
            )
            return True
        except ImportError:
            return False
    
    def get_partitioner(self):
        """获取 QNN 分区器."""
        if self._partitioner is None:
            try:
                from executorch.backends.qualcomm.partition.qnn_partitioner import (
                    QnnPartitioner,
                )
                from executorch.backends.qualcomm.serialization.qnn_definitions import (
                    QcomChipset,
                )
                
                # 配置 SoC 模型
                soc_model = self._get_soc_model()
                
                # 创建分区器
                partitioner_kwargs = {
                    "soc_model": soc_model,
                }
                
                # 添加量化相关配置
                if self.config.quantize:
                    partitioner_kwargs["quantization_mode"] = self.config.quantization_mode
                
                self._partitioner = QnnPartitioner(**partitioner_kwargs)
                logger.info(f"Created QNN partitioner for SoC: {soc_model}")
                
            except ImportError as e:
                logger.error(f"Failed to import QNN partitioner: {e}")
                raise RuntimeError("QNN partitioner not available") from e
        
        return self._partitioner
    
    def _get_soc_model(self):
        """获取 SoC 模型配置."""
        try:
            from executorch.backends.qualcomm.serialization.qnn_definitions import (
                QcomChipset,
            )
            
            # 默认使用 Snapdragon 8 Gen 2
            default_soc = QcomChipset.SM8550
            
            if self._soc_model is None:
                return default_soc
            
            # 映射常见的 SoC 名称
            soc_mapping = {
                "sm8550": QcomChipset.SM8550,  # Snapdragon 8 Gen 2
                "sm8650": QcomChipset.SM8650,  # Snapdragon 8 Gen 3
                "sm8475": QcomChipset.SM8475,  # Snapdragon 8+ Gen 1
                "sm8450": QcomChipset.SM8450,  # Snapdragon 8 Gen 1
                "sm7475": QcomChipset.SM7475,  # Snapdragon 7+ Gen 2
            }
            
            soc_key = self._soc_model.lower()
            return soc_mapping.get(soc_key, default_soc)
            
        except ImportError:
            logger.warning("QNN QcomChipset not available, using default")
            return None
    
    def export(
        self,
        model: nn.Module,
        example_inputs,
        config: Optional[BackendConfig] = None,
    ) -> ExportResult:
        """导出模型为 QNN 格式.
        
        Args:
            model: PyTorch 模型
            example_inputs: 示例输入
            config: 导出配置
            
        Returns:
            ExportResult: 导出结果
        """
        if config is not None:
            self.config = config
        
        # 检查平台
        if sys.platform == "win32":
            return ExportResult(
                success=False,
                error_message="QNN backend is not supported on Windows. Use Linux or Android.",
            )
        
        try:
            from torch.export import export
            from executorch.exir import to_edge_transform_and_lower
            from executorch.exir.passes import MemoryPlanningPass
            from executorch.exir.program._program import ExecutorchBackendConfig
            
            logger.info(f"Starting QNN export for SoC: {self._soc_model or 'default'}")
            
            # 1. 确保模型处于评估模式
            model.eval()
            
            # 2. 导出为 ExportedProgram
            logger.info("Exporting to ExportedProgram...")
            if isinstance(example_inputs, dict):
                ep = export(model, (), example_inputs)
            else:
                if not isinstance(example_inputs, tuple):
                    example_inputs = (example_inputs,)
                ep = export(model, example_inputs)
            
            # 3. 使用新的工作流：to_edge_transform_and_lower
            logger.info("Converting to Edge Dialect and lowering to QNN...")
            partitioner = self.get_partitioner()
            edge = to_edge_transform_and_lower(
                ep,
                partitioner=[partitioner],
            )
            logger.info("Edge conversion and QNN lowering successful")
            
            # 4. 生成 .pte 文件
            logger.info("Generating ExecuTorch program...")
            # memory_planning_algo 需要是一个 callable，而不是字符串
            from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
            
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
            
            exec_prog = edge.to_executorch(exec_config)
            buffer = exec_prog.buffer
            
            logger.info(f"QNN export successful, buffer size: {len(buffer)} bytes")
            
            metadata = {
                "buffer_size_bytes": len(buffer),
                "backend": "qnn",
                "soc_model": self._soc_model,
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
            logger.error(f"QNN export failed: {error_msg}")
            return ExportResult(
                success=False,
                error_message=error_msg,
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取后端元数据."""
        metadata = super().get_metadata()
        metadata.update({
            "supported_platforms": ["android", "linux_embedded"],
            "supported_soc": ["SM8550", "SM8650", "SM8475", "SM8450", "SM7475"],
            "supports_quantization": True,
            "recommended_quantization": "static",
            "recommended_for": ["qualcomm_npu", "android_high_end"],
            "requires_sdk": True,
        })
        return metadata


# 注册后端
from .base import BackendRegistry
BackendRegistry.register("qnn", QNNBackend)
