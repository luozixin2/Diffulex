# DiffuLex Edge - Agent Guide

AI Agent 工作指南 for DiffuLex Edge project.

## 快速导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目概览、快速开始 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 技术架构详情 |
| [docs/TEST_PLAN.md](docs/TEST_PLAN.md) | 测试策略 |
| [docs/EXPORT.md](docs/EXPORT.md) | 模型导出指南 |

## 项目结构

```
diffulex_edge/
├── model/                    # 模型实现
│   ├── fast_dllm_v2_edge.py  # FastdLLM V2
│   ├── dream_edge.py         # Dream
│   ├── llada_edge.py         # LLaDA
│   ├── sdar_edge.py          # SDAR
│   └── model_loader.py       # HF模型加载
├── runtime/                  # 推理运行时
│   ├── engine.py             # InferenceEngine (自回归)
│   ├── diffusion.py          # DiffusionEngine (扩散)
│   ├── block.py              # DiffusionBlock管理
│   └── sampler/              # 采样器
│       ├── base.py           # 基础采样函数
│       ├── shift.py          # Logits偏移
│       └── models/           # 模型特有采样器
├── export/                   # 模型导出
│   ├── exporter.py           # DiffuLexExporter
│   └── config.py             # 导出配置
├── backends/                 # 后端实现
│   ├── xnnpack_backend.py    # XNNPACK (CPU)
│   ├── coreml_backend.py     # CoreML (Apple)
│   └── qnn_backend.py        # QNN (Qualcomm)
└── scripts/
    └── export_model.py       # 导出脚本
```

## 关键设计原则

### 1. 采样器对齐 (已完成)

四个模型各自有独立的采样器实现，与原版 diffulex 行为对齐：

| 模型 | Shift Logits | Per-block Threshold | pre_block_complete |
|------|-------------|---------------------|-------------------|
| FastdLLM V2 | ✅ | ❌ | ❌ |
| LLaDA | ❌ | ✅ | ✅ |
| Dream | ✅ | ✅ | ✅ |
| SDAR | ✅ | ❌ | ❌ |

核心文件：
- `runtime/sampler/base.py`: `sample_tokens`, `top_p_logits`, `top_k_logits`
- `runtime/sampler/shift.py`: `ShiftLogitsSampler`, `NoShiftLogitsSampler`
- `runtime/sampler/models/`: 四个模型各自的采样器

### 2. 模型导出流程

```python
# 从HuggingFace加载并导出
from diffulex_edge.model import load_hf_model
from diffulex_edge.export import DiffuLexExporter, ExportConfig

model, model_type, hf_config = load_hf_model("path/to/hf/model", dtype=torch.bfloat16)

export_config = ExportConfig(
    output_path="model.pte",
    backend="xnnpack",
    dtype=torch.bfloat16,
)

exporter = DiffuLexExporter(export_config)
example_inputs = (torch.zeros(1, 128, dtype=torch.long),)  # input_ids
result = exporter.export(model, example_inputs)
```

### 3. 推理引擎

两种引擎，API一致：

```python
# 自回归生成
from diffulex_edge.runtime import InferenceEngine, GenerationConfig
engine = InferenceEngine.from_pte("model.pte")
tokens = engine.generate(prompt, GenerationConfig(max_new_tokens=100))

# 扩散生成
from diffulex_edge.runtime import DiffusionEngine, DiffusionGenerationConfig
engine = DiffusionEngine.from_pte("model.pte", model_type="fast_dllm_v2")
tokens = engine.generate(prompt, DiffusionGenerationConfig(max_new_tokens=100))
```

## 常用任务

### 添加新模型支持

1. 创建 `model/{model_name}_edge.py`
2. 创建 `runtime/sampler/models/{model_name}.py`
3. 添加到 `runtime/sampler/models/__init__.py` 的 `SAMPLER_REGISTRY`
4. 添加到 `model/model_loader.py` 的 `MODEL_REGISTRY`
5. 添加测试到 `tests/test_multi_model_support.py`

### 运行测试

```bash
# 全部测试
pytest diffulex_edge/tests/ -v

# 特定模块
pytest diffulex_edge/tests/test_sampler_alignment.py -v
pytest diffulex_edge/tests/test_diffusion_pte.py -v

# 覆盖率
pytest --cov=diffulex_edge --cov-report=html
```

## 平台限制

- **Windows**: 无法生成 .pte 文件 (缺少 flatc)，可用 WSL2
- **CoreML**: 仅 macOS/iOS
- **QNN**: 仅 Linux/Android

## 依赖

```bash
# 核心
pip install torch numpy

# 导出
pip install executorch

# 完整
pip install -e .
```
