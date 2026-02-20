# Diffulex Edge 架构重构实施指南

## 概述

本文档提供从当前架构迁移到目标架构的具体实施步骤。重构分为多个阶段，每个阶段都是独立的，可以单独实施和验证。

---

## 已完成的重构组件

以下组件已在新架构中实现：

### 1. 共享组件模块 (`components/`)
```
diffulex_edge/components/
├── __init__.py
├── normalization.py   # RMSNorm
├── rope.py           # RotaryEmbedding  
└── mlp.py            # SwiGLUMLP
```

### 2. 模型基类 (`model/base.py`)
- `ModelConfig` - 统一配置基类
- `DiffusionModel` - 模型抽象基类
- `KVCacheModel` - KV缓存模型基类

### 3. 导出包装器 (`model/wrapper.py`)
- `ExportWrapper` - 通用导出包装器
- `BlockDiffusionWrapper` - Block Diffusion专用包装器

### 4. 后端解耦 (`backends/xnnpack_backend.py`)
- 移除 SDAR 特定逻辑
- 使用通用的 `_prepare_model_for_export()` 方法

### 5. 重构示例 (`model/sdar_edge_refactored.py`)
- 展示如何使用新架构实现模型

---

## 实施阶段

### Phase 1: 验证新组件（建议立即执行）

**目标：** 确保新组件可以独立工作

**步骤：**

1. **测试共享组件**
   ```python
   from diffulex_edge.components import RMSNorm, RotaryEmbedding, SwiGLUMLP
   
   # 验证组件可以正常实例化和使用
   norm = RMSNorm(hidden_size=512)
   rope = RotaryEmbedding(head_dim=64)
   mlp = SwiGLUMLP(hidden_size=512, intermediate_size=2048)
   ```

2. **测试包装器**
   ```python
   from diffulex_edge.model.wrapper import ExportWrapper
   from diffulex_edge.model.sdar_edge import SDAREdge, SDAREdgeConfig
   
   config = SDAREdgeConfig(num_hidden_layers=2)  # 小型模型用于测试
   model = SDAREdge(config)
   wrapper = ExportWrapper(model)
   ```

3. **验证后端集成**
   ```python
   from diffulex_edge.backends import XNNPACKBackend
   
   backend = XNNPACKBackend()
   # 确保后端可以正常实例化
   ```

**成功标准：** 所有导入和新组件实例化不报错

---

### Phase 2: 后端迁移（低风险，建议立即执行）

**目标：** 让当前 SDAR 模型通过新架构导出

**步骤：**

1. **更新 XNNPACK 后端**
   
   当前的 `xnnpack_backend.py` 已经包含新逻辑，但需要确保回退机制正常工作：
   
   ```python
   def _prepare_model_for_export(self, model):
       # 新架构：使用模型的 get_export_wrapper
       if hasattr(model, 'get_export_wrapper'):
           wrapper = model.get_export_wrapper()
           if wrapper is not None:
               return wrapper
       
       # 向后兼容：直接使用 forward_export
       if hasattr(model, 'forward_export'):
           from ..model.wrapper import ExportWrapper
           return ExportWrapper(model)
       
       return model
   ```

2. **更新 CoreML 和 QNN 后端**
   
   为 `coreml_backend.py` 和 `qnn_backend.py` 添加相同的 `_prepare_model_for_export` 方法。

3. **测试导出**
   ```bash
   python -m diffulex_edge.scripts.export_model \
       --demo --model-type sdar \
       -o /tmp/test_sdar.pte
   ```

**成功标准：** SDAR 模型可以正常导出，生成的 .pte 文件大小正确

---

### Phase 3: 模型逐步迁移（中风险）

**目标：** 逐个迁移模型使用新架构

**建议顺序：** SDAR → FastdLLM → Dream → LLaDA

**每个模型的迁移步骤：**

以 SDAR 为例：

1. **创建新模型文件**（已完成示例：`sdar_edge_refactored.py`）

2. **修改点对比：**

   | 方面 | 旧实现 | 新实现 |
   |------|--------|--------|
   | 导入 | 内联组件定义 | `from ..components import ...` |
   | 配置 | 独立 dataclass | 继承 `ModelConfig` |
   | 基类 | `nn.Module` | `DiffusionModel` |
   | RMSNorm | `SDARRMSNorm` | `RMSNorm` |
   | RoPE | `SDARRotaryEmbedding` | `RotaryEmbedding` |
   | MLP | `SDARMLP` | `SwiGLUMLP` |
   | 导出包装 | 无 | `get_export_wrapper()` |

3. **验证新实现**
   ```python
   # 加载旧模型
   from diffulex_edge.model.sdar_edge import SDAREdge, SDAREdgeConfig as OldConfig
   old_model = SDAREdge(OldConfig(num_hidden_layers=2))
   
   # 加载新模型
   from diffulex_edge.model.sdar_edge_refactored import SDAREdge as NewSDAR
   from diffulex_edge.model.sdar_edge_refactored import SDAREdgeConfig as NewConfig
   new_model = NewSDAR(NewConfig(num_hidden_layers=2))
   
   # 复制权重
   new_model.load_state_dict(old_model.state_dict())
   
   # 验证输出一致
   import torch
   x = torch.randint(0, 1000, (1, 10))
   pos = torch.arange(10).unsqueeze(0)
   
   with torch.no_grad():
       out_old, _ = old_model(x, pos)
       out_new, _ = new_model(x, pos)
   
   assert torch.allclose(out_old, out_new, atol=1e-5)
   ```

4. **替换原文件**
   
   验证通过后：
   ```bash
   mv diffulex_edge/model/sdar_edge.py diffulex_edge/model/sdar_edge_legacy.py
   mv diffulex_edge/model/sdar_edge_refactored.py diffulex_edge/model/sdar_edge.py
   ```

**成功标准：** 新模型输出与旧模型在数值上一致

---

### Phase 4: 更新导出脚本（低风险）

**目标：** 简化导出脚本，使用模型的 `get_export_inputs()`

**修改 `scripts/export_model.py`：**

```python
# 旧方式（需要知道模型类型）
if hasattr(model, 'forward_export'):
    # SDAR 特有的输入格式
    ...
else:
    example_inputs = (torch.randint(...),)

# 新方式（通用）
if hasattr(model, 'get_export_inputs'):
    example_inputs = model.get_export_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
else:
    # 向后兼容
    example_inputs = (torch.randint(...),)
```

**成功标准：** 导出脚本可以更简洁，支持新模型无需修改

---

### Phase 5: 清理和优化（低风险）

**目标：** 移除旧代码，完善架构

**步骤：**

1. **移除旧模型文件**
   ```bash
   rm diffulex_edge/model/sdar_edge_legacy.py
   rm diffulex_edge/model/fast_dllm_v2_edge_legacy.py
   # ... etc
   ```

2. **更新 `__init__.py` 导出**
   ```python
   # diffulex_edge/model/__init__.py
   from .base import ModelConfig, DiffusionModel
   from .wrapper import ExportWrapper, BlockDiffusionWrapper
   from .sdar_edge import SDAREdge, SDAREdgeConfig
   # ... etc
   ```

3. **统一配置系统**
   
   更新 `model_loader.py` 使用 `ModelConfig` 基类：
   ```python
   from .base import ModelConfig
   
   def create_edge_config(model_type: str, hf_config: Dict) -> ModelConfig:
       # 使用统一的配置创建逻辑
       ...
   ```

---

## 风险评估与缓解

| 阶段 | 风险等级 | 主要风险 | 缓解措施 |
|------|----------|----------|----------|
| Phase 1 | 低 | 新组件有 bug | 充分单元测试 |
| Phase 2 | 低 | 导出失败 | 保留向后兼容代码 |
| Phase 3 | 中 | 数值不一致 | 对比测试输出 |
| Phase 4 | 低 | 脚本不工作 | 保留旧逻辑作为 fallback |
| Phase 5 | 低 | 误删文件 | 使用 git，可恢复 |

---

## 测试检查清单

每个阶段完成后，验证以下功能：

- [ ] 组件可以正常实例化
- [ ] 模型可以加载和推理
- [ ] 导出流程正常工作
- [ ] 生成的 .pte 文件有效
- [ ] 运行时引擎可以加载模型
- [ ] 数值输出与旧版本一致

---

## 时间估计

| 阶段 | 预计时间 | 依赖 |
|------|----------|------|
| Phase 1 | 2-4 小时 | 无 |
| Phase 2 | 4-6 小时 | Phase 1 |
| Phase 3 (每个模型) | 1-2 天 | Phase 2 |
| Phase 4 | 2-4 小时 | Phase 3 |
| Phase 5 | 4-8 小时 | Phase 4 |

**总计：约 1-2 周（全职工作）**

---

## 回滚策略

如果重构过程中发现问题：

1. **立即停止**，不要继续后续阶段
2. 使用 git 回滚到上一个稳定提交：
   ```bash
   git checkout <stable-commit>
   ```
3. 修复问题后再继续

每个阶段的修改应该在一个独立的 commit 中，便于回滚。

---

## 结论

这个重构计划将显著提升代码质量：

1. **消除重复代码** - 减少 ~400 行重复实现
2. **解耦后端和模型** - 后端代码减少 ~30%
3. **统一接口** - 更容易添加新模型
4. **提高可测试性** - 组件可以独立测试

建议按阶段逐步实施，每个阶段充分验证后再继续。
