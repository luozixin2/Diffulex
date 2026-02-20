# Diffulex Edge 架构评审报告

## 1. 概述

本文档从架构师角度对 `diffulex_edge` 代码库进行全面评审，识别架构问题并提出改进建议。

### 1.1 当前架构概况

```
diffulex_edge/
├── backends/          # 导出后端（XNNPACK, CoreML, QNN）
├── export/            # 导出配置和主逻辑
├── model/             # 4个模型实现（FastdLLM, Dream, LLaDA, SDAR）
├── runtime/           # 推理引擎和采样器
├── quant/             # 量化支持
└── scripts/           # 导出脚本
```

---

## 2. 架构问题识别

### 2.1 问题一：Wrapper 位置不当（严重）

**现状：**
- `SDAREdgeExportWrapper` 定义在 `xnnpack_backend.py` 内部（第115-153行）
- 这是 SDAR 模型特有的逻辑，却嵌入到 XNNPACK 后端中
- 其他后端（CoreML、QNN）没有处理 `forward_export` 的逻辑

**代码位置：** `backends/xnnpack_backend.py:115-153`

```python
# XNNPACK 后端中的 SDAR 特定逻辑
if hasattr(model, 'forward_export'):
    logger.info("Detected forward_export method, wrapping for export...")
    
    class SDAREdgeExportWrapper(torch.nn.Module):
        # SDAR 模型特有的包装器
        ...
    
    wrapped_model = SDAREdgeExportWrapper(model)
```

**影响：**
1. 每支持一个新模型需要修改所有后端代码
2. 每添加一个新后端需要复制 SDAR 处理逻辑
3. 违反**开闭原则**（对扩展开放，对修改关闭）
4. CoreML 和 QNN 后端无法导出 SDAR 模型

---

### 2.2 问题二：严重代码重复（高）

**现状：**
四个模型文件各自定义几乎相同的组件：

| 组件 | FastdLLM | Dream | LLaDA | SDAR |
|------|----------|-------|-------|------|
| RMSNorm | `RMSNorm` | `DreamRMSNorm` | `LLaDARMSNorm` | `SDARRMSNorm` |
| RoPE | `RotaryEmbedding` | `DreamRotaryEmbedding` | `LLaDARotaryEmbedding` | `SDARRotaryEmbedding` |
| MLP | `MLP` | `DreamMLP` | `LLaDAMLP` | `SDARMLP` |
| Attention | `AttentionEdge` | `DreamAttention` | `LLaDAAttention` | `SDARAttention` |

**代码统计：**
- RMSNorm 实现重复 4 次（每处约 15 行）
- RoPE 实现重复 4 次（每处约 70 行）
- MLP 实现重复 4 次（每处约 20 行）
- 总计约 420 行重复代码

**影响：**
1. 修复 bug 需要修改多处
2. 添加功能需要重复实现
3. 代码库膨胀，维护困难

---

### 2.3 问题三：模型接口不统一（高）

**现状：**
各模型的 forward 方法签名不一致：

```python
# FastdLLM V2
forward(self, input_ids, positions=None, attention_mask=None, 
        kv_cache=None, start_pos=0)

# Dream / LLaDA
forward(self, input_ids, positions=None, mask=None, 
        kv_cache=None, start_pos=0)

# SDAR（两种模式）
forward(self, input_ids, positions, mask=None, kv_cache=None, max_seq_len=None)
forward_export(self, input_ids, positions, kv_cache, attention_mask, 
               insert_matrix, keep_mask)
```

**影响：**
1. `DiffusionEngine` 需要特殊处理不同模型
2. 无法编写通用的模型无关代码
3. 添加新模型需要理解并适配接口差异

---

### 2.4 问题四：导出逻辑分散（中）

**现状：**
`export_model.py` 需要为每个模型类型创建不同的 example_inputs：

```python
# Check if model uses forward_export (SDAREdge style with KV cache)
if hasattr(model, 'forward_export'):
    # SDAR 特有的输入格式
    input_ids = torch.zeros(batch_size, block_size, dtype=torch.long)
    positions = torch.arange(block_size, dtype=torch.long)...
    kv_cache = torch.zeros(num_layers, 2, batch_size, num_kv_heads, ...)
    attention_mask = torch.zeros(...)
    insert_matrix = torch.zeros(...)
    keep_mask = torch.zeros(...)
    example_inputs = (input_ids, positions, kv_cache, attention_mask, ...)
else:
    # 标准格式
    example_inputs = (torch.randint(0, vocab_size, (args.batch_size, args.seq_len)),)
```

**影响：**
1. 导出脚本需要了解模型内部实现细节
2. 添加新模型需要修改导出脚本
3. 模型和导出逻辑耦合

---

### 2.5 问题五：配置重复（中）

**现状：**
每个模型有自己的 Config dataclass，结构几乎相同：

```python
@dataclass
class FastdLLMV2EdgeConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    ...

@dataclass  
class DreamEdgeConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    ...

# LLaDAEdgeConfig, SDAREdgeConfig 类似...
```

---

### 2.6 问题六：后端注册方式不一致（低）

**现状：**
后端注册在文件末尾使用全局代码：

```python
# 每个后端文件末尾
from .base import BackendRegistry
BackendRegistry.register("xnnpack", XNNPACKBackend)
```

这种方式：
1. 依赖文件导入顺序
2. 可能导致循环导入
3. 不易于测试（注册发生在导入时）

---

## 3. 架构设计原则评估

| 原则 | 当前状态 | 评分 |
|------|----------|------|
| **单一职责** (SRP) | 后端负责导出+模型特定包装逻辑 | ⚠️ 差 |
| **开闭原则** (OCP) | 添加新模型需修改后端 | ❌ 差 |
| **里氏替换** (LSP) | 模型接口不一致 | ⚠️ 中 |
| **接口隔离** (ISP) | 后端接口与模型实现耦合 | ⚠️ 中 |
| **依赖倒置** (DIP) | 后端依赖具体模型实现 | ❌ 差 |
| **DRY (不要重复)** | 大量代码重复 | ❌ 差 |
| **高内聚** | 模型组件分散在各文件中 | ⚠️ 中 |
| **低耦合** | 后端与模型强耦合 | ❌ 差 |

**总体评价：** 架构存在明显缺陷，急需重构。

---

## 4. 重构目标

### 4.1 短期目标
1. 提取共享组件，消除代码重复
2. 将 Wrapper 从后端移到模型层
3. 统一模型接口

### 4.2 长期目标
1. 实现真正的后端-模型解耦
2. 支持插件式扩展新模型
3. 建立清晰的抽象层次

---

## 5. 推荐架构方案

### 5.1 目标架构

```
diffulex_edge/
├── components/           # 共享组件（新模块）
│   ├── __init__.py
│   ├── normalization.py  # RMSNorm（统一实现）
│   ├── rope.py          # RoPE（统一实现）
│   └── mlp.py           # MLP（统一实现）
├── models/              # 重命名：model -> models
│   ├── __init__.py
│   ├── base.py          # 模型抽象基类（新）
│   ├── wrapper.py       # 导出包装器（新）
│   ├── fast_dllm_v2.py  # 简化后的模型
│   ├── dream.py
│   ├── llada.py
│   ├── sdar.py
│   └── loader.py        # 模型加载器
├── backends/            # 后端保持简洁
│   ├── base.py
│   ├── xnnpack.py
│   ├── coreml.py
│   └── qnn.py
└── ...
```

### 5.2 核心抽象

```python
# models/base.py
class DiffusionModel(nn.Module, ABC):
    """所有扩散模型的抽象基类。"""
    
    @abstractmethod
    def forward(self, input_ids: Tensor, positions: Tensor, 
                **kwargs) -> Tuple[Tensor, ...]:
        """推理 forward。"""
        pass
    
    @property
    @abstractmethod
    def supports_export(self) -> bool:
        """是否支持导出（有 forward_export）。"""
        return False
    
    def get_export_wrapper(self) -> Optional[nn.Module]:
        """获取导出包装器，默认返回 None。"""
        if not self.supports_export:
            return None
        return ExportWrapper(self)


# models/wrapper.py
class ExportWrapper(nn.Module):
    """通用导出包装器。"""
    
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model
        self._sanitize_parameters()
    
    def _sanitize_parameters(self):
        """清理参数名（替换 . 为 _）。"""
        for name, param in self.model.named_parameters():
            safe_name = name.replace('.', '_')
            self.register_parameter(safe_name, param)
    
    def forward(self, *args, **kwargs):
        """调用模型的 forward_export。"""
        return self.model.forward_export(*args, **kwargs)


# backends/base.py
class EdgeBackend(ABC):
    """简化后的后端基类。"""
    
    def export(self, model: nn.Module, example_inputs, config=None) -> ExportResult:
        """导出模型。"""
        # 通用逻辑，不处理模型特定细节
        model_to_export = model.get_export_wrapper() if hasattr(model, 'get_export_wrapper') else model
        # ... 导出流程
```

---

## 6. 重构计划

### Phase 1: 提取共享组件（低风险）
- 创建 `components/` 模块
- 提取 RMSNorm, RoPE, MLP
- 更新各模型使用共享组件
- **不影响功能，纯代码移动**

### Phase 2: 引入模型抽象（中风险）
- 创建 `models/base.py`
- 为 SDAR 实现 `get_export_wrapper()`
- 从 XNNPACK 后端移除 SDAREdgeExportWrapper
- **需要测试导出流程**

### Phase 3: 统一接口（中风险）
- 统一所有模型的 forward 签名
- 更新 DiffusionEngine 使用统一接口
- **需要测试推理流程**

### Phase 4: 完善架构（低风险）
- 统一配置系统
- 改进后端注册机制
- 添加架构测试

---

## 7. 具体修改建议

### 7.1 立即修改（高优先级）

1. **将 Wrapper 移出后端**
   ```python
   # 在 models/sdar_edge.py 中添加
   class SDAREdge(nn.Module):
       ...
       
       def get_export_wrapper(self):
           """返回导出包装器。"""
           return SDAREdgeExportWrapper(self)
   
   # 在 backends/xnnpack_backend.py 中改为
   if hasattr(model, 'get_export_wrapper'):
       wrapper = model.get_export_wrapper()
       if wrapper is not None:
           export_model = wrapper
   ```

2. **为 CoreML 和 QNN 添加相同逻辑**

### 7.2 短期修改（中优先级）

1. 创建 `components/` 模块
2. 提取共享的 RMSNorm、RoPE、MLP

### 7.3 长期修改（低优先级）

1. 统一所有模型接口
2. 建立完整的抽象层次
3. 重构配置系统

---

## 8. 结论

当前架构存在以下核心问题：
1. **后端与模型强耦合** - 需要优先解决
2. **严重代码重复** - 影响维护效率  
3. **接口不统一** - 阻碍通用代码编写

建议按 Phase 1-4 的计划逐步重构，每次重构后运行完整测试确保功能正确。

**预估工作量：**
- Phase 1: 1-2 天
- Phase 2: 2-3 天
- Phase 3: 3-5 天
- Phase 4: 2-3 天

**总计：1-2 周完成完整重构**
