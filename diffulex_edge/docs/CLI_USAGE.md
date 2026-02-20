# DiffuLex Edge CLI 使用指南

## 概述

DiffuLex Edge 现在提供了一个交互式命令行界面(CLI)，支持 PyTorch 和 PTE 两种模型格式的对话生成，并完全支持 **HuggingFace Tokenizer**。

## 安装要求

```bash
# 基础功能（使用简单演示tokenizer）
pip install torch

# 完整功能（支持HuggingFace Tokenizer）
pip install transformers
```

## 安装后使用

安装包后，可以直接使用以下命令：

```bash
diffulex-edge --help
```

## 运行方式

### 方式1: 使用演示模型
```bash
python -m diffulex_edge
```

### 方式2: 加载 PyTorch 模型
```bash
python -m diffulex_edge --model-path path/to/model.pt
```

### 方式3: 加载 PTE 模型
```bash
python -m diffulex_edge --pte-path path/to/model.pte
```

### 方式4: 使用 HuggingFace Tokenizer

```bash
# 使用预训练模型的tokenizer（会自动下载）
python -m diffulex_edge --model-path model.pt --tokenizer gpt2

# 使用本地tokenizer目录
python -m diffulex_edge --model-path model.pt --tokenizer ./models/tokenizer/

# 使用特定模型的tokenizer
python -m diffulex_edge --pte-path model.pte --tokenizer meta-llama/Llama-2-7b-hf

# 禁用chat template（某些模型可能不支持）
python -m diffulex_edge --tokenizer gpt2 --no-chat-template
```

### 方式5: 完整参数
```bash
python -m diffulex_edge \
    --pte-path model.pte \
    --tokenizer gpt2 \
    --max-tokens 200 \
    --temperature 0.7 \
    --top-k 40 \
    --top-p 0.95
```

## 交互命令

启动 CLI 后，可以使用以下命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/quit` 或 `/exit` | 退出程序 |
| `/clear` | 清空对话历史 |
| `/config` | 显示当前配置 |
| `/temp <0.0-2.0>` | 设置 temperature |
| `/topk <n>` | 设置 top_k (0=禁用) |
| `/topp <0.0-1.0>` | 设置 top_p |
| `/max <n>` | 设置最大生成 token 数 |
| `/iter <n>` | 设置扩散迭代次数 |
| `/conf <0.0-1.0>` | 设置置信度阈值 |

## HuggingFace Tokenizer 支持

### 自动检测

如果指定了 `--model-path`，CLI 会自动检测同级目录下的 tokenizer 文件：
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

### Chat Template

支持 HuggingFace 的 `apply_chat_template` 功能，自动处理对话格式：

```bash
# 使用支持 chat template 的模型（如 Llama-2）
python -m diffulex_edge \
    --model-path model.pt \
    --tokenizer meta-llama/Llama-2-7b-chat-hf

# 禁用 chat template
python -m diffulex_edge \
    --tokenizer gpt2 \
    --no-chat-template
```

### 支持的 Tokenizer 类型

- **GPT-2/GPT-Neo/GPT-J**: `gpt2`, `EleutherAI/gpt-neo-2.7B`
- **Llama/Llama-2**: `meta-llama/Llama-2-7b-hf`
- **CodeLlama**: `codellama/CodeLlama-7b-hf`
- **本地路径**: `./my_tokenizer/`

## 使用示例

### 示例会话（使用演示tokenizer）

```bash
$ python -m diffulex_edge

============================================================
DiffuLex Edge - Interactive Chat CLI
============================================================

未提供模型路径，使用演示模型
模型参数: 4,448,512 (4.45M)
Using simple demo tokenizer (character-based)

模型类型: 扩散模型
Tokenizer: 简单演示
输入 /help 查看可用命令
------------------------------------------------------------

You: 你好
AI: <...生成的回复...>

You: /temp 0
  temperature = 0.0

You: /max 50
  max_new_tokens = 50

You: /quit
再见!
```

### 示例会话（使用 HuggingFace Tokenizer）

```bash
$ python -m diffulex_edge --tokenizer gpt2

============================================================
DiffuLex Edge - Interactive Chat CLI
============================================================

未提供模型路径，使用演示模型
创建演示模型...
模型参数: 4,448,512 (4.45M)
Loaded HuggingFace tokenizer from: gpt2
Vocab size: 50257

模型类型: 扩散模型
Tokenizer: HuggingFace
输入 /help 查看可用命令
------------------------------------------------------------

You: Hello, how are you?
AI: <...generated response using GPT-2 tokenizer...>

You: /quit
再见!
```

### 调整生成参数

```bash
# 更创造性的回复
You: /temp 1.2
  temperature = 1.2

# 更确定性的回复
You: /temp 0
  temperature = 0.0

# 增加生成长度
You: /max 200
  max_new_tokens = 200

# 调整扩散迭代次数（仅扩散模型）
You: /iter 20
  num_iterations = 20
```

## 参数说明

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | - | PyTorch 模型文件路径 |
| `--pte-path` | - | ExecuTorch PTE 模型路径 |
| `--tokenizer` | - | HuggingFace tokenizer 名称或路径 |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 0.8 | 采样温度 (0.0-2.0) |
| `--top-k` | 50 | Top-K 采样 (0=禁用) |
| `--top-p` | 0.9 | Top-P (nucleus) 采样 |
| `--device` | cpu | 运行设备 (cpu/cuda) |
| `--no-chat-template` | False | 禁用 chat template |

### 生成参数

**Temperature**:
- 0.0: 完全确定性，总是选择概率最高的token
- 0.5-0.8: 平衡，适合大多数场景
- 1.0+: 更随机，创造性更强

**Top-K**: 只从概率最高的 K 个token中采样
- 0: 禁用，使用全部词汇表
- 40-50: 常用值

**Top-P**: 从累积概率达到 P 的最小集合中采样
- 0.9: 常用值
- 1.0: 禁用

**扩散模型特有参数**:
- `num_iterations`: 扩散迭代次数 (默认: 10)
- `confidence_threshold`: 置信度阈值 (默认: 0.9)

## 注意事项

1. **Tokenizer 选择**: 
   - 生产环境推荐使用 HuggingFace Tokenizer (`--tokenizer`)
   - 演示模式使用简单字符编码，仅用于测试

2. **对话历史**: 默认保留最近 1000 个 token 的历史。使用 `/clear` 可以清空历史开始新对话。

3. **模型格式**:
   - PyTorch 模型: `.pt` 或 `.pth` 文件
   - PTE 模型: 通过 ExecuTorch 导出的 `.pte` 文件

4. **Chat Template**: 仅在 tokenizer 支持且未禁用 `--no-chat-template` 时使用。

5. **错误处理**: 如果模型加载失败，CLI 会显示错误信息并退出。

## 高级用法

### 从模型目录自动加载 Tokenizer

```bash
# 如果模型目录包含 tokenizer 文件，会自动检测
python -m diffulex_edge --model-path ./models/my_model/model.pt
# 输出: 自动检测到tokenizer: ./models/my_model/
```

### 批量生成

可以通过管道传递输入：

```bash
echo -e "问题1\n问题2\n问题3" | python -m diffulex_edge --tokenizer gpt2
```

### 配置文件（未来版本）

未来版本将支持从配置文件加载默认参数：

```bash
python -m diffulex_edge --config chat_config.yaml
```

## 故障排除

### Tokenizer 加载失败

```
Warning: Failed to load tokenizer from xxx: ...
Falling back to simple tokenizer
```
解决: 检查 tokenizer 路径或网络连接（如果是 HuggingFace 模型名称）

### transformers 未安装

```
Warning: transformers not installed, using simple tokenizer demo
```
解决: `pip install transformers`

### 模型加载失败

```
错误: 不能同时指定 --model-path 和 --pte-path
```
解决: 只使用其中一个参数

### 内存不足

```bash
# 使用较小的 max_tokens
python -m diffulex_edge --tokenizer gpt2 --max-tokens 50
```

### ExecuTorch 未安装

```
加载PTE失败: No module named 'executorch'
```
解决: 安装 ExecuTorch 或使用 PyTorch 模型

## 相关文档

- [快速开始](./QUICK_START.md)
- [模型导出指南](./EXPORT_GUIDE.md)
- [API 参考](./API_REFERENCE.md)
- [HuggingFace Tokenizer 文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)
