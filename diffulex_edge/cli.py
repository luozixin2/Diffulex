#!/usr/bin/env python3
"""
DiffuLex Edge - Interactive Chat CLI

支持四种扩散模型:
1. FastdLLM V2: python -m diffulex_edge --model-type fast_dllm
2. Dream: python -m diffulex_edge --model-type dream
3. LLaDA: python -m diffulex_edge --model-type llada
4. SDAR: python -m diffulex_edge --model-type sdar

交互命令:
  /help       - 显示帮助
  /quit       - 退出
  /clear      - 清空历史
  /config     - 显示当前配置
  /temp <n>   - 设置temperature (0.0-2.0)
  /topk <n>   - 设置top_k (0=disabled)
  /topp <n>   - 设置top_p (0.0-1.0)
  /max <n>    - 设置最大生成token数
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch not installed")
    sys.exit(1)

try:
    from .runtime.engine import DiffusionEngine, DiffusionGenerationConfig
    from .model import (
        FastdLLMV2Edge, FastdLLMV2EdgeConfig,
        DreamEdge, DreamEdgeConfig,
        LLaDAEdge, LLaDAEdgeConfig,
        SDAREdge, SDAREdgeConfig,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from runtime.engine import DiffusionEngine, DiffusionGenerationConfig
    from model import (
        FastdLLMV2Edge, FastdLLMV2EdgeConfig,
        DreamEdge, DreamEdgeConfig,
        LLaDAEdge, LLaDAEdgeConfig,
        SDAREdge, SDAREdgeConfig,
    )

# HuggingFace transformers
try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False


# ============================================================================
# Tokenizer Wrapper
# ============================================================================

class TokenizerWrapper:
    """Tokenizer包装器，支持HuggingFace Tokenizer和简单演示tokenizer"""
    
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 130000):
        self.tokenizer = None
        self.is_hf_tokenizer = False
        self.vocab_size = vocab_size
        
        if tokenizer_path and HF_TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.is_hf_tokenizer = True
                self.vocab_size = len(self.tokenizer)
                print(f"Loaded HuggingFace tokenizer: {tokenizer_path} (vocab_size={self.vocab_size})")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
        
        if self.tokenizer is None:
            print(f"Using simple tokenizer (vocab_size={vocab_size})")
    
    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self._simple_encode(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token IDs为文本"""
        if self.is_hf_tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self._simple_decode(token_ids)
    
    def apply_chat_template(self, messages: List[dict], tokenize: bool = True) -> Union[str, List[int]]:
        """应用对话模板"""
        if self.is_hf_tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            result = self.tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True)
            # Handle dict / BatchEncoding / Encoding objects
            if isinstance(result, dict):
                # Dictionary with 'input_ids' key
                return result['input_ids']
            elif hasattr(result, 'input_ids'):
                # BatchEncoding object
                ids = result.input_ids
                return ids.tolist() if hasattr(ids, 'tolist') else ids
            elif hasattr(result, 'ids'):
                # Encoding object
                return result.ids
            return result
        else:
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            text += "assistant: "
            return self.encode(text) if tokenize else text
    
    def _simple_encode(self, text: str) -> List[int]:
        """简单字符编码"""
        tokens = []
        for char in text[:1000]:
            code = ord(char)
            if code < 256:
                token_id = code
            else:
                token_id = 256 + (code % (self.vocab_size - 256))
            token_id = min(token_id, self.vocab_size - 10)
            tokens.append(token_id)
        return tokens
    
    def _simple_decode(self, token_ids: List[int]) -> str:
        """简单字符解码"""
        chars = []
        for t in token_ids:
            if 32 <= t < 127:
                chars.append(chr(t))
            elif t >= 127 and t < self.vocab_size:
                chars.append(chr(t))
            else:
                chars.append(f"<{t}>")
        return "".join(chars)


# ============================================================================
# Chat Configuration
# ============================================================================

class ChatConfig:
    """对话配置"""
    def __init__(self, block_size: int = 10):
        self.max_new_tokens: int = 100
        self.temperature: float = 0.8
        self.top_k: int = 50
        self.top_p: float = 0.9
        self.confidence_threshold: float = 0.9
        self.num_iterations: int = 10
        self.use_diffusion: bool = True
        self.use_chat_template: bool = True
        self.block_size: int = block_size  # Diffusion block size (model-specific)
    
    def to_diffusion_config(self) -> DiffusionGenerationConfig:
        return DiffusionGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_iterations=self.num_iterations,
            block_size=self.block_size,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            confidence_threshold=self.confidence_threshold,
        )
    
    def to_generation_config(self) -> DiffusionGenerationConfig:
        return DiffusionGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
    
    def __str__(self):
        return f"""当前配置:
  模式: {'扩散生成' if self.use_diffusion else '自回归生成'}
  max_new_tokens: {self.max_new_tokens}
  temperature: {self.temperature}
  top_k: {self.top_k}
  top_p: {self.top_p}
  confidence_threshold: {self.confidence_threshold}
  num_iterations: {self.num_iterations}
"""


# ============================================================================
# Chat Session
# ============================================================================

class ChatSession:
    """对话会话管理"""
    
    def __init__(self, engine, config: ChatConfig, tokenizer: TokenizerWrapper, is_diffusion: bool = True):
        self.engine = engine
        self.config = config
        self.tokenizer = tokenizer
        self.is_diffusion = is_diffusion
        self.history: List[int] = []
        self.messages: List[dict] = []
    
    def generate(self, prompt_text: str) -> str:
        """生成回复"""
        self.messages.append({"role": "user", "content": prompt_text})
        
        if self.config.use_chat_template and self.tokenizer.is_hf_tokenizer:
            prompt_tokens = self.tokenizer.apply_chat_template(self.messages, tokenize=True)
        else:
            if self.history:
                current_tokens = self.tokenizer.encode(prompt_text)
                prompt_tokens = self.history + current_tokens
            else:
                prompt_tokens = self.tokenizer.encode(prompt_text)
        
        # 确保token IDs在有效范围内
        prompt_tokens = [min(t, self.tokenizer.vocab_size - 1) for t in prompt_tokens]
        
        if self.is_diffusion:
            gen_config = self.config.to_diffusion_config()
            result = self.engine.generate(prompt_tokens, gen_config)
            generated_tokens = result[len(prompt_tokens):]
        else:
            gen_config = self.config.to_generation_config()
            generated_tokens = self.engine.generate(prompt_tokens, gen_config)
        
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        self.messages.append({"role": "assistant", "content": response_text})
        full_sequence = prompt_tokens + generated_tokens
        self.history = full_sequence[-1000:]
        
        return response_text
    
    def clear_history(self):
        """清空历史"""
        self.history = []
        self.messages = []
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"  {key} = {value}")


# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_type: str, vocab_size: int = 130000):
    """创建指定类型的模型
    
    Args:
        model_type: 模型类型 (fast_dllm, dream, llada, sdar)
        vocab_size: 词汇表大小
    
    Returns:
        (model, is_diffusion, vocab_size)
    """
    model_type = model_type.lower()
    
    if model_type == "fast_dllm":
        config = FastdLLMV2EdgeConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
        )
        model = FastdLLMV2Edge(config)
        
    elif model_type == "dream":
        config = DreamEdgeConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            attention_bias=True,
        )
        model = DreamEdge(config)
        
    elif model_type == "llada":
        config = LLaDAEdgeConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            mask_token_id=min(126336, vocab_size - 1),
        )
        model = LLaDAEdge(config)
        
    elif model_type == "sdar":
        config = SDAREdgeConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            attention_bias=False,
        )
        model = SDAREdge(config)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: fast_dllm, dream, llada, sdar")
    
    model.eval()
    return model, True, vocab_size


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: Optional[str], pte_path: Optional[str], model_type: str = "fast_dllm", max_seq_len: int = 32):
    """加载模型，返回 (engine, is_diffusion, vocab_size, diffusion_block_size)"""
    if pte_path:
        print(f"加载PTE模型: {pte_path}")
        try:
            # Load metadata if exists
            import json
            metadata_path = Path(pte_path).with_suffix('.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"  元数据: {metadata_path}")
            
            # Get model type and block size from metadata
            pte_model_type = metadata.get('model_type', model_type)
            internal_model_type = pte_model_type.replace('fast_dllm', 'fast_dllm_v2')
            diffusion_block_size = metadata.get('diffusion_block_size', 
                                                4 if pte_model_type == 'sdar' else 32)
            vocab_size = metadata.get('vocab_size', 151936)
            
            print(f"  模型类型: {pte_model_type}, block_size: {diffusion_block_size}")
            
            # PTE models exported with static shapes need max_seq_len
            engine = DiffusionEngine.from_pte(pte_path, model_type=internal_model_type, max_seq_len=max_seq_len)
            return engine, True, vocab_size, diffusion_block_size
        except Exception as e:
            print(f"加载PTE失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif model_path:
        print(f"加载模型: {model_path}")
        # TODO: 实现从checkpoint加载真实权重
        # 目前创建对应类型的演示模型
        model, is_diffusion, vocab_size = create_model(model_type)
        print(f"创建{model_type}模型完成")
        # Map CLI model type to internal model type
        internal_model_type = model_type.replace("fast_dllm", "fast_dllm_v2")
        engine = DiffusionEngine.from_model(model, model_type=internal_model_type)
        # Default block size based on model type
        diffusion_block_size = 4 if model_type == 'sdar' else 32
        return engine, is_diffusion, vocab_size, diffusion_block_size
    
    else:
        # 创建演示模型
        print(f"创建演示模型: {model_type}")
        model, is_diffusion, vocab_size = create_model(model_type)
        engine = DiffusionEngine.from_model(model)
        # Default block size based on model type
        diffusion_block_size = 4 if model_type == 'sdar' else 32
        return engine, is_diffusion, vocab_size, diffusion_block_size


def find_tokenizer_path(model_path: Optional[str]) -> Optional[str]:
    """查找tokenizer路径"""
    if not model_path:
        return None
    model_path = Path(model_path)
    if model_path.is_dir():
        return str(model_path)
    parent = model_path.parent
    for filename in ['tokenizer.json', 'tokenizer_config.json']:
        if (parent / filename).exists():
            return str(parent)
    return None


# ============================================================================
# Help
# ============================================================================

def print_help():
    """打印帮助"""
    help_text = """
交互命令:
  /help, /h           - 显示此帮助
  /quit, /q, /exit    - 退出程序
  /clear              - 清空对话历史
  /config             - 显示当前配置
  
生成参数调整:
  /temp <0.0-2.0>     - 设置temperature
  /topk <n>           - 设置top_k (0=禁用)
  /topp <0.0-1.0>     - 设置top_p
  /max <n>            - 设置最大生成token数
  /iter <n>           - 设置扩散迭代次数
  /conf <0.0-1.0>     - 设置置信度阈值
"""
    print(help_text)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DiffuLex Edge 交互式对话CLI - 支持4种扩散模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用演示模型 (FastdLLM)
  %(prog)s
  
  # 使用特定模型类型
  %(prog)s --model-type dream
  %(prog)s --model-type llada
  %(prog)s --model-type sdar
  
  # 加载真实模型
  %(prog)s --model-path model.pt --model-type fast_dllm
  
  # 加载PTE模型
  %(prog)s --pte-path model.pte
  
  # 使用HuggingFace Tokenizer
  %(prog)s --tokenizer gpt2
        """
    )
    
    parser.add_argument("--model-path", type=str, help="PyTorch模型路径")
    parser.add_argument("--pte-path", type=str, help="PTE模型路径")
    parser.add_argument("--model-type", type=str, default="fast_dllm",
                        choices=["fast_dllm", "dream", "llada", "sdar"],
                        help="模型类型 (默认: fast_dllm)")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer路径或HuggingFace模型名称")
    parser.add_argument("--max-seq-len", type=int, default=32, help="PTE模型固定序列长度 (默认: 32)")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K采样")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P采样")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备")
    parser.add_argument("--no-chat-template", action="store_true", help="禁用chat template")
    
    args = parser.parse_args()
    
    if args.model_path and args.pte_path:
        print("错误: 不能同时指定 --model-path 和 --pte-path")
        sys.exit(1)
    
    print("=" * 60)
    print("DiffuLex Edge - Interactive Chat CLI")
    print("=" * 60)
    print()
    
    # 加载模型
    try:
        engine, is_diffusion, vocab_size, diffusion_block_size = load_model(
            args.model_path, args.pte_path, args.model_type, 
            max_seq_len=args.max_seq_len
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 加载tokenizer
    tokenizer_path = args.tokenizer
    if not tokenizer_path and args.model_path:
        tokenizer_path = find_tokenizer_path(args.model_path)
        if tokenizer_path:
            print(f"自动检测到tokenizer: {tokenizer_path}")
    
    tokenizer = TokenizerWrapper(tokenizer_path, vocab_size=vocab_size)
    
    # 初始化配置 (使用模型指定的block_size)
    chat_config = ChatConfig(block_size=diffusion_block_size)
    chat_config.max_new_tokens = args.max_tokens
    chat_config.temperature = args.temperature
    chat_config.top_k = args.top_k
    chat_config.top_p = args.top_p
    chat_config.use_diffusion = is_diffusion
    chat_config.use_chat_template = not args.no_chat_template
    
    # 创建会话
    session = ChatSession(engine, chat_config, tokenizer, is_diffusion)
    
    print()
    print(f"模型类型: {args.model_type.upper()} ({'扩散' if is_diffusion else '自回归'})")
    print(f"Tokenizer: {'HuggingFace' if tokenizer.is_hf_tokenizer else '简单'}")
    print("输入 /help 查看可用命令，/quit 退出")
    print("-" * 60)
    
    # 主循环
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()
                if not cmd:
                    continue
                
                if cmd[0] in ["quit", "exit", "q"]:
                    print("再见!")
                    break
                elif cmd[0] in ["help", "h"]:
                    print_help()
                elif cmd[0] == "clear":
                    session.clear_history()
                    print("对话历史已清空")
                elif cmd[0] == "config":
                    print(session.config)
                elif cmd[0] == "temp" and len(cmd) > 1:
                    session.update_config(temperature=float(cmd[1]))
                elif cmd[0] == "topk" and len(cmd) > 1:
                    session.update_config(top_k=int(cmd[1]))
                elif cmd[0] == "topp" and len(cmd) > 1:
                    session.update_config(top_p=float(cmd[1]))
                elif cmd[0] == "max" and len(cmd) > 1:
                    session.update_config(max_new_tokens=int(cmd[1]))
                elif cmd[0] == "iter" and len(cmd) > 1:
                    session.update_config(num_iterations=int(cmd[1]))
                elif cmd[0] == "conf" and len(cmd) > 1:
                    session.update_config(confidence_threshold=float(cmd[1]))
                else:
                    print(f"未知命令: /{cmd[0]}")
                continue
            
            # 生成回复
            print("AI: ", end="", flush=True)
            try:
                response = session.generate(user_input)
                print(response)
            except Exception as e:
                print(f"\n[生成错误: {e}]")
                
        except KeyboardInterrupt:
            print("\n\n使用 /quit 退出")
        except EOFError:
            print("\n再见!")
            break
        except Exception as e:
            print(f"\n[错误: {e}]")


if __name__ == "__main__":
    main()
