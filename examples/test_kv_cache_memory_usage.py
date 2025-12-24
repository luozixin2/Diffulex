"""
Test KV cache memory usage comparison between FP8 and BF16.
"""
import os
import sys
import torch
import gc

from diffulex_legacy import LLM, SamplingParams
from transformers import AutoTokenizer


def get_gpu_memory_info():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
        }
    return None


def test_kv_cache_memory(kv_cache_dtype="bf16"):
    """Test KV cache memory usage with specified dtype."""
    print(f"\n{'='*80}")
    print(f"Testing KV cache memory usage with kv_cache_dtype='{kv_cache_dtype}'")
    print(f"{'='*80}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    # Memory before initialization
    mem_before = get_gpu_memory_info()
    print(f"\n[Before initialization]")
    if mem_before:
        print(f"  GPU Memory - Allocated: {mem_before['allocated_mb']:.2f} MB, Reserved: {mem_before['reserved_mb']:.2f} MB")
    
    # Initialize LLM
    print(f"\n[1/4] Initializing LLM with kv_cache_dtype='{kv_cache_dtype}'...")
    llm = LLM(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        kv_cache_dtype=kv_cache_dtype,
    )
    
    # Memory after initialization (before generation)
    mem_after_init = get_gpu_memory_info()
    print(f"\n[After initialization, before generation]")
    if mem_after_init and mem_before:
        allocated_diff = mem_after_init['allocated_mb'] - mem_before['allocated_mb']
        reserved_diff = mem_after_init['reserved_mb'] - mem_before['reserved_mb']
        print(f"  GPU Memory - Allocated: {mem_after_init['allocated_mb']:.2f} MB (+{allocated_diff:.2f} MB)")
        print(f"  GPU Memory - Reserved: {mem_after_init['reserved_mb']:.2f} MB (+{reserved_diff:.2f} MB)")
        print(f"  Max Allocated: {mem_after_init['max_allocated_mb']:.2f} MB")
    
    # Get KV cache info from model_runner
    model_runner = llm.model_runner
    if hasattr(model_runner, 'kv_cache') and model_runner.kv_cache is not None:
        kv_cache = model_runner.kv_cache
        kv_cache_size_mb = kv_cache.element_size() * kv_cache.numel() / 1024**2
        print(f"\n[KV Cache Info]")
        print(f"  Shape: {kv_cache.shape}")
        print(f"  Dtype: {kv_cache.dtype}")
        print(f"  Element size: {kv_cache.element_size()} bytes")
        print(f"  Total elements: {kv_cache.numel()}")
        print(f"  Total size: {kv_cache_size_mb:.2f} MB")
        print(f"  Number of blocks: {model_runner.config.num_kvcache_blocks}")
    else:
        print(f"\n[KV Cache Info] KV cache not accessible directly")
        kv_cache_size_mb = None
    
    # Generate a small batch to trigger KV cache usage
    print(f"\n[2/4] Running small generation to ensure KV cache is used...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [tokenizer.bos_token + "Hello"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=10)
    outputs = llm.generate(test_prompts, sampling_params)
    
    # Memory after generation
    mem_after_gen = get_gpu_memory_info()
    print(f"\n[After generation]")
    if mem_after_gen:
        print(f"  GPU Memory - Allocated: {mem_after_gen['allocated_mb']:.2f} MB")
        print(f"  GPU Memory - Reserved: {mem_after_gen['reserved_mb']:.2f} MB")
        print(f"  Max Allocated: {mem_after_gen['max_allocated_mb']:.2f} MB")
    
    # Calculate KV cache memory from model config
    config = model_runner.config
    if hasattr(config, 'num_kvcache_blocks') and config.num_kvcache_blocks > 0:
        # Calculate expected KV cache size
        # KV cache shape: [2 (k/v), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        hf_config = config.hf_config
        num_layers = hf_config.num_hidden_layers
        block_size = config.kvcache_block_size
        num_blocks = config.num_kvcache_blocks
        
        # Get head_dim and num_kv_heads from model
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            head_dim = 128  # default fallback
        
        num_kv_heads = getattr(hf_config, 'num_key_value_heads', getattr(hf_config, 'num_attention_heads', 32))
        
        # Calculate based on dtype
        from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype
        spec = parse_kv_cache_dtype(kv_cache_dtype)
        itemsize = 1 if spec.is_fp8 else (2 if kv_cache_dtype in ['bf16', 'fp16'] else 4)
        
        expected_kv_cache_elements = 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim
        expected_kv_cache_size_mb = expected_kv_cache_elements * itemsize / 1024**2
        
        # Also calculate per-block size for comparison
        elements_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim
        size_per_block_mb = elements_per_block * itemsize / 1024**2
        
        print(f"\n[Expected KV Cache Size Calculation]")
        print(f"  num_layers: {num_layers}")
        print(f"  num_blocks: {num_blocks}")
        print(f"  block_size: {block_size}")
        print(f"  num_kv_heads: {num_kv_heads}")
        print(f"  head_dim: {head_dim}")
        print(f"  itemsize: {itemsize} bytes (for {kv_cache_dtype})")
        print(f"  Elements per block: {elements_per_block}")
        print(f"  Size per block: {size_per_block_mb:.2f} MB")
        print(f"  Total elements: {expected_kv_cache_elements}")
        print(f"  Total size: {expected_kv_cache_size_mb:.2f} MB")
    
    return {
        "kv_cache_dtype": kv_cache_dtype,
        "mem_before": mem_before,
        "mem_after_init": mem_after_init,
        "mem_after_gen": mem_after_gen,
        "kv_cache_size_mb": kv_cache_size_mb,
        "num_blocks": getattr(model_runner.config, 'num_kvcache_blocks', None),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dtype = sys.argv[1]
        result = test_kv_cache_memory(dtype)
        print(f"\n{'='*80}")
        print(f"SUMMARY for {dtype}:")
        print(f"{'='*80}")
        if result['kv_cache_size_mb']:
            print(f"KV Cache Size: {result['kv_cache_size_mb']:.2f} MB")
        if result['num_blocks']:
            print(f"Number of blocks: {result['num_blocks']}")
        if result['mem_after_init']:
            print(f"GPU Memory after init: {result['mem_after_init']['allocated_mb']:.2f} MB")
    else:
        print("Usage: python test_kv_cache_memory_usage.py [bf16|fp8_e4m3]")
        print("Running BF16 test by default...\n")
        result_bf16 = test_kv_cache_memory("bf16")
        
        print("\n\n" + "="*80)
        print("Now testing FP8...")
        print("="*80)
        # Need to restart Python process to avoid process group issues
        print("\nNote: Please run with 'fp8_e4m3' argument separately to test FP8")
        print("      Due to process group initialization, cannot test both in same process")

