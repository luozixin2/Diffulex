"""
Compare KV cache speed between FP8 and BF16.
Note: Run BF16 and FP8 tests separately to avoid process group initialization issues.
"""
import os
import sys
import time

from diffulex_legacy import LLM, SamplingParams
from transformers import AutoTokenizer


def test_kv_cache_speed(kv_cache_dtype="bf16", num_prompts=3):
    """Test generation speed with specified KV cache dtype."""
    print(f"\n{'='*80}")
    print(f"Testing with kv_cache_dtype='{kv_cache_dtype}'")
    print(f"{'='*80}")
    
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    # Initialize LLM
    print(f"\n[1/3] Initializing LLM with kv_cache_dtype='{kv_cache_dtype}'...")
    start_init = time.time()
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
    init_time = time.time() - start_init
    print(f"✓ Initialized in {init_time:.2f}s")
    
    # Generate text
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [
        tokenizer.bos_token + "Hello, how are you?",
        tokenizer.bos_token + "The capital of France is",
        tokenizer.bos_token + "Python is a programming language that",
    ][:num_prompts]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    print(f"\n[2/3] Generating text for {len(test_prompts)} prompts...")
    start_gen = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    gen_time = time.time() - start_gen
    
    # Collect stats
    total_tokens = sum(len(o.get("token_ids", [])) for o in outputs)
    
    print(f"\n[3/3] Results for kv_cache_dtype='{kv_cache_dtype}':")
    print(f"  - Generation time: {gen_time:.2f}s")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Throughput: {total_tokens/gen_time:.2f} tok/s")
    
    return {
        "kv_cache_dtype": kv_cache_dtype,
        "init_time": init_time,
        "gen_time": gen_time,
        "total_tokens": total_tokens,
        "throughput": total_tokens / gen_time,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific dtype from command line
        dtype = sys.argv[1]
        test_kv_cache_speed(dtype, num_prompts=3)
    else:
        # Default: test BF16
        print("Usage: python test_kv_cache_speed_comparison.py [bf16|fp8_e4m3]")
        print("Running BF16 test by default...\n")
        test_kv_cache_speed("bf16", num_prompts=3)

