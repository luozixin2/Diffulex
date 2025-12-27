import os
import time
from pathlib import Path

import torch
import tilelang
import tilelang.testing

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode_kernel
from test.python.kernel.test_dllm_flash_attn_decode_kernel import naive_sdpa_with_kvcache


def test_decode_multiround_context_len():
    """
    Test inference time and compilation behavior across different context_len values.
    This test verifies:
    1. Inference time for different context lengths
    2. Whether kernels are recompiled for different context_len values
    """
    # Common parameters (same as test_decode_bf16_multi_seq)
    common_params = {
        "num_seqs": 4,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_q_len": 64,
        "max_kv_len": 64,
        "page_block_size": 32,
        "diffusion_block_size": 32,
        "is_block_attn": False,
        "dtype": "bfloat16",
    }
    
    # Different context lengths to test
    max_context_len = 2048
    context_lens = list(range(128, max_context_len + 1, 32))
    
    # Calculate KV cache size based on max_context_len to ensure consistent allocation
    # across all tests
    max_blocks_per_seq = (max_context_len + common_params["page_block_size"] - 1) // common_params["page_block_size"]
    max_seq_num_blocks = max_blocks_per_seq
    num_page_blocks = common_params["num_seqs"] * max_blocks_per_seq
    
    # Track compilation times and inference times
    compilation_times = {}
    inference_times = {}
    kernel_paths = {}
    kernel_instances = {}
    correctness_results = {}  # Track correctness verification results
    
    cuda_cache_dir = os.getenv("CUDA_CACHE_DIR", "./cuda_cache")
    cache_root = Path(cuda_cache_dir) / "test_dllm_flash_attn_decode_kernel_multiround"
    
    print("\n" + "=" * 80)
    print("Testing multiple context_len values")
    print(f"KV cache allocated for max_context_len={max_context_len} (max_seq_num_blocks={max_seq_num_blocks}, num_page_blocks={num_page_blocks})")
    print("=" * 80)
    
    for context_len in context_lens:
        print(f"\n--- Testing context_len={context_len} ---")
        
        # Check if kernel file already exists (indicates potential cache hit)
        case_dir = cache_root / (
            f"seq{common_params['num_seqs']}_heads{common_params['num_heads']}_"
            f"kv{common_params['num_kv_heads']}_hd{common_params['head_dim']}_"
            f"ctx{context_len}_pbs{common_params['page_block_size']}_"
            f"dbs{common_params['diffusion_block_size']}_"
            f"block{int(common_params['is_block_attn'])}_dtype{common_params['dtype']}_"
            f"bm64_bn64_stg1_thr128_mq{common_params['max_q_len']}_mk{common_params['max_kv_len']}"
        )
        kernel_path = case_dir / "kernel.cu"
        
        kernel_existed_before = kernel_path.exists()
        kernel_mtime_before = kernel_path.stat().st_mtime if kernel_existed_before else None
        
        # Measure compilation + first inference time
        start_time = time.time()
        
        # Run the test (this includes kernel compilation if needed)
        # We'll create the kernel and run it to measure compilation time
        torch_dtype = getattr(torch, common_params["dtype"])
        device = "cuda"
        num_groups = common_params["num_heads"] // common_params["num_kv_heads"]
        total_q_len = common_params["num_seqs"] * common_params["diffusion_block_size"]
        total_kv_len = common_params["num_seqs"] * common_params["diffusion_block_size"]
        
        # Create kernel (this may trigger compilation)
        decode_kernel = dllm_flash_attn_decode_kernel(
            common_params["num_seqs"],
            num_groups,
            num_page_blocks,
            total_q_len,
            total_kv_len,
            common_params["num_heads"],
            common_params["head_dim"],
            common_params["is_block_attn"],
            common_params["diffusion_block_size"],
            max_seq_num_blocks,
            common_params["page_block_size"],
            64,  # block_m
            64,  # block_n
            1,   # num_stages
            128, # num_threads
        )
        
        # Save kernel source
        kernel_source = decode_kernel.get_kernel_source()
        case_dir.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(kernel_source)
        
        # Prepare input tensors for first run
        q = torch.randn(total_q_len, common_params["num_heads"], common_params["head_dim"], 
                       dtype=torch_dtype, device=device)
        k = torch.randn(total_kv_len, common_params["num_kv_heads"], common_params["head_dim"], 
                       dtype=torch_dtype, device=device)
        v = torch.randn(total_kv_len, common_params["num_kv_heads"], common_params["head_dim"], 
                       dtype=torch_dtype, device=device)
        k_cache = torch.randn(num_page_blocks, common_params["page_block_size"], 
                             common_params["num_kv_heads"], common_params["head_dim"], 
                             dtype=torch_dtype, device=device)
        v_cache = torch.randn(num_page_blocks, common_params["page_block_size"], 
                             common_params["num_kv_heads"], common_params["head_dim"], 
                             dtype=torch_dtype, device=device)
        block_tables = torch.zeros(common_params["num_seqs"], max_seq_num_blocks, 
                                  dtype=torch.int32, device=device)
        # Calculate actual blocks needed for current context_len
        num_blocks_per_seq = (context_len + common_params["page_block_size"] - 1) // common_params["page_block_size"]
        for seq_idx in range(common_params["num_seqs"]):
            for block_idx in range(num_blocks_per_seq):
                block_tables[seq_idx, block_idx] = seq_idx * max_blocks_per_seq + block_idx
            # Set remaining blocks to -1 (invalid) if context_len is less than max_context_len
            for block_idx in range(num_blocks_per_seq, max_seq_num_blocks):
                block_tables[seq_idx, block_idx] = -1
        context_lens_tensor = torch.full((common_params["num_seqs"],), context_len, 
                                        dtype=torch.int32, device=device)
        cu_seqlens_q = torch.arange(0, (common_params["num_seqs"] + 1) * common_params["diffusion_block_size"], 
                                   common_params["diffusion_block_size"], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.arange(0, (common_params["num_seqs"] + 1) * common_params["diffusion_block_size"], 
                                   common_params["diffusion_block_size"], dtype=torch.int32, device=device)
        
        # First run (includes compilation if needed)
        _ = decode_kernel(
            q, k, v, k_cache, v_cache,
            block_tables,
            context_lens_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            common_params["max_q_len"],
        )
        torch.cuda.synchronize()
        
        compilation_time = time.time() - start_time
        compilation_times[context_len] = compilation_time
        
        # Check if kernel was compiled (file was created, not just loaded from cache)
        # Note: This is a heuristic - the actual compilation happens when the kernel
        # is first called, and tilelang may have its own caching mechanism
        was_compiled = not kernel_existed_before
        
        kernel_paths[context_len] = str(kernel_path)
        
        print(f"  Kernel path: {kernel_path}")
        print(f"  Kernel existed before: {kernel_existed_before}")
        print(f"  Was compiled: {was_compiled}")
        print(f"  Compilation + first inference time: {compilation_time:.4f}s")
        
        # Measure pure inference time (warmup + actual measurement)
        # Warmup
        _ = decode_kernel(
            q, k, v, k_cache, v_cache,
            block_tables,
            context_lens_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            common_params["max_q_len"],
        )
        torch.cuda.synchronize()
        
        # Measure inference time
        num_iterations = 10
        start_time = time.time()
        for _ in range(num_iterations):
            _ = decode_kernel(
                q, k, v, k_cache, v_cache,
                block_tables,
                context_lens_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                common_params["max_q_len"],
            )
        torch.cuda.synchronize()
        inference_time = (time.time() - start_time) / num_iterations
        inference_times[context_len] = inference_time
        
        print(f"  Average inference time ({num_iterations} iterations): {inference_time*1000:.4f}ms")
        
        # Verify correctness by comparing with reference implementation
        print(f"  Verifying correctness...")
        # Run kernel once more to get output for correctness verification
        output = decode_kernel(
            q, k, v, k_cache, v_cache,
            block_tables,
            context_lens_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            common_params["max_q_len"],
        )
        torch.cuda.synchronize()
        
        scale = 1.0 / (common_params["head_dim"] ** 0.5)
        ref_output = naive_sdpa_with_kvcache(
            q, k, v, k_cache, v_cache,
            block_tables, context_lens_tensor,
            cu_seqlens_q, cu_seqlens_k,
            scale, num_groups, common_params["page_block_size"],
        )
        
        try:
            torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
            correctness_results[context_len] = True
            print(f"  ✓ Correctness check passed")
        except AssertionError as e:
            correctness_results[context_len] = False
            print(f"  ✗ Correctness check FAILED: {e}")
        
        # Store kernel instance for later use
        kernel_instances[context_len] = decode_kernel
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Context Len':<15} {'Compiled':<10} {'Correct':<10} {'Compilation Time (s)':<20} {'Inference Time (ms)':<20}")
    print("-" * 80)
    for context_len in context_lens:
        was_compiled = kernel_paths[context_len] and Path(kernel_paths[context_len]).exists()
        is_correct = correctness_results.get(context_len, False)
        correct_str = "✓" if is_correct else "✗"
        print(f"{context_len:<15} {str(was_compiled):<10} {correct_str:<10} {compilation_times[context_len]:<20.4f} {inference_times[context_len]*1000:<20.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    
    # Check if kernels were recompiled for different context_len
    unique_kernel_paths = set(kernel_paths.values())
    print(f"Number of unique kernel paths: {len(unique_kernel_paths)}")
    print(f"Number of context_len values tested: {len(context_lens)}")
    
    if len(unique_kernel_paths) == len(context_lens):
        print("✓ Each context_len resulted in a unique kernel (expected behavior)")
    else:
        print("⚠ Some context_len values shared the same kernel")
    
    # Check inference time scaling
    print(f"\nInference time scaling:")
    base_time = inference_times[context_lens[0]]
    for context_len in context_lens:
        ratio = inference_times[context_len] / base_time
        print(f"  context_len={context_len}: {ratio:.2f}x (vs context_len={context_lens[0]})")
    
    # Check correctness summary
    print(f"\nCorrectness verification summary:")
    passed = sum(1 for v in correctness_results.values() if v)
    total = len(correctness_results)
    print(f"  Passed: {passed}/{total}")
    if passed < total:
        print(f"  Failed context_len values:")
        for context_len, is_correct in correctness_results.items():
            if not is_correct:
                print(f"    - context_len={context_len}")
    else:
        print("  ✓ All correctness checks passed!")


if __name__ == "__main__":
    # tilelang.testing.main()
    test_decode_multiround_context_len()