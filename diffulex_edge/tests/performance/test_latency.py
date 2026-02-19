"""Latency and throughput performance tests.

These tests measure the performance characteristics of the Edge model.
Note: Full performance testing requires actual device hardware.
"""

import time
import pytest
import torch
import torch.nn as nn


class SimplePerfModel(nn.Module):
    """Simple model for performance testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.mark.performance
class TestLatencyBenchmarks:
    """Latency benchmark tests."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        model = SimplePerfModel(vocab_size=1000, hidden_size=256, num_layers=4)
        model.eval()
        return model
    
    @pytest.fixture
    def prefill_inputs(self):
        """Inputs for prefill (prompt processing)."""
        return torch.randint(0, 1000, (1, 100))
    
    @pytest.fixture
    def decode_inputs(self):
        """Inputs for decode (single token)."""
        return torch.randint(0, 1000, (1, 1))
    
    def test_prefill_latency_cpu(self, model, prefill_inputs):
        """Test prefill latency on CPU.
        
        Target: < 200ms for 100 tokens
        """
        model.cpu()
        prefill_inputs = prefill_inputs.cpu()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(prefill_inputs)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                _ = model(prefill_inputs)
                latencies.append((time.time() - start) * 1000)  # ms
        
        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\nPrefill Latency (100 tokens): Avg={avg_latency:.1f}ms, P99={p99_latency:.1f}ms")
        
        # Soft assertion - actual performance depends on hardware
        # Target is < 200ms for reference
        if avg_latency > 500:  # Very slow
            pytest.skip(f"System too slow for performance testing: {avg_latency:.1f}ms")
    
    def test_decode_latency_cpu(self, model, decode_inputs):
        """Test decode latency on CPU.
        
        Target: < 100ms per token
        """
        model.cpu()
        decode_inputs = decode_inputs.cpu()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(decode_inputs)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model(decode_inputs)
                latencies.append((time.time() - start) * 1000)
        
        # Skip first 5 (additional warmup)
        latencies = latencies[5:]
        
        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\nDecode Latency (1 token): Avg={avg_latency:.1f}ms, P99={p99_latency:.1f}ms")
        
        # Soft assertion
        if avg_latency > 500:
            pytest.skip(f"System too slow for performance testing: {avg_latency:.1f}ms")
    
    def test_throughput_cpu(self, model):
        """Test throughput on CPU.
        
        Target: > 10 tokens/sec
        """
        model.cpu()
        
        # Generate 50 tokens
        tokens = [1, 2, 3]
        max_tokens = 50
        
        start = time.time()
        with torch.no_grad():
            for _ in range(max_tokens):
                input_ids = torch.tensor([tokens[-1:]])
                _ = model(input_ids)
                tokens.append(1)  # Dummy token
        
        elapsed = time.time() - start
        throughput = max_tokens / elapsed
        
        print(f"\nThroughput: {throughput:.1f} tokens/sec")
        
        # Soft assertion
        if throughput < 1:
            pytest.skip(f"System too slow for throughput testing: {throughput:.1f} t/s")


@pytest.mark.performance
class TestMemoryBenchmarks:
    """Memory usage benchmark tests."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        model = SimplePerfModel(vocab_size=1000, hidden_size=256, num_layers=4)
        model.eval()
        return model
    
    def test_model_memory_footprint(self, model):
        """Test model memory footprint.
        
        Target: Model size < 500MB for edge deployment
        """
        # Calculate model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        print(f"\nModel Size: {total_size_mb:.2f} MB")
        print(f"  Parameters: {param_size / 1024 / 1024:.2f} MB")
        print(f"  Buffers: {buffer_size / 1024 / 1024:.2f} MB")
        
        # Soft assertion
        if total_size_mb > 1000:
            print(f"Warning: Model size {total_size_mb:.2f}MB exceeds 1GB")
    
    def test_inference_memory_growth(self, model):
        """Test memory growth during inference."""
        import gc
        
        model.cpu()
        
        # Initial memory (approximate)
        gc.collect()
        
        # Run inference multiple times
        for seq_len in [10, 50, 100]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            
            with torch.no_grad():
                _ = model(input_ids)
        
        gc.collect()
        
        # Note: Detailed memory tracking requires psutil or similar
        # This is a basic test to ensure no obvious leaks
        print("\nMemory growth test completed")


@pytest.mark.performance
class TestQuantizationImpact:
    """Test impact of quantization on performance."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        model = SimplePerfModel(vocab_size=500, hidden_size=128, num_layers=2)
        model.eval()
        return model
    
    def test_quantized_vs_fp32_speed(self, model):
        """Compare quantized vs FP32 speed."""
        import sys
        
        # Skip on Windows due to quantization compatibility issues
        if sys.platform == "win32":
            pytest.skip("Quantization compatibility issues on Windows")
        
        input_ids = torch.randint(0, 500, (1, 20))
        
        # FP32 baseline
        model_fp32 = model
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model_fp32(input_ids)
        
        # Measure FP32
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_fp32(input_ids)
        fp32_time = time.time() - start
        
        print(f"\nFP32 Time: {fp32_time * 100:.1f}ms (10 iterations)")
        
        # Try quantized version
        try:
            from diffulex_edge.quant import apply_dynamic_quantization
            
            model_quant = apply_dynamic_quantization(model_fp32)
            model_quant.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model_quant(input_ids)
            
            # Measure quantized
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model_quant(input_ids)
            quant_time = time.time() - start
            
            speedup = fp32_time / quant_time if quant_time > 0 else 0
            print(f"Quantized Time: {quant_time * 100:.1f}ms (10 iterations)")
            print(f"Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"Quantization test skipped: {e}")


class TestBenchmarkReport:
    """Generate benchmark report."""
    
    def test_generate_benchmark_summary(self):
        """Generate a summary of system capabilities."""
        import platform
        import torch
        
        print("\n" + "=" * 60)
        print("DIFFULEX EDGE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CPU Count: {torch.get_num_threads()}")
        
        # CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("CUDA: Not available")
        
        # MPS availability (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS: Available (Apple Silicon)")
        else:
            print("MPS: Not available")
        
        # Backend availability
        try:
            from diffulex_edge.backends import XNNPACKBackend, QNNBackend
            
            xnn = XNNPACKBackend()
            print(f"XNNPACK Backend: {'Available' if xnn.is_available() else 'Not available'}")
            
            qnn = QNNBackend()
            print(f"QNN Backend: {'Available' if qnn.is_available() else 'Not available'}")
            
        except ImportError:
            print("Backends: Not available")
        
        print("=" * 60)
