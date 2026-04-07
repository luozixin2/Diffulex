from diffulex_bench.config import BenchmarkConfig, EngineConfig, EvalConfig
from diffulex_bench.main import config_to_model_args


def test_engine_config_exposes_multi_block_prefix_full():
    engine = EngineConfig(
        model_path="/tmp/model",
        decoding_strategy="multi_bd",
        buffer_size=2,
        multi_block_prefix_full=True,
    )

    kwargs = engine.get_diffulex_kwargs()

    assert kwargs["multi_block_prefix_full"] is True
    assert kwargs["buffer_size"] == 2


def test_benchmark_config_to_model_args_includes_multi_block_prefix_full():
    config = BenchmarkConfig(
        engine=EngineConfig(
            model_path="/tmp/model",
            decoding_strategy="multi_bd",
            multi_block_prefix_full=True,
        ),
        eval=EvalConfig(),
    )

    model_args = config_to_model_args(config)

    assert "multi_block_prefix_full=True" in model_args
