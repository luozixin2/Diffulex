import pytest


def test_linear_strategy_registry_bf16_pair():
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="bf16", act_dtype="bf16")
    assert s.linear_weight_format == "bf16"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_non_bf16_returns_stub():
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    assert s.linear_weight_format == "int8"
    assert s.linear_act_format == "bf16"


def test_factory_injects_linear_strategies_into_context():
    from dataclasses import dataclass

    from diffulex.utils.quantization.factory import QuantizationStrategyFactory
    from diffulex.utils.quantization.context import get_quantization_context

    @dataclass
    class DummyConfig:
        kv_cache_dtype: str = "bf16"
        attn_q_dtype: str = "bf16"
        linear_attn_weight_dtype: str = "bf16"
        linear_mlp_weight_dtype: str = "bf16"
        linear_attn_act_dtype: str = "bf16"
        linear_mlp_act_dtype: str = "bf16"

    ctx = QuantizationStrategyFactory.create_from_config(DummyConfig())
    assert ctx is get_quantization_context()
    assert ctx.get_linear_strategy("attn") is not None
    assert ctx.get_linear_strategy("mlp") is not None


def test_linear_forward_raises_on_stub(monkeypatch):
    # Avoid requiring torch.distributed process group init in unit tests.
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    from diffulex.layer.linear import ColumnParallelLinear
    from diffulex.utils.quantization.registry import create_linear_strategy
    from diffulex.utils.quantization.context import get_quantization_context

    # Install a stub strategy for attention linears.
    ctx = get_quantization_context()
    ctx.set_linear_strategy("attn", create_linear_strategy(weight_dtype="int8", act_dtype="bf16"))

    lin = ColumnParallelLinear(4, 8, bias=False, quant_kind="attn")
    # NOTE: default Linear weights are float32 unless a checkpoint loader overwrites them.
    # Keep dtypes consistent for this unit test.
    x = torch.randn(2, 4, dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        _ = lin(x)

    # Ensure bf16 path still works for other kinds.
    lin2 = ColumnParallelLinear(4, 8, bias=False, quant_kind="other")
    y = lin2(x)
    ref = F.linear(x, lin2.weight, None)
    assert torch.allclose(y, ref)


