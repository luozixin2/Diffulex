"""KV Cache dtype utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

try:
    from vllm.platforms import current_platform
except Exception:
    current_platform = None


class KvCacheDType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP32 = 2
    FP8_E4M3 = 3
    FP8_E5M2 = 4


@dataclass(frozen=True)
class KvCacheDTypeSpec:
    enum: KvCacheDType
    is_fp8: bool
    fp8_view_dtype: torch.dtype | None
    fp8_min: float | None
    fp8_max: float | None


def _normalize_kv_cache_dtype(kv_cache_dtype: str) -> str:
    """Normalize dtype string to standard format."""
    s = (kv_cache_dtype or "").strip().lower()
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "fp8": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "e5m2": "fp8_e5m2",
    }
    if s not in aliases:
        raise ValueError(
            f"Unsupported kv_cache_dtype={kv_cache_dtype!r}. "
            "Supported: bf16/fp16/fp32/fp8/fp8_e4m3/fp8_e5m2"
        )
    return aliases[s]


def _get_fp8_e4m3_dtype() -> torch.dtype:
    """Get FP8 E4M3 dtype (platform-specific)."""
    if current_platform is None:
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        raise RuntimeError("FP8 requested but vLLM current_platform is unavailable.")
    return current_platform.fp8_dtype()


def _get_fp8_e5m2_dtype() -> torch.dtype:
    """Get FP8 E5M2 dtype."""
    if hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    if hasattr(torch, "float8_e5m2fnuz"):
        return torch.float8_e5m2fnuz
    raise RuntimeError("FP8 E5M2 dtype not available in this torch build.")


def parse_kv_cache_dtype(kv_cache_dtype: str) -> KvCacheDTypeSpec:
    """Parse kv_cache_dtype string into a spec."""
    norm = _normalize_kv_cache_dtype(kv_cache_dtype)
    if norm == "bf16":
        return KvCacheDTypeSpec(KvCacheDType.BF16, False, None, None, None)
    if norm == "fp16":
        return KvCacheDTypeSpec(KvCacheDType.FP16, False, None, None, None)
    if norm == "fp32":
        return KvCacheDTypeSpec(KvCacheDType.FP32, False, None, None, None)

    if norm == "fp8_e4m3":
        fp8 = _get_fp8_e4m3_dtype()
        enum = KvCacheDType.FP8_E4M3
    elif norm == "fp8_e5m2":
        fp8 = _get_fp8_e5m2_dtype()
        enum = KvCacheDType.FP8_E5M2
    else:
        raise AssertionError(norm)

    info = torch.finfo(fp8)
    return KvCacheDTypeSpec(
        enum=enum,
        is_fp8=True,
        fp8_view_dtype=fp8,
        fp8_min=float(info.min),
        fp8_max=float(info.max),
    )


def ensure_scale_tensor(
    scale: Any,
    *,
    num_kv_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert scale to tensor with shape [num_kv_heads]."""
    if scale is None:
        return torch.ones((num_kv_heads,), device=device, dtype=dtype)
    if isinstance(scale, (float, int)):
        return torch.full((num_kv_heads,), float(scale), device=device, dtype=dtype)
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 1:
            return torch.full((num_kv_heads,), float(scale.item()), device=device, dtype=dtype)
        if scale.numel() != num_kv_heads:
            raise ValueError(
                f"scale must be scalar or shape [num_kv_heads]={num_kv_heads}, got {tuple(scale.shape)}"
            )
        return scale.to(device=device, dtype=dtype).contiguous()
    raise TypeError(f"Unsupported scale type: {type(scale)}")


def view_fp8_cache(cache: torch.Tensor, kv_cache_dtype: str) -> torch.Tensor:
    """Return FP8 cache view with correct dtype."""
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    if not spec.is_fp8:
        return cache
    assert spec.fp8_view_dtype is not None
    if cache.dtype == torch.uint8:
        return cache.view(spec.fp8_view_dtype)
    if cache.dtype == spec.fp8_view_dtype:
        return cache
    raise AssertionError(
        f"FP8 cache must be torch.uint8 (storage) or {spec.fp8_view_dtype}, got {cache.dtype}"
    )
