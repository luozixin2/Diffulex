"""KV Cache dtype utilities (re-export from quantization module)."""

from diffulex.utils.quantization.kv_cache_dtype import (
    KvCacheDType,
    KvCacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
    _normalize_kv_cache_dtype,
    _get_fp8_e4m3_dtype,
    _get_fp8_e5m2_dtype,
)

__all__ = [
    'KvCacheDType',
    'KvCacheDTypeSpec',
    'parse_kv_cache_dtype',
    'ensure_scale_tensor',
    'view_fp8_cache',
]
