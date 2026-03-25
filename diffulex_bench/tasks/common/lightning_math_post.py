"""Backward-compat re-exports: canonical math eval lives in gsm8k/sdar_utils.py."""
from diffulex_bench.tasks.gsm8k.sdar_utils import (
    extract_boxed_answer,
    ground_truth_string_from_doc,
    process_results_math as process_results_math_lightning,
)

__all__ = [
    "extract_boxed_answer",
    "ground_truth_string_from_doc",
    "process_results_math_lightning",
]
