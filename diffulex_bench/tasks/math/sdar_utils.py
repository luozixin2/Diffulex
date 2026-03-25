"""MATH500 prompts; scoring uses the same pipeline as gsm8k/sdar_utils.py."""
from diffulex_bench.tasks.common.lightning_math_prompts import MATH_4SHOT_EXAMPLES
from diffulex_bench.tasks.gsm8k.sdar_utils import (
    doc_to_text_math as doc_to_text_math_0shot,
    process_results_math,
)


def doc_to_text_math_4shot_math500(doc: dict) -> str:
    """MATH500 4-shot（LightningRL: MATH_4SHOT_EXAMPLES + 题面 + 指令）。"""
    q = doc["question"]
    return (
        MATH_4SHOT_EXAMPLES
        + f"{q}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
