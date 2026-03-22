"""SDAR-compatible evaluation utilities for math tasks"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

from diffulex_bench.tasks.utils import is_equal, get_answer_str


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} - compatible with LightningRL"""
    tag = r"\boxed{"
    start = text.rfind(tag)
    if start == -1:
        return text.strip()

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return "".join(buf) if depth == 0 else text.strip()


def ground_truth_string_from_doc(doc: dict) -> str:
    """HF GSM8K 用 `answer`；Lightning JSON 用 `ground_truth_answer`。`####` 切分与原先一致。"""
    answer_str = doc.get("answer", "")
    if answer_str is None or not str(answer_str).strip():
        answer_str = doc.get("ground_truth_answer", "")
    if answer_str is None:
        answer_str = ""
    if "####" in str(answer_str):
        return str(answer_str).split("####")[-1].strip()
    return str(answer_str).strip()


def doc_to_text_math(doc: dict) -> str:
    """Format math problem with SDAR prompt"""
    return f"<|im_start|>user\n{doc['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"


def process_results_math(doc: dict, results: list[str]) -> dict[str, Any]:
    """\\boxed{} 提取 + ground_truth_string_from_doc + is_equal（HF GSM8K 与 Lightning 数学 JSON 共用）。"""
    prediction = results[0] if results else ""
    extracted = extract_boxed_answer(prediction)
    ground_truth = ground_truth_string_from_doc(doc)

    executor = ThreadPoolExecutor(max_workers=1)

    async def check():
        return await is_equal(extracted, ground_truth, executor)

    correct = asyncio.run(check())

    return {
        "exact_match": int(correct),
    }
