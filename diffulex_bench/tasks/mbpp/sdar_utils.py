"""SDAR-compatible evaluation utilities for code tasks"""
import re
from typing import Any

from diffulex_bench.tasks.utils import evaluate_code_function, evaluate_code_stdio


def extract_code(text: str) -> str:
    """Extract code using LightningRL's 17-pattern matching"""
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
        r"BEGIN\s*(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*(.*)\s*DONE",
        r"BEGIN\s*(.*)\s*DONE",
        r"```python\s*(.*)\s*```",
        r"```\s*(.*)\s*```",
        r"```python\s*(.*)\s*$",
        r"```\s*(.*)\s*$",
        r"(.*)\s*```.*",
        r"\[BEGIN\]\s*'(.*)",
        r"\[BEGIN\](.*)",
        r"'(.*)'\s*\[DONE\]",
    ]

    for p in patterns:
        try:
            match = re.search(p, text, re.DOTALL)
            if match:
                text = match.group(1)
                break
        except Exception:
            pass

    text = text.split("```")[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace("\\_", "_")
    return text.strip()


def doc_to_text_code_function(doc: dict) -> str:
    """Format code problem (function mode)"""
    prefix = doc.get("prefix", "")
    problem = doc.get("text", doc.get("prompt", doc.get("question", "")))
    return f"<|im_start|>user\n{problem}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block.<|im_end|>\n<|im_start|>assistant\n{prefix}"


def doc_to_text_code_stdio(doc: dict) -> str:
    """Format code problem (stdio mode)"""
    problem = doc.get("text", doc.get("prompt", doc.get("question", "")))
    return f"<|im_start|>user\nThis is the problem:\n{problem}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script.<|im_end|>\n<|im_start|>assistant\n"


def process_results_code(doc: dict, results: list[str]) -> dict[str, Any]:
    """Process code results using LightningRL's execution"""
    prediction = results[0] if results else ""

    test_method = doc.get("test_method", "function")

    if test_method == "function":
        prefix = doc.get("prefix", "")
        code = extract_code(prefix + prediction)
        tests = doc.get("test_list", [])
        timeout = doc.get("test_time_limit", 1)

        correctness = evaluate_code_function(code, tests, timeout)
        passed = all(correctness)
    else:
        code = extract_code(prediction)
        test_inputs = doc.get("test_input", [])
        test_outputs = doc.get("test_output", [])
        timeout = doc.get("test_time_limit", 1)

        correctness = []
        for inp, out in zip(test_inputs, test_outputs):
            result = evaluate_code_stdio(code, inp, out, timeout)
            correctness.append(result)
        passed = all(correctness)

    return {
        "exact_match": int(passed),
    }
