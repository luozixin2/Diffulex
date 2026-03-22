from functools import partial

from .math_utils import is_equal, get_answer_str, solution2answer
from .code_exec import evaluate_code_function, evaluate_code_stdio

__all__ = [
    "is_equal",
    "get_answer_str",
    "solution2answer",
    "evaluate_code_function",
    "evaluate_code_stdio",
    "setup_postprocess_generate_until",
    "postprocess_generate_until",
    "strategy_config_yaml_path_from_lm_eval_model_args",
]


def strategy_config_yaml_path_from_lm_eval_model_args(model_args) -> str | None:
    """Resolve strategy_config_dict YAML path from lm_eval ``--model_args`` (dict or legacy string)."""
    if isinstance(model_args, dict):
        v = model_args.get("strategy_config_dict")
        return str(v) if v is not None else None
    if isinstance(model_args, str) and "strategy_config_dict=" in model_args:
        rest = model_args.split("strategy_config_dict=", 1)[1].strip()
        return rest.split(",")[0].strip() if "," in rest else rest
    return None


def _postprocess_generate_until(response: str, escape_until: bool, until_terms: list[str]) -> str:
    if not escape_until:
        for until_term in until_terms:
            response = response.split(until_term)[0]
    return response


_POSTPROCESS_GENERATE_UNTIL_FN = None


def setup_postprocess_generate_until(escape_until: bool, until_terms: list[str]):
    global _POSTPROCESS_GENERATE_UNTIL_FN
    _POSTPROCESS_GENERATE_UNTIL_FN = partial(_postprocess_generate_until, escape_until=escape_until, until_terms=until_terms)


def postprocess_generate_until(response: str) -> str:
    global _POSTPROCESS_GENERATE_UNTIL_FN
    if _POSTPROCESS_GENERATE_UNTIL_FN is None:
        raise ValueError("Postprocess generate until function not setup")
    return _POSTPROCESS_GENERATE_UNTIL_FN(response)
