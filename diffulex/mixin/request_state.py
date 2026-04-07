"""Cooperative pickle hooks for request mixins."""

from __future__ import annotations

from typing import Any


class ReqStateMixin:
    """Single pickle entrypoint for req classes composed from multiple mixins.

    Feature mixins should override `_customize_req_state()` and/or
    `_restore_req_runtime_state()` and always call `super()` to keep the chain
    intact.
    """

    def __getstate__(self) -> dict[str, Any]:
        return self._customize_req_state(self.__dict__.copy())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._restore_req_runtime_state()

    def _customize_req_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def _restore_req_runtime_state(self) -> None:
        pass
