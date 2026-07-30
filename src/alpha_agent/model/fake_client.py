"""Deterministic client for tests and examples."""

from collections import deque
from collections.abc import Iterable
from typing import Any

from alpha_agent.errors import ModelClientError


class FakeModelClient:
    def __init__(self, responses: Iterable[dict[str, Any]]) -> None:
        self._responses = deque(responses)
        self.requests: list[tuple[str, str]] = []

    def generate_structured(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.requests.append((system_prompt, user_prompt))
        if not self._responses:
            raise ModelClientError("fake model client has no response remaining")
        return self._responses.popleft()
