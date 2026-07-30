"""Structured model-client boundary."""

from typing import Any, Protocol


class ModelClient(Protocol):
    def generate_structured(self, *, system_prompt: str, user_prompt: str) -> Any:
        """Return a JSON-compatible structured response."""
        ...
