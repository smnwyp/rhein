"""Explicit application errors."""


class AlphaAgentError(Exception):
    """Base error for the package."""


class ModelClientError(AlphaAgentError):
    """A model client failed to produce a usable response."""


class ModelResponseError(ModelClientError):
    """The provider response did not match the requested contract."""


class StrategyValidationError(AlphaAgentError):
    """A structurally valid strategy violates domain semantics."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))
