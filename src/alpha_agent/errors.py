"""Explicit errors with machine-readable codes and detail dictionaries."""
from typing import Any
class AlphaAgentError(Exception):
    code = "alpha_agent_error"
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None: self.message, self.details = message, details or {}; super().__init__(message)
class ModelClientFailure(AlphaAgentError): code = "model_client_failure"
class ModelResponseParsingFailure(AlphaAgentError): code = "model_response_parsing_failure"
class InvalidDSLschema(AlphaAgentError): code = "invalid_dsl_schema"
class SemanticStrategyValidationFailure(AlphaAgentError):
    code = "semantic_strategy_validation_failure"
    def __init__(self, issues: list[dict[str, Any]]) -> None: self.issues = issues; super().__init__("strategy violates semantic validation", details={"issues": issues})
class UnsupportedStrategyFeature(AlphaAgentError): code = "unsupported_strategy_feature"
ModelClientError = ModelClientFailure
ModelResponseError = ModelResponseParsingFailure
StrategyValidationError = SemanticStrategyValidationFailure
