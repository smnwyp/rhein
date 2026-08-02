"""Provider-independent boundary for the Strategy Interpreter Agent."""
from collections.abc import Mapping
from typing import Any, Protocol
from alpha_agent.domain.interpretation import StrategyInterpretationRequest


class StrategyModelClient(Protocol):
    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]: ...


class SemanticInventoryModelClient(Protocol):
    """Optional first-phase capability for complex strategy interpretation."""

    def inspect_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]: ...
