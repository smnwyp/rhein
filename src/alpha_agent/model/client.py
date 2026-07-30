"""Provider-independent boundary for the Strategy Interpreter Agent."""
from collections.abc import Mapping
from typing import Any, Protocol
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
class StrategyModelClient(Protocol):
    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]: ...
