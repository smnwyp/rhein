"""Network-free client with queued structured responses or errors."""
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
class FakeModelClient:
    def __init__(self, responses: Iterable[Mapping[str, Any] | Exception]) -> None: self._responses, self.requests = deque(responses), []
    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        self.requests.append(request)
        if not self._responses: raise ModelClientFailure("fake model client has no response remaining")
        value = self._responses.popleft()
        if isinstance(value, Exception): raise value
        return value
