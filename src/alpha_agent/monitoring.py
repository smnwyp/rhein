"""Privacy-preserving local monitoring for interpreter quality and reliability."""
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
from statistics import fmean
from typing import Literal, Protocol
from uuid import UUID, uuid4
from pydantic import Field, NonNegativeFloat, NonNegativeInt
from alpha_agent.domain.indicators import DSLModel

class InterpretationMonitoringEvent(DSLModel):
    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: UUID
    input_hash: str
    input_character_count: NonNegativeInt
    symbol_supplied: bool
    outcome: Literal["parsed", "clarification_required", "failed"]
    duration_ms: NonNegativeFloat
    question_count: NonNegativeInt = 0
    warning_count: NonNegativeInt = 0
    error_code: str | None = None
    strategy_fingerprint: str | None = None
    cache_hit: bool = False
class InterpretationEvaluation(DSLModel):
    """Human or offline-evaluation judgment; never inferred from model output."""
    request_id: UUID
    is_correct: bool
    category: Literal["parse", "clarification", "unsupported_feature", "safety"]
    note: str | None = None
class InterpreterMonitor(Protocol):
    def record(self, event: InterpretationMonitoringEvent) -> None: ...
class InMemoryInterpreterMonitor:
    def __init__(self) -> None: self.events: list[InterpretationMonitoringEvent] = []; self.evaluations: list[InterpretationEvaluation] = []
    def record(self, event: InterpretationMonitoringEvent) -> None: self.events.append(event)
    def record_evaluation(self, evaluation: InterpretationEvaluation) -> None: self.evaluations.append(evaluation)
    def summary(self) -> Mapping[str, object]:
        totals, failures, durations = Counter(e.outcome for e in self.events), Counter(e.error_code for e in self.events if e.error_code), [e.duration_ms for e in self.events]
        n = len(self.events)
        judged = len(self.evaluations)
        correct = sum(item.is_correct for item in self.evaluations)
        cache_hits = sum(event.cache_hit for event in self.events)
        return {"total_requests": n, "parsed_rate": totals["parsed"] / n if n else 0.0, "clarification_rate": totals["clarification_required"] / n if n else 0.0, "failure_rate": totals["failed"] / n if n else 0.0, "average_duration_ms": fmean(durations) if durations else 0.0, "cache_hits": cache_hits, "cache_hit_rate": cache_hits / n if n else 0.0, "failure_counts_by_code": dict(failures), "evaluated_requests": judged, "evaluation_accuracy": correct / judged if judged else None}
def fingerprint(value: str) -> str: return sha256(value.encode("utf-8")).hexdigest()
