"""Orchestration for the one Strategy Interpreter Agent."""
from collections.abc import Mapping
from time import perf_counter
from uuid import UUID, uuid4
from pydantic import TypeAdapter, ValidationError
from alpha_agent.domain.interpretation import ClarificationRequired, ParsedStrategy, StrategyInterpretationRequest, StrategyInterpretationResult
from alpha_agent.errors import AlphaAgentError, ModelClientFailure, ModelResponseParsingFailure
from alpha_agent.model.client import StrategyModelClient
from alpha_agent.monitoring import InterpretationMonitoringEvent, InterpreterMonitor, fingerprint
from alpha_agent.parser.validation import validate_strategy
from alpha_agent.parser.completeness import enrich_request, preflight_clarifications, validate_coverage
_RESULT_ADAPTER = TypeAdapter(StrategyInterpretationResult)


def _normalize_provider_result(raw: Mapping[str, object]) -> dict[str, object]:
    """Remove only empty inactive fields leaked by the Bedrock transport envelope.

    Bedrock's tool schema must be non-recursive, whereas the DSL condition tree is
    intentionally recursive. Some models therefore return empty fields from the
    other result variant. Populated inactive fields and all unknown fields remain
    errors under the strict domain contract.
    """
    normalized = dict(raw)
    inactive_empty_fields: dict[object, dict[str, object]] = {
        "parsed": {"partial_strategy": None, "questions": [], "ambiguous_terms": []},
        "clarification_required": {"strategy": None, "assumptions": [], "warnings": []},
    }
    for field_name, empty_value in inactive_empty_fields.get(normalized.get("status"), {}).items():
        if normalized.get(field_name, object()) == empty_value:
            normalized.pop(field_name, None)
    return normalized


class StrategyInterpreterService:
    def __init__(self, client: StrategyModelClient, *, monitor: InterpreterMonitor | None = None) -> None: self._client, self._monitor = client, monitor
    def interpret(self, request: StrategyInterpretationRequest) -> StrategyInterpretationResult:
        request_id, start = uuid4(), perf_counter()
        try:
            request = enrich_request(request)
            preflight = preflight_clarifications(request)
            if preflight is not None:
                self._record(request_id, request, preflight, perf_counter() - start)
                return preflight
            raw = self._client.interpret_strategy(request)
            try: result = _RESULT_ADAPTER.validate_python(_normalize_provider_result(raw))
            except ValidationError as error: raise ModelResponseParsingFailure("model response does not match the interpretation contract", details={"validation_errors": error.errors(include_url=False)}) from error
            result = result.model_copy(update={"source_clauses": request.source_clauses})
            strategy = result.strategy if isinstance(result, ParsedStrategy) else result.partial_strategy
            if strategy is not None: validate_strategy(strategy)
            validate_coverage(
                request.source_clauses,
                result.coverage,
                strategy_payload=strategy.model_dump(mode="json") if strategy is not None else None,
                parsed=isinstance(result, ParsedStrategy),
            )
            self._record(request_id, request, result, perf_counter() - start)
            return result
        except AlphaAgentError as error:
            self._record(request_id, request, None, perf_counter() - start, error.code); raise
        except Exception as error:
            wrapped = ModelClientFailure("model client raised an unexpected exception", details={"exception_type": type(error).__name__})
            self._record(request_id, request, None, perf_counter() - start, wrapped.code); raise wrapped from error
    def _record(self, request_id: UUID, request: StrategyInterpretationRequest, result: StrategyInterpretationResult | None, seconds: float, error_code: str | None = None) -> None:
        if self._monitor is None: return
        outcome = "failed" if error_code else result.status  # type: ignore[union-attr]
        questions = len(result.questions) if isinstance(result, ClarificationRequired) else 0
        warnings = len(result.warnings) if isinstance(result, ParsedStrategy) else 0
        strategy = result.strategy if isinstance(result, ParsedStrategy) else (result.partial_strategy if isinstance(result, ClarificationRequired) else None)
        self._monitor.record(InterpretationMonitoringEvent(request_id=request_id, input_hash=fingerprint(request.strategy_text), input_character_count=len(request.strategy_text), symbol_supplied=request.symbol is not None, outcome=outcome, duration_ms=seconds * 1000, question_count=questions, warning_count=warnings, error_code=error_code, strategy_fingerprint=fingerprint(strategy.model_dump_json()) if strategy else None))
StrategyInterpreter = StrategyInterpreterService
