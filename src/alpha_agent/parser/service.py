"""Orchestration for the one Strategy Interpreter Agent."""
import re
import json
from collections.abc import Mapping
from time import perf_counter
from uuid import UUID, uuid4
from pydantic import TypeAdapter, ValidationError
from alpha_agent.domain.interpretation import ClarificationRequired, ParsedStrategy, StrategyInterpretationRequest, StrategyInterpretationResult
from alpha_agent.domain.semantic_inventory import SemanticInventory
from alpha_agent.errors import AlphaAgentError, ModelClientFailure, ModelResponseParsingFailure, SemanticStrategyValidationFailure
from alpha_agent.model.client import SemanticInventoryModelClient, StrategyModelClient
from alpha_agent.monitoring import InterpretationMonitoringEvent, InterpreterMonitor, fingerprint
from alpha_agent.parser.validation import validate_strategy
from alpha_agent.parser.completeness import enrich_request, preflight_clarifications, validate_coverage
from alpha_agent.parser.conformance import validate_source_conformance
from alpha_agent.parser.inventory import inventory_clarification, raise_inventory_unsupported, validate_inventory
from alpha_agent.interpretation_cache import ParsedInterpretationCache
_RESULT_ADAPTER = TypeAdapter(StrategyInterpretationResult)


_EXPLICIT_ONE_DAY_RETURN = re.compile(
    r"(?:相对(?:前一|上一)(?:个)?交易日(?:收盘价)?|单日).*?(?:涨幅|收益|回报).*?"
    r"(?:大于\s*0|>\s*0|0\s*<).*?(?:不超过|小于等于|≤|<=)\s*(\d+(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE,
)


def _normalize_provider_result(raw: Mapping[str, object]) -> dict[str, object]:
    """Remove only empty inactive fields leaked by the Bedrock transport envelope.

    Bedrock's tool schema must be non-recursive, whereas the DSL condition tree is
    intentionally recursive. Some models therefore return empty fields from the
    other result variant. Populated inactive fields and all unknown fields remain
    errors under the strict domain contract.
    """
    normalized = dict(raw)
    # Claude occasionally emits the requested inner StrategyDefinition itself,
    # rather than the surrounding interpretation-result envelope.  It is safe
    # to recognise only an unmistakable DSL root and adapt it; any unrelated
    # JSON continues through strict result parsing and is rejected.
    required_dsl_keys = {"schema_version", "frequency", "direction", "position_mode", "anchor", "entry", "exit_rules"}
    if "status" not in normalized and required_dsl_keys.issubset(normalized):
        normalized = {
            "status": "parsed",
            "strategy": normalized,
            "assumptions": [],
            "warnings": [{
                "code": "provider_direct_dsl_envelope_repair",
                "message": "模型返回了裸 DSL；系统已补齐解释结果信封，并将生成需人工核对的章节级映射。",
            }],
            "coverage": [],
        }
    inactive_empty_fields: dict[object, dict[str, object]] = {
        "parsed": {"partial_strategy": None, "questions": [], "ambiguous_terms": []},
        "clarification_required": {"strategy": None, "assumptions": [], "warnings": []},
    }
    for field_name, empty_value in inactive_empty_fields.get(normalized.get("status"), {}).items():
        if normalized.get(field_name, object()) == empty_value:
            normalized.pop(field_name, None)
    return _repair_provider_envelope_paths(_repair_anchor_indicator_encoding(normalized))


def _direct_dsl_fallback_coverage(raw: Mapping[str, object], request: StrategyInterpretationRequest) -> dict[str, object]:
    """Add transparent, coarse coverage only for an adapted provider bare DSL.

    The ordinary contract still requires model-produced, precise clause paths.
    This narrow fallback prevents a valid large DSL from being discarded merely
    because its provider omitted the outer envelope.  It maps strategy *sections*
    rather than pretending each formula received an exact atomic mapping, and
    makes that limitation visible in both warning and coverage explanation.
    """
    warnings = raw.get("warnings")
    if not (
        raw.get("status") == "parsed"
        and isinstance(warnings, list)
        and any(isinstance(note, Mapping) and note.get("code") == "provider_direct_dsl_envelope_repair" for note in warnings)
        and raw.get("coverage") == []
    ):
        return dict(raw)
    strategy = raw.get("strategy")
    if not isinstance(strategy, Mapping):
        return dict(raw)

    def available(*paths: str) -> list[str]:
        return [path for path in paths if path.split(".", 1)[0].split("[", 1)[0] in strategy]

    coverage: list[dict[str, object]] = []
    for clause in request.source_clauses:
        text = clause.text
        if re.search(r"(?:卖出|出场|止损|十字星|大阴线|高位|持仓管理|成交量)", text):
            paths = available("exit_rules", "persistent_states", "lifecycle_policy")
        elif re.search(r"(?:数据|周期|复权|执行|成交价格|手续费|滑点|日线)", text):
            paths = available("frequency", "data_requirement", "entry", "exit_rules")
        elif re.search(r"(?:持仓|停牌|涨跌停|无法成交|重新寻找|再(?:次)?入场|最长持仓)", text):
            paths = available("exit_rules", "lifecycle_policy", "persistent_states")
        else:
            paths = available("anchor", "entry")
        # v0.1 strategies cannot be bare timed DSLs, but keep the fallback
        # structurally total if future model variants broaden the recogniser.
        if not paths:
            paths = available("entry_condition", "exit_condition")
        coverage.append({
            "clause_id": clause.clause_id,
            "disposition": "mapped",
            "dsl_paths": paths,
            "explanation": "提供者漏掉逐条映射；系统按已验证 DSL 的对应策略章节建立回退映射，请在解释审阅中核对。",
        })
    repaired = dict(raw)
    repaired["coverage"] = coverage
    return repaired


def _discard_invalid_optional_partial_strategy(raw: Mapping[str, object]) -> dict[str, object]:
    """Keep usable clarification questions when only their optional draft is invalid.

    A provider sometimes identifies a real unresolved lifecycle choice but also
    emits an incomplete v0.3 ``partial_strategy``.  That draft is optional by
    contract, so returning it would turn a useful clarification into a generic
    parsing failure.  We discard it only when *every* Pydantic error is rooted
    in that optional field; errors in questions, coverage, or the envelope are
    never hidden.
    """
    if raw.get("status") != "clarification_required" or raw.get("partial_strategy") is None:
        return dict(raw)
    try:
        _RESULT_ADAPTER.validate_python(raw)
    except ValidationError as error:
        errors = error.errors(include_url=False)
        is_only_partial_draft_error = bool(errors) and all(
            len(item.get("loc", ())) >= 2
            and item["loc"][0] == "clarification_required"
            and item["loc"][1] == "partial_strategy"
            for item in errors
        )
        if is_only_partial_draft_error:
            repaired = dict(raw)
            repaired.pop("partial_strategy", None)
            return repaired
    return dict(raw)


def _repair_explicit_one_day_return_encoding(raw: Mapping[str, object], strategy_text: str) -> dict[str, object]:
    """Repair one known impossible encoding only when the source is explicit.

    This is not a default or an interpretation guess.  It applies only to the
    exact model failure ``close[t0] > close[t0]`` plus ``close[t0] <= 0`` when
    the user's source explicitly states a positive one-day return capped by a
    percentage.  The percentage is copied verbatim from that source.
    """
    matched = _EXPLICIT_ONE_DAY_RETURN.search(strategy_text)
    if matched is None:
        return dict(raw)
    result = dict(raw)
    candidate = result.get("strategy") if result.get("status") == "parsed" else None
    if not isinstance(candidate, Mapping) or candidate.get("schema_version") != "0.3":
        return result
    strategy = dict(candidate)
    entry = strategy.get("entry")
    if not isinstance(entry, Mapping):
        return result
    active_day = entry.get("active_day")
    condition = entry.get("condition")
    if not (isinstance(active_day, Mapping) and active_day.get("start_offset_days") == 0 and active_day.get("end_offset_days") == 0):
        return result
    if not (isinstance(condition, Mapping) and condition.get("node_type") == "group" and isinstance(condition.get("conditions"), list)):
        return result

    def is_market_close(value: object) -> bool:
        return isinstance(value, Mapping) and value.get("kind") == "market_field" and value.get("field") == "close"

    def is_anchor_close(value: object) -> bool:
        return isinstance(value, Mapping) and value.get("kind") == "anchor_market_field" and value.get("anchor") == "t0" and value.get("field") == "close"

    self_comparison_index: int | None = None
    price_cap_index: int | None = None
    for index, item in enumerate(condition["conditions"]):
        if not isinstance(item, Mapping) or item.get("node_type") != "comparison":
            continue
        left, right, operator = item.get("left"), item.get("right"), item.get("operator")
        if operator == "greater_than" and is_market_close(left) and is_anchor_close(right):
            self_comparison_index = index
        if operator == "less_than" and is_anchor_close(left) and is_market_close(right):
            self_comparison_index = index
        if operator in {"less_than", "less_than_or_equal"} and is_market_close(left) and isinstance(right, Mapping) and right.get("kind") == "scalar":
            price_cap_index = index
    if self_comparison_index is None:
        return result

    return_operand = {
        "kind": "anchor_indicator",
        "anchor": "t0",
        "offset_days": 0,
        "indicator": {"indicator": "rolling_return", "field": "close", "window": 1},
    }
    repaired_conditions = list(condition["conditions"])
    repaired_conditions[self_comparison_index] = {
        "node_type": "comparison", "operator": "greater_than",
        "left": return_operand, "right": {"kind": "scalar", "value": 0},
    }
    return_cap = {
        "node_type": "comparison", "operator": "less_than_or_equal",
        "left": return_operand, "right": {"kind": "scalar", "value": float(matched.group(1)) / 100},
    }
    if price_cap_index is None:
        repaired_conditions.append(return_cap)
    else:
        repaired_conditions[price_cap_index] = return_cap
    entry_copy = dict(entry)
    condition_copy = dict(condition)
    condition_copy["conditions"] = repaired_conditions
    entry_copy["condition"] = condition_copy
    strategy["entry"] = entry_copy
    result["strategy"] = strategy
    warnings = list(result.get("warnings", []))
    warnings.append({
        "code": "source_preserving_one_day_return_repair",
        "message": "已按原文明确的前一日涨幅区间，将不可能的 t0 自比较修复为 rolling_return(1)。",
    })
    result["warnings"] = warnings
    return result


def _repair_explicit_cross_encoding(raw: Mapping[str, object], strategy_text: str) -> dict[str, object]:
    """Fold an exact current/prior comparison pair into the source's cross node."""
    operator_data = (
        ("上穿", "greater_than", "less_than_or_equal", "cross_above"),
        ("下穿", "less_than", "greater_than_or_equal", "cross_below"),
    )
    selected = next((item for item in operator_data if item[0] in strategy_text), None)
    if selected is None:
        return dict(raw)
    _, current_operator, prior_operator, cross_operator = selected
    result = dict(raw)
    candidate = result.get("strategy") if result.get("status") == "parsed" else None
    if not isinstance(candidate, Mapping) or candidate.get("schema_version") not in {"0.2", "0.3"}:
        return result
    strategy = dict(candidate)
    anchor = strategy.get("anchor")
    if not isinstance(anchor, Mapping) or not isinstance(anchor.get("condition"), Mapping):
        return result
    condition = anchor["condition"]
    if condition.get("node_type") != "group" or not isinstance(condition.get("conditions"), list):
        return result
    conditions = list(condition["conditions"])
    # Pair each current comparison with a matching prior-day comparison.  Do
    # not retain just “the last current comparison”: an anchor often also has
    # MA5 > MA10 after the MA5/MA20 crossover pair, which previously caused us
    # to miss a valid earlier pair.
    match: tuple[int, int, Mapping[str, object], Mapping[str, object]] | None = None
    for current_index, current in enumerate(conditions):
        if not isinstance(current, Mapping) or current.get("node_type") != "comparison" or current.get("operator") != current_operator:
            continue
        current_left, current_right = current.get("left"), current.get("right")
        if not (isinstance(current_left, Mapping) and isinstance(current_right, Mapping) and current_left.get("kind") == current_right.get("kind") == "indicator"):
            continue
        for prior_index, prior in enumerate(conditions):
            if not isinstance(prior, Mapping) or prior.get("node_type") != "comparison" or prior.get("operator") != prior_operator:
                continue
            prior_left, prior_right = prior.get("left"), prior.get("right")
            if not (isinstance(prior_left, Mapping) and isinstance(prior_right, Mapping) and prior_left.get("kind") == prior_right.get("kind") == "lagged_indicator"):
                continue
            if (
                prior_left.get("offset_days") == prior_right.get("offset_days") == -1
                and prior_left.get("indicator") == current_left.get("indicator")
                and prior_right.get("indicator") == current_right.get("indicator")
            ):
                match = (current_index, prior_index, current_left, current_right)
                break
        if match is not None:
            break
    if match is None:
        return result
    current_index, prior_index, current_left, current_right = match
    repaired_conditions = [item for index, item in enumerate(conditions) if index != prior_index]
    adjusted_current_index = current_index - 1 if prior_index < current_index else current_index
    repaired_conditions[adjusted_current_index] = {
        "node_type": "cross", "operator": cross_operator,
        "left": dict(current_left), "right": dict(current_right),
    }
    condition_copy = dict(condition)
    condition_copy["conditions"] = repaired_conditions
    anchor_copy = dict(anchor)
    anchor_copy["condition"] = condition_copy
    strategy["anchor"] = anchor_copy
    result["strategy"] = strategy
    # Coverage paths are user-visible audit metadata.  This transformation
    # removes one list member, so migrate every path in the same atomic step.
    # Leaving stale positional paths is a code bug, never an LLM error.
    coverage = result.get("coverage")
    if isinstance(coverage, list):
        migrated_coverage: list[object] = []
        pattern = re.compile(r"^anchor\.condition\.conditions\[(\d+)\](.*)$")
        for coverage_item in coverage:
            if not isinstance(coverage_item, Mapping):
                migrated_coverage.append(coverage_item)
                continue
            migrated_item = dict(coverage_item)
            paths = migrated_item.get("dsl_paths")
            if isinstance(paths, list):
                migrated_paths: list[object] = []
                for path in paths:
                    match = pattern.match(path) if isinstance(path, str) else None
                    if match is None:
                        migrated_paths.append(path)
                        continue
                    original_index, suffix = int(match.group(1)), match.group(2)
                    if original_index == prior_index:
                        new_index = adjusted_current_index
                    elif original_index > prior_index:
                        new_index = original_index - 1
                    else:
                        new_index = original_index
                    migrated_paths.append(f"anchor.condition.conditions[{new_index}]{suffix}")
                migrated_item["dsl_paths"] = migrated_paths
            migrated_coverage.append(migrated_item)
        result["coverage"] = migrated_coverage
    warnings = list(result.get("warnings", []))
    warnings.append({
        "code": "source_preserving_cross_repair",
        "message": "已按原文明确的上穿/下穿语义，将当前与前一日的精确比较对折叠为 cross 节点。",
    })
    result["warnings"] = warnings
    return result


def _targeted_repair_instruction(
    request: StrategyInterpretationRequest,
    error: AlphaAgentError,
    candidate: StrategyInterpretationResult | None,
) -> str:
    """Focus an LLM repair on failing source clauses instead of the whole draft."""
    details = error.details
    raw_issues = details.get("issues") or details.get("validation_errors") or []
    source_ids = {
        match.group(1)
        for issue in raw_issues
        if isinstance(issue, Mapping)
        for match in [re.match(r"^source\.(C\d+)$", str(issue.get("path", "")))]
        if match is not None
    }
    if not source_ids or candidate is None:
        return (
            "Your previous candidate was rejected by deterministic validation. "
            "Reinterpret the original strategy from scratch; preserve every source clause and return a complete corrected result. "
            f"Validation details: {details}"
        )
    source_by_id = {clause.clause_id: clause.text for clause in request.source_clauses}
    coverage_by_id = {entry.clause_id: entry.dsl_paths for entry in candidate.coverage}
    targets = [
        {
            "clause_id": clause_id,
            "source_text": source_by_id.get(clause_id, ""),
            "current_dsl_paths": coverage_by_id.get(clause_id, []),
        }
        for clause_id in sorted(source_ids)
    ]
    strategy = candidate.strategy if isinstance(candidate, ParsedStrategy) else candidate.partial_strategy
    locked_strategy = strategy.model_dump(mode="json") if strategy is not None else None
    return (
        "TARGETED REPAIR MODE. Repair only the following failed source clause(s), and only their listed DSL paths. "
        "All other strategy fields and all other coverage entries are locked: preserve them exactly in semantic meaning. "
        "Return the complete corrected interpretation result, including complete coverage. "
        f"Targets: {json.dumps(targets, ensure_ascii=False)}\n"
        f"Validation details: {json.dumps(details, ensure_ascii=False)}\n"
        f"Locked strategy candidate: {json.dumps(locked_strategy, ensure_ascii=False, separators=(',', ':'))}"
    )


def _repair_provider_envelope_paths(raw: dict[str, object]) -> dict[str, object]:
    """Canonicalize coverage paths that unnecessarily include the result envelope.

    Coverage addresses the DSL value itself, but providers sometimes describe it
    as if it were nested under the outer result's `strategy` or
    `partial_strategy` field. Removing exactly that leading namespace is
    lossless; other path spelling errors still fail the coverage guardrail.
    """
    coverage = raw.get("coverage")
    if not isinstance(coverage, list):
        return raw
    result = dict(raw)
    prefixes = ("strategy.", "partial_strategy.")
    normalized_coverage: list[object] = []
    for item in coverage:
        if not isinstance(item, Mapping):
            normalized_coverage.append(item)
            continue
        entry = dict(item)
        paths = entry.get("dsl_paths")
        if isinstance(paths, list):
            entry["dsl_paths"] = [
                next((path[len(prefix):] for prefix in prefixes if isinstance(path, str) and path.startswith(prefix)), path)
                for path in paths
            ]
        normalized_coverage.append(entry)
    result["coverage"] = normalized_coverage
    return result


def _repair_anchor_indicator_encoding(raw: dict[str, object]) -> dict[str, object]:
    """Normalize a structurally misplaced but semantically exact v0.3 operand.

    Some providers emit `anchor_indicator(t0, -1)` in `anchor.condition` for
    MA20[t0] > MA20[t0-1]. Within an anchor condition, t0 is the current bar,
    so that representation is exactly the `lagged_indicator` operand. This is
    a lossless schema repair—not an inferred trading assumption—and is limited
    to this one location.
    """
    result = dict(raw)
    strategy_key = "strategy" if result.get("status") == "parsed" else "partial_strategy"
    candidate = result.get(strategy_key)
    if not isinstance(candidate, Mapping) or candidate.get("schema_version") != "0.3":
        return result
    strategy = dict(candidate)
    anchor = strategy.get("anchor")
    if not isinstance(anchor, Mapping):
        return result
    anchor_copy = dict(anchor)

    def repair_condition(node: object) -> object:
        if not isinstance(node, Mapping):
            return node
        repaired = dict(node)
        if repaired.get("node_type") == "group" and isinstance(repaired.get("conditions"), list):
            repaired["conditions"] = [repair_condition(child) for child in repaired["conditions"]]
        elif repaired.get("node_type") == "comparison":
            for side in ("left", "right"):
                operand = repaired.get(side)
                if isinstance(operand, Mapping) and operand.get("kind") == "anchor_indicator" and operand.get("anchor") == "t0":
                    lagged = dict(operand)
                    lagged["kind"] = "lagged_indicator"
                    lagged.pop("anchor", None)
                    repaired[side] = lagged
        return repaired

    anchor_copy["condition"] = repair_condition(anchor_copy.get("condition"))
    strategy["anchor"] = anchor_copy
    result[strategy_key] = strategy
    return result


class StrategyInterpreterService:
    def __init__(self, client: StrategyModelClient, *, monitor: InterpreterMonitor | None = None, inventory_client: SemanticInventoryModelClient | None = None, parsed_cache: ParsedInterpretationCache | None = None) -> None:
        self._client, self._monitor, self._inventory_client, self._parsed_cache = client, monitor, inventory_client, parsed_cache

    def interpret(self, request: StrategyInterpretationRequest) -> StrategyInterpretationResult:
        request_id, start = uuid4(), perf_counter()
        try:
            request = enrich_request(request)
            preflight = preflight_clarifications(request)
            if preflight is not None:
                self._record(request_id, request, preflight, perf_counter() - start)
                return preflight
            if self._parsed_cache is not None:
                cached = self._parsed_cache.get(request)
                if cached is not None:
                    try:
                        result = _RESULT_ADAPTER.validate_python(cached)
                        if not isinstance(result, ParsedStrategy):
                            raise ModelResponseParsingFailure("cached result is not a parsed strategy")
                        result = result.model_copy(update={"source_clauses": request.source_clauses})
                        validate_strategy(result.strategy)
                        strategy_payload = result.strategy.model_dump(mode="json")
                        validate_coverage(request.source_clauses, result.coverage, strategy_payload=strategy_payload, parsed=True)
                        validate_source_conformance(request.source_clauses, result.coverage, strategy_payload)
                    except (ModelResponseParsingFailure, SemanticStrategyValidationFailure, ValidationError):
                        # A cache entry cannot bypass a new or strengthened
                        # guardrail. Drop it and interpret normally.
                        self._parsed_cache.delete(request)
                    else:
                        self._record(request_id, request, result, perf_counter() - start, cache_hit=True)
                        return result
            if self._inventory_client is not None:
                inventory_request = request
                inventory: SemanticInventory | None = None
                for inventory_attempt in range(2):
                    try:
                        try:
                            inventory = SemanticInventory.model_validate(self._inventory_client.inspect_strategy(inventory_request))
                        except ValidationError as error:
                            raise ModelResponseParsingFailure(
                                "model response does not match the semantic-inventory contract",
                                details={"validation_errors": error.errors(include_url=False)},
                            ) from error
                        validate_inventory(inventory, request)
                        break
                    except ModelResponseParsingFailure as error:
                        if inventory_attempt == 1:
                            raise
                        inventory_request = inventory_request.model_copy(update={
                            "repair_instruction": (
                                "SEMANTIC INVENTORY REPAIR MODE. Your previous inventory was rejected by deterministic "
                                "validation. Preserve every valid semantic item, symbol, and finding, but ensure every "
                                "source clause Cnn is represented by at least one item, symbol, closure finding, or "
                                f"unsupported feature. Validation details: {json.dumps(error.details, ensure_ascii=False)}"
                            )
                        })
                assert inventory is not None
                if inventory.status == "clarification_required":
                    result = inventory_clarification(inventory, request)
                    self._record(request_id, request, result, perf_counter() - start)
                    return result
                if inventory.status == "unsupported":
                    raise_inventory_unsupported(inventory)
            repair_error: AlphaAgentError | None = None
            candidate_result: StrategyInterpretationResult | None = None
            for attempt in range(2):
                try:
                    raw = _normalize_provider_result(self._client.interpret_strategy(request))
                    raw = _direct_dsl_fallback_coverage(raw, request)
                    raw = _discard_invalid_optional_partial_strategy(raw)
                    raw = _repair_explicit_cross_encoding(raw, request.strategy_text)
                    raw = _repair_explicit_one_day_return_encoding(raw, request.strategy_text)
                    try:
                        result = _RESULT_ADAPTER.validate_python(raw)
                    except ValidationError as error:
                        raise ModelResponseParsingFailure("model response does not match the interpretation contract", details={"validation_errors": error.errors(include_url=False)}) from error
                    result = result.model_copy(update={"source_clauses": request.source_clauses})
                    candidate_result = result
                    strategy = result.strategy if isinstance(result, ParsedStrategy) else result.partial_strategy
                    if strategy is not None:
                        validate_strategy(strategy)
                    strategy_payload = strategy.model_dump(mode="json") if strategy is not None else None
                    validate_coverage(request.source_clauses, result.coverage, strategy_payload=strategy_payload, parsed=isinstance(result, ParsedStrategy))
                    if isinstance(result, ParsedStrategy) and strategy_payload is not None:
                        validate_source_conformance(request.source_clauses, result.coverage, strategy_payload)
                        if self._parsed_cache is not None:
                            self._parsed_cache.put(request, result.model_dump(mode="json"))
                    self._record(request_id, request, result, perf_counter() - start)
                    return result
                except (ModelResponseParsingFailure, SemanticStrategyValidationFailure) as error:
                    if attempt == 1:
                        raise
                    repair_error = error
                    request = request.model_copy(update={
                        "repair_instruction": _targeted_repair_instruction(request, error, candidate_result)
                    })
                except ModelClientFailure as error:
                    # A repair attempt must not hide the deterministic reason
                    # the original candidate was rejected (and this also keeps
                    # a failed provider retry from turning into a generic error).
                    if repair_error is not None:
                        raise repair_error from error
                    raise
            raise AssertionError("unreachable")
        except AlphaAgentError as error:
            self._record(request_id, request, None, perf_counter() - start, error.code); raise
        except Exception as error:
            wrapped = ModelClientFailure("model client raised an unexpected exception", details={"exception_type": type(error).__name__})
            self._record(request_id, request, None, perf_counter() - start, wrapped.code); raise wrapped from error
    def _record(self, request_id: UUID, request: StrategyInterpretationRequest, result: StrategyInterpretationResult | None, seconds: float, error_code: str | None = None, cache_hit: bool = False) -> None:
        if self._monitor is None: return
        outcome = "failed" if error_code else result.status  # type: ignore[union-attr]
        questions = len(result.questions) if isinstance(result, ClarificationRequired) else 0
        warnings = len(result.warnings) if isinstance(result, ParsedStrategy) else 0
        strategy = result.strategy if isinstance(result, ParsedStrategy) else (result.partial_strategy if isinstance(result, ClarificationRequired) else None)
        self._monitor.record(InterpretationMonitoringEvent(request_id=request_id, input_hash=fingerprint(request.strategy_text), input_character_count=len(request.strategy_text), symbol_supplied=request.symbol is not None, outcome=outcome, duration_ms=seconds * 1000, question_count=questions, warning_count=warnings, error_code=error_code, strategy_fingerprint=fingerprint(strategy.model_dump_json()) if strategy else None, cache_hit=cache_hit))
StrategyInterpreter = StrategyInterpreterService
