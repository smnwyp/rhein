from copy import deepcopy

import pytest
from pydantic import TypeAdapter
from alpha_agent.domain.interpretation import ClarificationRequired, ParsedStrategy, StrategyInterpretationRequest, StrategyInterpretationResult
from alpha_agent.errors import ModelClientFailure, ModelResponseParsingFailure, SemanticStrategyValidationFailure
from alpha_agent.model.fake_client import FakeModelClient
from alpha_agent.monitoring import InMemoryInterpreterMonitor, InterpretationEvaluation
from alpha_agent.interpretation_cache import JsonParsedInterpretationCache
from alpha_agent.parser.service import StrategyInterpreterService, _direct_dsl_fallback_coverage, _discard_invalid_optional_partial_strategy, _normalize_provider_result, _repair_explicit_cross_encoding, _repair_explicit_one_day_return_encoding, _repair_explicit_prior_window_extremum
from alpha_agent.parser.completeness import segment_source_clauses
from alpha_agent.parser.prompts import INVENTORY_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_interpreter_prompts_require_chinese_user_facing_clarifications():
    assert "Simplified Chinese" in SYSTEM_PROMPT
    assert "English question" in SYSTEM_PROMPT
    assert "Simplified" in INVENTORY_SYSTEM_PROMPT

def f(name): return {"kind":"market_field","field":name}
def i(kind, window): return {"kind":"indicator","indicator":{"indicator":kind,"field":"volume" if kind == "rolling_mean" else "close","window":window}}
def s(value): return {"kind":"scalar","value":value}
def cmp(left,right,op="greater_than"): return {"node_type":"comparison","operator":op,"left":left,"right":right}
def cross(left,right,op): return {"node_type":"cross","operator":op,"left":left,"right":right}
def group(op,*items): return {"node_type":"group","operator":op,"conditions":list(items)}
def parsed(entry, exit): return {"status":"parsed","strategy":{"schema_version":"0.1","symbol":"AAPL","frequency":"1d","direction":"long_only","position_mode":"fully_invested_or_flat","entry_condition":entry,"exit_condition":exit},"assumptions":[],"warnings":[],"coverage":[{"clause_id":"C01","disposition":"mapped","dsl_paths":["entry_condition","exit_condition"],"explanation":"The supplied test clause maps to both conditions."}]}
@pytest.mark.parametrize("response", [
    parsed(cross(f("close"),i("sma",20),"cross_above"),cross(f("close"),i("sma",20),"cross_below")),
    parsed(group("and",cross(i("sma",10),i("sma",50),"cross_above"),cmp(f("volume"),i("rolling_mean",20))),cross(i("sma",10),i("sma",50),"cross_below")),
    parsed(cmp(i("rsi",14),s(30),"less_than"),cmp(i("rsi",14),s(70))),
    parsed(group("and",cmp(i("rolling_return",20),s(.05)),cmp(f("volume"),{"kind":"scaled_operand","operand":i("rolling_mean",20),"multiplier":1.5})),cmp(i("rolling_return",20),s(0),"less_than")),
])
def test_required_successful_examples(response):
    result = StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test",symbol="AAPL"))
    assert isinstance(result, ParsedStrategy)
def test_clarification_and_partial_strategy_are_validated():
    response = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"decline","question":"What period and percent?","target_path":"entry_condition","suggested_answers":["5-day return below -5%"],"answer_kind":"choice"}],"ambiguous_terms":["fallen significantly","volume increasing"],"coverage":[{"clause_id":"C01","disposition":"clarification_required","dsl_paths":[],"explanation":"The test clause remains ambiguous."}]}
    assert isinstance(StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="ambiguous")), ClarificationRequired)
    invalid = {**response,"partial_strategy":parsed(cmp(i("rsi",14),s(101)),cmp(f("close"),i("sma",20)))["strategy"]}
    with pytest.raises(SemanticStrategyValidationFailure): StrategyInterpreterService(FakeModelClient([invalid])).interpret(StrategyInterpretationRequest(strategy_text="ambiguous"))


def test_invalid_optional_clarification_draft_is_discarded_but_questions_are_preserved():
    raw = {
        "status": "clarification_required",
        "partial_strategy": {
            "schema_version": "0.3", "symbol": "AAON", "frequency": "1d",
            "direction": "long_only", "position_mode": "fully_invested_or_flat",
            "data_requirement": "daily_ohlcv", "anchor": {"name": "t0", "condition": cmp(f("close"), i("sma", 5)), "constraints": []},
            "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": cmp(f("close"), i("sma", 5)), "execution": "close"},
            "exit_rules": [],
        },
        "questions": [{"question_id": "sample_end", "question": "How should the sample end be handled?", "target_path": "lifecycle_policy.sample_end_open_position", "suggested_answers": ["Force close"], "answer_kind": "choice"}],
        "ambiguous_terms": ["sample-end position handling"],
    }
    normalized = _discard_invalid_optional_partial_strategy(raw)
    assert "partial_strategy" not in normalized
    assert isinstance(TypeAdapter(StrategyInterpretationResult).validate_python(normalized), ClarificationRequired)


def test_empty_provider_transport_fields_are_normalized_but_populated_ones_are_rejected():
    response = parsed(cmp(i("rsi",14),s(30),"less_than"),cmp(i("rsi",14),s(70)))
    response.update({"partial_strategy":None,"questions":[],"ambiguous_terms":[]})
    assert isinstance(StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test")), ParsedStrategy)
    response["questions"] = [{"question_id":"unexpected","question":"Unexpected","target_path":"entry_condition","suggested_answers":[],"answer_kind":"choice"}]
    with pytest.raises(ModelResponseParsingFailure): StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test"))


def test_provider_bare_dsl_is_adapted_with_visible_section_level_coverage():
    bare = {
        "schema_version": "0.3", "symbol": "AAON", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": cmp(f("close"), i("sma", 5)), "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": cmp(f("close"), i("sma", 5), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "leave_open_excluded", "allow_reentry_after_exit": True},
    }
    request = StrategyInterpretationRequest(strategy_text="日线数据。\n入场：C 点。\n卖出：跌破均线。", source_clauses=segment_source_clauses("日线数据。\n入场：C 点。\n卖出：跌破均线。"))
    adapted = _direct_dsl_fallback_coverage(_normalize_provider_result(bare), request)
    result = StrategyInterpreterService(FakeModelClient([bare])).interpret(request)
    assert isinstance(result, ParsedStrategy)
    assert len(adapted["coverage"]) == len(request.source_clauses)
    assert result.warnings[0].code == "provider_direct_dsl_envelope_repair"
    assert any(item.dsl_paths == ["anchor", "entry"] for item in result.coverage)


def test_bare_dsl_records_daily_execution_and_volume_proxy_assumptions():
    bare = {
        "schema_version": "0.3", "symbol": "AAON", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": cmp(f("close"), i("sma", 5)), "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": cmp(f("close"), i("sma", 5), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "leave_open_excluded", "allow_reentry_after_exit": True},
    }
    text = "成交价为收盘前 10 分钟价格。成交量使用分钟级累计量，缺失时按 96% 折算。"
    request = StrategyInterpretationRequest(strategy_text=text, source_clauses=segment_source_clauses(text))

    result = StrategyInterpreterService(FakeModelClient([bare])).interpret(request)

    assert isinstance(result, ParsedStrategy)
    assert {note.code for note in result.assumptions} == {"daily_close_execution_proxy", "daily_volume_proxy"}
    assert all(item.disposition == "assumption" for item in result.coverage)


def test_malformed_and_client_failures_are_typed_and_monitored():
    monitor = InMemoryInterpreterMonitor(); request = StrategyInterpretationRequest(strategy_text="x")
    with pytest.raises(ModelResponseParsingFailure): StrategyInterpreterService(FakeModelClient([{"status":"parsed"}]), monitor=monitor).interpret(request)
    with pytest.raises(ModelClientFailure): StrategyInterpreterService(FakeModelClient([RuntimeError("network")]), monitor=monitor).interpret(request)
    assert monitor.summary()["failure_rate"] == 1.0
    assert all(event.input_hash != "x" for event in monitor.events)
    monitor.record_evaluation(InterpretationEvaluation(request_id=monitor.events[0].request_id, is_correct=True, category="safety"))
    assert monitor.summary()["evaluation_accuracy"] == 1.0


def test_interpreter_retries_one_repair_with_structured_validation_feedback():
    corrected = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    client = FakeModelClient([{"status": "parsed"}, corrected])
    result = StrategyInterpreterService(client).interpret(StrategyInterpretationRequest(strategy_text="buy then sell", symbol="AAPL"))
    assert isinstance(result, ParsedStrategy)
    assert len(client.requests) == 2
    assert client.requests[0].repair_instruction is None
    assert client.requests[1].repair_instruction is not None
    assert "Validation details" in client.requests[1].repair_instruction


def test_interpreter_reports_auditable_phase_progress_without_affecting_result():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    phases: list[str] = []
    result = StrategyInterpreterService(FakeModelClient([response]), progress=phases.append).interpret(
        StrategyInterpretationRequest(strategy_text="buy then sell", symbol="AAPL")
    )
    assert isinstance(result, ParsedStrategy)
    assert phases[0] == "正在切分原始策略条款…"
    assert "正在执行 Pydantic schema 校验…" in phases
    assert "正在执行策略语义与时序校验…" in phases
    assert phases[-1] == "解释完成。"


def test_validated_parsed_result_is_reused_without_any_model_call(tmp_path):
    cache = JsonParsedInterpretationCache(tmp_path / "parsed_interpretations.json")
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    first_client = FakeModelClient([response])
    request = StrategyInterpretationRequest(strategy_text="buy then sell", symbol="AAPL")
    first = StrategyInterpreterService(first_client, parsed_cache=cache).interpret(request)
    assert isinstance(first, ParsedStrategy)
    assert len(first_client.requests) == 1

    monitor = InMemoryInterpreterMonitor()
    second_client = FakeModelClient([])
    second = StrategyInterpreterService(second_client, parsed_cache=cache, monitor=monitor).interpret(request)
    assert isinstance(second, ParsedStrategy)
    assert not second_client.requests
    assert monitor.events[-1].cache_hit is True
    assert monitor.summary()["cache_hits"] == 1


def test_invalid_cached_result_is_deleted_and_cannot_bypass_validation(tmp_path):
    cache = JsonParsedInterpretationCache(tmp_path / "parsed_interpretations.json")
    request = StrategyInterpretationRequest(strategy_text="buy then sell", symbol="AAPL")
    # This raw result cannot satisfy the typed result contract, so the service
    # must discard it and use the model once rather than returning it.
    cache.put(request.model_copy(update={"source_clauses": segment_source_clauses(request.strategy_text)}), {"status": "parsed"})
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    client = FakeModelClient([response])
    result = StrategyInterpreterService(client, parsed_cache=cache).interpret(request)
    assert isinstance(result, ParsedStrategy)
    assert len(client.requests) == 1


def test_source_validation_failure_uses_targeted_clause_repair_instruction():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    client = FakeModelClient([response, response])
    service = StrategyInterpreterService(client)
    with pytest.raises(SemanticStrategyValidationFailure):
        # Force a source-scoped validation failure after a valid candidate has
        # been parsed, without relying on a live model response.
        from alpha_agent.parser import service as service_module
        original = service_module.validate_source_conformance
        def fail_source(*_args, **_kwargs):
            raise SemanticStrategyValidationFailure([{"path": "source.C01", "rule": "test", "message": "target this clause"}])
        service_module.validate_source_conformance = fail_source
        try:
            service.interpret(StrategyInterpretationRequest(strategy_text="Buy when close is above SMA(20).", symbol="AAPL"))
        finally:
            service_module.validate_source_conformance = original
    assert len(client.requests) == 2
    instruction = client.requests[1].repair_instruction
    assert instruction is not None
    assert "TARGETED REPAIR MODE" in instruction
    assert '"clause_id": "C01"' in instruction
    assert "Buy when close is above SMA(20)." in instruction


def test_explicit_one_day_return_source_repair_is_narrow_and_machine_visible():
    raw = {
        "status": "parsed",
        "strategy": {
            "schema_version": "0.3",
            "entry": {
                "active_day": {"start_offset_days": 0, "end_offset_days": 0},
                "condition": {"node_type": "group", "operator": "and", "conditions": [
                    cmp(f("close"), {"kind": "anchor_market_field", "anchor": "t0", "field": "close"}),
                    cmp(f("close"), s(0), "less_than_or_equal"),
                ]},
            },
        },
        "warnings": [],
    }
    repaired = _repair_explicit_one_day_return_encoding(
        raw,
        "C 点当日相对前一交易日收盘价小幅上涨，涨幅大于 0 且不超过 3%。",
    )
    conditions = repaired["strategy"]["entry"]["condition"]["conditions"]
    assert conditions[0]["left"]["kind"] == "anchor_indicator"
    assert conditions[0]["left"]["indicator"]["indicator"] == "rolling_return"
    assert conditions[1]["right"]["value"] == 0.03
    assert repaired["warnings"][0]["code"] == "source_preserving_one_day_return_repair"
    assert _repair_explicit_one_day_return_encoding(raw, "价格突破后买入") == raw

    without_model_cap = deepcopy(raw)
    without_model_cap["strategy"]["entry"]["condition"]["conditions"] = [
        raw["strategy"]["entry"]["condition"]["conditions"][0]
    ]
    rebuilt = _repair_explicit_one_day_return_encoding(without_model_cap, "C 点当日相对前一交易日收盘价小幅上涨，涨幅大于 0 且不超过 3%。")
    assert len(rebuilt["strategy"]["entry"]["condition"]["conditions"]) == 2
    assert rebuilt["strategy"]["entry"]["condition"]["conditions"][1]["right"]["value"] == 0.03


def test_explicit_cross_source_repair_folds_only_the_exact_current_prior_pair():
    sma10 = i("sma", 10)
    sma20 = i("sma", 20)
    raw = {"status": "parsed", "strategy": {"schema_version": "0.3", "anchor": {"condition": group("and",
        cmp(sma10, sma20),
        cmp({"kind": "lagged_indicator", "offset_days": -1, "indicator": sma10["indicator"]}, {"kind": "lagged_indicator", "offset_days": -1, "indicator": sma20["indicator"]}, "less_than_or_equal"),
        cmp(f("close"), i("sma", 5)),
    )}}, "warnings": [], "coverage": [{"clause_id": "C01", "dsl_paths": ["anchor.condition.conditions[1]", "anchor.condition.conditions[2]"]}]}
    repaired = _repair_explicit_cross_encoding(raw, "十日均线在 C 点当日上穿二十日均线。")
    conditions = repaired["strategy"]["anchor"]["condition"]["conditions"]
    assert len(conditions) == 2
    assert conditions[0]["node_type"] == "cross"
    assert conditions[0]["operator"] == "cross_above"
    assert repaired["warnings"][0]["code"] == "source_preserving_cross_repair"
    assert repaired["coverage"][0]["dsl_paths"] == ["anchor.condition.conditions[0]", "anchor.condition.conditions[1]"]
    assert _repair_explicit_cross_encoding(raw, "均线高于另一根均线") == raw


def test_cross_repair_finds_the_matching_pair_even_when_another_ma_comparison_follows():
    sma5, sma10, sma20 = i("sma", 5), i("sma", 10), i("sma", 20)
    raw = {"status": "parsed", "strategy": {"schema_version": "0.3", "anchor": {"condition": group("and",
        cmp(sma5, sma10),
        cmp({"kind": "lagged_indicator", "offset_days": -1, "indicator": sma5["indicator"]}, {"kind": "lagged_indicator", "offset_days": -1, "indicator": sma20["indicator"]}, "less_than_or_equal"),
        cmp(sma5, sma20),
    )}}, "warnings": [], "coverage": []}
    repaired = _repair_explicit_cross_encoding(raw, "5 日均线上穿 20 日均线。")
    conditions = repaired["strategy"]["anchor"]["condition"]["conditions"]
    assert [condition.get("operator") for condition in conditions] == ["greater_than", "cross_above"]


def test_coverage_rejects_silent_omission_and_preflight_questions_are_structured():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    response["coverage"] = []
    with pytest.raises(ModelResponseParsingFailure, match="source-coverage"):
        StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="buy then sell"))

    text = "基准点：t0 需满足“t0-15 至t0低点不在t0、MA20(t0)较t0-2至少高0.01%”。入场点：在 t1，需满足“t1收盘不低于t0”后按收盘价入场。"
    result = StrategyInterpreterService(FakeModelClient([])).interpret(StrategyInterpretationRequest(strategy_text=text))
    assert isinstance(result, ClarificationRequired)
    assert {question.question_id for question in result.questions} == {"anchor_low_reference_field", "anchor_price_reference", "anchor_indicator_change_basis"}
    assert len(result.coverage) == len(result.source_clauses)
    assert [clause.clause_id for clause in result.source_clauses] == ["C01", "C02", "C03", "C04"]


def test_numbered_specification_is_covered_by_semantic_sections_not_every_line():
    strategy_text = """1. 数据口径
价格使用收盘价。
定义：
Close[t] 为收盘价。
2.1 买入条件
Close[t] > MA5[t]
MA5[t] > MA10[t]
2.2 卖出条件
Close[t] < MA10[t]"""
    clauses = segment_source_clauses(strategy_text)
    assert [clause.clause_id for clause in clauses] == ["C01", "C02", "C03", "C04"]
    assert "Close[t] > MA5[t]" in clauses[2].text
    assert "Close[t] < MA10[t]" in clauses[3].text


def test_stateful_v03_features_are_sent_to_the_interpreter_for_structured_resolution():
    text = """在 50 日窗口内找到最高收盘价对应的高点 A；在 A 之后找到最低收盘价对应的低点 B；A 至 B 回撤至少 25%。
当日成交量为 C 点以来的最大成交量：Volume[t] = max(Volume[C], ..., Volume[t])。
当日 K 线是十字星或大阴线。"""
    clauses = segment_source_clauses(text)
    response = {"status": "clarification_required", "partial_strategy": None,
                "questions": [{"question_id": "sample_end", "question": "How should an open position at sample end be handled?", "target_path": "research_policy.sample_end", "suggested_answers": ["Force close", "Leave open"], "answer_kind": "choice"}],
                "ambiguous_terms": ["sample-end position handling"],
                "coverage": [{"clause_id": clause.clause_id, "disposition": "clarification_required", "dsl_paths": [], "explanation": "Will be represented in DSL v0.3 after policy clarification."} for clause in clauses]}
    client = FakeModelClient([response])
    result = StrategyInterpreterService(client).interpret(StrategyInterpretationRequest(strategy_text=text))
    assert isinstance(result, ClarificationRequired)
    assert len(client.requests) == 1


def test_lifecycle_policy_is_clarified_locally_before_the_provider_is_called():
    client = FakeModelClient([])
    text = "如用于完整回测，还应另行定义样本结束时未平仓头寸的处理方式，以及卖出后是否允许重新寻找下一次 C 点。"
    result = StrategyInterpreterService(client).interpret(StrategyInterpretationRequest(strategy_text=text))
    assert isinstance(result, ClarificationRequired)
    assert {question.question_id for question in result.questions} == {"sample_end_open_position", "allow_reentry_after_exit"}
    assert not client.requests


def test_no_maximum_holding_period_requires_sample_end_policy_without_a_model_call():
    client = FakeModelClient([])
    result = StrategyInterpreterService(client).interpret(
        StrategyInterpretationRequest(strategy_text="不设置最长持仓时间；持仓持续至卖出条件触发。")
    )
    assert isinstance(result, ClarificationRequired)
    assert [question.question_id for question in result.questions] == ["sample_end_open_position"]
    assert "样本结束时仍未平仓" in result.questions[0].question
    assert not client.requests


class InventoryClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def inspect_strategy(self, request):
        self.requests.append(request)
        return self.response


def _inventory(status, *, disposition="resolved", questions=None, unsupported_features=None):
    return {
        "status": status,
        "symbols": [],
        "items": [{
            "semantic_id": "I-ENTRY", "source_clause_ids": ["C01"], "category": "entry_rule",
            "pseudo_dsl": "enter when condition", "dependencies": [], "required_capabilities": [],
            "disposition": disposition,
        }],
        "closure": [{
            "dimension": "capability",
            "status": "closed" if status == "compile_eligible" else status,
            "explanation": "test finding", "source_clause_ids": ["C01"], "semantic_ids": ["I-ENTRY"],
        }],
        "questions": questions or [],
        "ambiguous_terms": ["ambiguous"] if questions else [],
        "unsupported_features": unsupported_features or [],
    }


def test_semantic_inventory_blocks_unsupported_complex_capability_before_full_dsl_request():
    inventory = _inventory(
        "unsupported",
        disposition="unsupported",
        unsupported_features=[{
            "capability_id": "persistent_regime_state", "source_clause_ids": ["C01"],
            "source_concept": "high state persists until exit", "required_extension": "persistent state transition",
        }],
    )
    final_client = FakeModelClient([])
    inventory_client = InventoryClient(inventory)

    from alpha_agent.errors import UnsupportedStrategyFeature
    with pytest.raises(UnsupportedStrategyFeature, match="cannot express faithfully"):
        StrategyInterpreterService(final_client, inventory_client=inventory_client).interpret(
            StrategyInterpretationRequest(strategy_text="high state persists")
        )

    assert len(inventory_client.requests) == 1
    assert not final_client.requests


def test_semantic_inventory_returns_user_clarification_before_full_dsl_request():
    question = {"question_id": "execution_time", "question": "Which execution price?", "target_path": "execution", "suggested_answers": ["close"], "answer_kind": "choice"}
    final_client = FakeModelClient([])
    result = StrategyInterpreterService(
        final_client,
        inventory_client=InventoryClient(_inventory("clarification_required", disposition="clarification_required", questions=[question])),
    ).interpret(StrategyInterpretationRequest(strategy_text="buy later"))

    assert isinstance(result, ClarificationRequired)
    assert result.questions[0].question_id == "execution_time"
    assert not final_client.requests


def test_compile_eligible_inventory_allows_existing_final_dsl_pipeline():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    final_client = FakeModelClient([response])
    result = StrategyInterpreterService(
        final_client,
        inventory_client=InventoryClient(_inventory("compile_eligible")),
    ).interpret(StrategyInterpretationRequest(strategy_text="buy then sell"))

    assert isinstance(result, ParsedStrategy)
    assert len(final_client.requests) == 1


def test_inventory_coverage_accepts_a_clause_grounded_in_a_closure_finding():
    from alpha_agent.domain.semantic_inventory import SemanticInventory
    from alpha_agent.parser.inventory import validate_inventory

    inventory = SemanticInventory.model_validate({
        "status": "compile_eligible", "symbols": [],
        "items": [{"semantic_id": "I-ENTRY", "source_clause_ids": ["C01"], "category": "entry_rule", "pseudo_dsl": "enter", "dependencies": [], "required_capabilities": [], "disposition": "resolved"}],
        "closure": [
            {"dimension": "temporal", "status": "closed", "explanation": "entry timing", "source_clause_ids": ["C01"], "semantic_ids": ["I-ENTRY"]},
            {"dimension": "execution", "status": "closed", "explanation": "daily close policy", "source_clause_ids": ["C02"], "semantic_ids": []},
        ],
        "questions": [], "ambiguous_terms": [], "unsupported_features": [],
    })
    request = StrategyInterpretationRequest(strategy_text="test", source_clauses=[
        {"clause_id": "C01", "text": "entry"}, {"clause_id": "C02", "text": "execution"},
    ])

    validate_inventory(inventory, request)


def test_inventory_coverage_failure_retries_only_the_inventory_before_final_dsl():
    class QueuedInventoryClient:
        def __init__(self, responses):
            self.responses, self.requests = list(responses), []

        def inspect_strategy(self, request):
            self.requests.append(request)
            return self.responses.pop(0)

    invalid = _inventory("compile_eligible")
    invalid["items"][0]["source_clause_ids"] = ["C99"]
    corrected = _inventory("compile_eligible")
    inventory_client = QueuedInventoryClient([invalid, corrected])
    final_client = FakeModelClient([parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))])

    result = StrategyInterpreterService(final_client, inventory_client=inventory_client).interpret(
        StrategyInterpretationRequest(strategy_text="buy then sell")
    )

    assert isinstance(result, ParsedStrategy)
    assert len(inventory_client.requests) == 2
    assert inventory_client.requests[1].repair_instruction is not None
    assert len(final_client.requests) == 1


def test_v03_anchor_indicator_from_provider_is_losslessly_normalized_to_a_lagged_indicator():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    response["strategy"] = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": cmp(i("sma", 20), {"kind": "anchor_indicator", "anchor": "t0", "offset_days": -1, "indicator": {"indicator": "sma", "field": "close", "window": 20}}), "constraints": []},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": cmp(f("close"), i("sma", 20)), "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": cmp(f("close"), i("sma", 20), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    response["coverage"] = [{"clause_id": "C01", "disposition": "mapped", "dsl_paths": ["anchor.condition", "entry.condition", "exit_rules[0].condition"], "explanation": "All test clauses map to the strategy."}]
    result = StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test", symbol="AAPL"))
    assert isinstance(result, ParsedStrategy)
    right = result.strategy.anchor.condition.right
    assert right.kind == "lagged_indicator"
    assert right.offset_days == -1


def test_provider_coverage_paths_are_canonicalized_only_at_the_outer_envelope_boundary():
    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    response["coverage"][0]["dsl_paths"] = ["strategy.entry_condition", "strategy.exit_condition"]
    result = StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test", symbol="AAPL"))
    assert isinstance(result, ParsedStrategy)
    assert result.coverage[0].dsl_paths == ["entry_condition", "exit_condition"]

    response = parsed(cmp(f("close"), i("sma", 20)), cmp(f("close"), i("sma", 20), "less_than"))
    response["coverage"][0]["dsl_paths"] = ["strategy.no_such_field"]
    with pytest.raises(ModelResponseParsingFailure, match="source-coverage"):
        StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test", symbol="AAPL"))


def test_explicit_prior_window_breakout_repairs_an_impossible_inclusive_rolling_maximum():
    raw = {
        "status": "parsed",
        "strategy": {
            "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
            "anchor": {"name": "t0", "condition": cmp(f("close"), {"kind": "indicator", "indicator": {"indicator": "rolling_max", "field": "close", "window": 20}}), "constraints": []},
            "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
            "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": cmp(f("close"), i("sma", 5), "less_than"), "execution": "close"}],
            "lifecycle_policy": {"sample_end_open_position": "leave_open_excluded", "allow_reentry_after_exit": True},
        },
        "assumptions": [], "warnings": [],
    }

    repaired = _repair_explicit_prior_window_extremum(raw, "当日收盘价严格高于此前 20 个交易日的最高收盘价。")

    right = repaired["strategy"]["anchor"]["condition"]["right"]
    assert right == {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "rolling_max", "field": "close", "window": 20}}
    assert repaired["warnings"][-1]["code"] == "source_preserving_prior_window_extremum_repair"
