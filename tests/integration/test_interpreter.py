from copy import deepcopy

import pytest
from alpha_agent.domain.interpretation import ClarificationRequired, ParsedStrategy, StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure, ModelResponseParsingFailure, SemanticStrategyValidationFailure
from alpha_agent.model.fake_client import FakeModelClient
from alpha_agent.monitoring import InMemoryInterpreterMonitor, InterpretationEvaluation
from alpha_agent.parser.service import StrategyInterpreterService, _repair_explicit_cross_encoding, _repair_explicit_one_day_return_encoding
from alpha_agent.parser.completeness import segment_source_clauses

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


def test_empty_provider_transport_fields_are_normalized_but_populated_ones_are_rejected():
    response = parsed(cmp(i("rsi",14),s(30),"less_than"),cmp(i("rsi",14),s(70)))
    response.update({"partial_strategy":None,"questions":[],"ambiguous_terms":[]})
    assert isinstance(StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test")), ParsedStrategy)
    response["questions"] = [{"question_id":"unexpected","question":"Unexpected","target_path":"entry_condition","suggested_answers":[],"answer_kind":"choice"}]
    with pytest.raises(ModelResponseParsingFailure): StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test"))


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
