import pytest

from alpha_agent.domain.interpretation import ClauseCoverage, SourceClause
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.errors import SemanticStrategyValidationFailure
from alpha_agent.parser.conformance import validate_source_conformance
from alpha_agent.parser.validation import validate_strategy


def test_source_conformance_rejects_comparison_expansion_of_cross_and_price_level_for_daily_return():
    source = [
        SourceClause(clause_id="C01", text="十日均线在 C 点当日上穿二十日均线。"),
        SourceClause(clause_id="C02", text="C 点当日相对前一交易日收盘价小幅上涨，涨幅不超过 3%。"),
    ]
    coverage = [
        ClauseCoverage(clause_id="C01", disposition="mapped", dsl_paths=["anchor.condition"], explanation="Mapped."),
        ClauseCoverage(clause_id="C02", disposition="mapped", dsl_paths=["entry.condition"], explanation="Mapped."),
    ]
    payload = {
        "anchor": {"condition": {"node_type": "group", "operator": "and", "conditions": [
            {"node_type": "comparison", "operator": "greater_than"},
            {"node_type": "comparison", "operator": "less_than_or_equal"},
        ]}},
        "entry": {"condition": {"node_type": "comparison", "operator": "less_than_or_equal", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "scalar", "value": 0}}},
    }
    with pytest.raises(SemanticStrategyValidationFailure) as error:
        validate_source_conformance(source, coverage, payload)
    assert {issue["rule"] for issue in error.value.issues} == {
        "cross_requires_cross_node", "daily_return_requires_rolling_return",
    }


def test_source_conformance_accepts_explicit_cross_and_one_day_rolling_return():
    source = [
        SourceClause(clause_id="C01", text="十日均线在 C 点当日上穿二十日均线。"),
        SourceClause(clause_id="C02", text="C 点当日相对前一交易日收盘价小幅上涨，涨幅不超过 3%。"),
    ]
    coverage = [
        ClauseCoverage(clause_id="C01", disposition="mapped", dsl_paths=["anchor.condition"], explanation="Mapped."),
        ClauseCoverage(clause_id="C02", disposition="mapped", dsl_paths=["entry.condition"], explanation="Mapped."),
    ]
    payload = {
        "anchor": {"condition": {"node_type": "cross", "operator": "cross_above"}},
        "entry": {"condition": {"node_type": "comparison", "operator": "less_than_or_equal", "left": {"kind": "anchor_indicator", "indicator": {"indicator": "rolling_return", "field": "close", "window": 1}}, "right": {"kind": "scalar", "value": .03}}},
    }
    validate_source_conformance(source, coverage, payload)


def test_source_conformance_accepts_touch_inclusive_macd_dead_cross_as_two_day_comparison_pair():
    source = [SourceClause(
        clause_id="C01",
        text="DIF 下穿 DEA，形成死叉：前一日：DIF > DEA；当日：DIF ≤ DEA。",
    )]
    coverage = [ClauseCoverage(
        clause_id="C01", disposition="mapped", dsl_paths=["exit_rules[0].condition"], explanation="DIF/DEA 的等号包含死叉已映射。",
    )]
    line = {"indicator": "macd_line", "field": "close", "fast_window": 10, "slow_window": 24, "signal_window": 8}
    signal = {"indicator": "macd_signal", "field": "close", "fast_window": 10, "slow_window": 24, "signal_window": 8}
    payload = {"exit_rules": [{"condition": {"node_type": "group", "operator": "and", "conditions": [
        {"node_type": "comparison", "operator": "less_than_or_equal", "left": {"kind": "indicator", "indicator": line}, "right": {"kind": "indicator", "indicator": signal}},
        {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "lagged_indicator", "offset_days": -1, "indicator": line}, "right": {"kind": "lagged_indicator", "offset_days": -1, "indicator": signal}},
    ]}}]}

    validate_source_conformance(source, coverage, payload)


def test_source_conformance_accepts_dif_zero_axis_cross_as_current_prior_scalar_pair():
    source = [SourceClause(clause_id="C01", text="DIF 从零轴下方上穿零轴。")]
    coverage = [ClauseCoverage(clause_id="C01", disposition="mapped", dsl_paths=["anchor.condition"], explanation="DIF 零轴上穿以相邻两日比较表示。")]
    line = {"indicator": "macd_line", "field": "close", "fast_window": 10, "slow_window": 24, "signal_window": 8}
    payload = {"anchor": {"condition": {"node_type": "group", "operator": "and", "conditions": [
        {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "indicator", "indicator": line}, "right": {"kind": "scalar", "value": 0}},
        {"node_type": "comparison", "operator": "less_than_or_equal", "left": {"kind": "lagged_indicator", "offset_days": -1, "indicator": line}, "right": {"kind": "scalar", "value": 0}},
    ]}}}

    validate_source_conformance(source, coverage, payload)


def test_source_conformance_accepts_price_downcross_written_as_ma_crosses_above_close():
    source = [SourceClause(clause_id="C01", text="Cross(20ma,Close);{股价下穿20Ma}")]
    coverage = [ClauseCoverage(
        clause_id="C01", disposition="mapped", dsl_paths=["exit_rules[0].condition"],
        explanation="公式的左操作数是 MA20；其上穿 Close 等价于股价下穿 MA20。",
    )]
    payload = {
        "exit_rules": [{"condition": {
            "node_type": "cross", "operator": "cross_above",
            "left": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}},
            "right": {"kind": "market_field", "field": "close"},
        }}],
    }
    validate_source_conformance(source, coverage, payload)


def test_timed_semantic_validation_rejects_strict_current_vs_anchor_price_on_t0():
    payload = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 5}}}, "constraints": []},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "anchor_market_field", "anchor": "t0", "field": "close"}}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": {"node_type": "comparison", "operator": "less_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}}}, "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    with pytest.raises(SemanticStrategyValidationFailure) as error:
        validate_strategy(TimedStrategyDefinition.model_validate(payload))
    assert error.value.issues[0]["rule"] == "anchor_day_identity_comparison"
