import json
import pytest
from pydantic import TypeAdapter, ValidationError
from alpha_agent.domain.interpretation import StrategyInterpretationResult
from alpha_agent.domain.strategy import StrategyDefinition
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.errors import SemanticStrategyValidationFailure
from alpha_agent.parser.validation import validate_strategy
from alpha_agent.parser.review import build_review_items

def field(name): return {"kind": "market_field", "field": name}
def scalar(value): return {"kind": "scalar", "value": value}
def indicator(kind, window, field_name="close"): return {"kind": "indicator", "indicator": {"indicator": kind, "field": "volume" if kind == "rolling_mean" else field_name, "window": window}}
def comparison(left, right, op="greater_than"): return {"node_type": "comparison", "operator": op, "left": left, "right": right}
def strategy(entry=None, exit=None):
    base = comparison(field("close"), indicator("sma", 20))
    return {"schema_version":"0.1", "symbol":"AAPL", "frequency":"1d", "direction":"long_only", "position_mode":"fully_invested_or_flat", "entry_condition":entry or base, "exit_condition":exit or base}

@pytest.mark.parametrize("value", [0, -1])
def test_invalid_indicator_window_is_rejected(value):
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), indicator("sma", value))))
def test_unsupported_indicator_and_cross_scalar_are_rejected():
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), indicator("macd", 12))))
    cross = {"node_type":"cross", "operator":"cross_above", "left":field("close"), "right":scalar(1)}
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(cross))
@pytest.mark.parametrize("threshold", [-1, 101])
def test_rsi_threshold_ranges_are_semantic_errors(threshold):
    model = StrategyDefinition.model_validate(strategy(comparison(indicator("rsi", 14), scalar(threshold))))
    with pytest.raises(SemanticStrategyValidationFailure, match="semantic"): validate_strategy(model)
def test_empty_groups_and_invalid_volume_indicator_are_rejected_but_singletons_are_valid():
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy({"node_type":"group","operator":"and","conditions":[]}))
    one = StrategyDefinition.model_validate(strategy({"node_type":"group","operator":"and","conditions":[comparison(field("close"), indicator("sma",20))]}))
    validate_strategy(one)
    bad = {"kind":"indicator","indicator":{"indicator":"rolling_mean","field":"close","window":20}}
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), bad)))
def test_nested_and_or_and_comparison_operators_validate():
    leaf = comparison(field("close"), indicator("sma", 20), "greater_than_or_equal")
    tree = {"node_type":"group","operator":"and","conditions":[leaf,{"node_type":"group","operator":"or","conditions":[leaf,comparison(field("close"), indicator("ema", 10), "less_than_or_equal")]}]}
    validate_strategy(StrategyDefinition.model_validate(strategy(tree)))
def test_required_fixed_fields_and_schema_versions_are_rejected():
    for key, value in (("frequency","daily"),("direction","short"),("schema_version","0.2")):
        payload = strategy(); payload[key] = value
        with pytest.raises(ValidationError): StrategyDefinition.model_validate(payload)
    payload = strategy(); del payload["schema_version"]
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(payload)
def test_json_round_trips_preserve_both_result_variants():
    adapter = TypeAdapter(StrategyInterpretationResult)
    parsed = {"status":"parsed","strategy":strategy(),"assumptions":[],"warnings":[]}
    clarification = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"window","question":"Which window?","target_path":"entry_condition","suggested_answers":["20"],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
    for payload in (parsed, clarification):
        restored = adapter.validate_json(adapter.dump_json(adapter.validate_json(json.dumps(payload))))
        assert restored.status == payload["status"]
    assert adapter.validate_python(parsed).strategy.schema_version == "0.1"


def test_timed_strategy_v02_represents_generic_relative_days_and_intraday_requirements():
    timed = {
        "schema_version": "0.2", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "intraday_ohlcv",
        "anchor": {
            "name": "t0",
            "condition": {"node_type": "group", "operator": "and", "conditions": [
                comparison(indicator("rolling_return", 1), scalar(.05), "greater_than_or_equal"),
                comparison(field("close"), indicator("sma", 5)),
            ]},
            "constraints": [
                {"kind": "rolling_low_anchor_constraint", "lookback_days": 15, "low_must_precede_anchor": True, "maximum_anchor_close_gain": .2},
                {"kind": "anchor_indicator_change_constraint", "indicator": {"indicator": "sma", "field": "close", "window": 20}, "comparison_offset_days": -2, "minimum_relative_change": .0001},
            ],
        },
        "entry": {"active_day": {"anchor": "t0", "start_offset_days": 1, "end_offset_days": 1}, "condition": {"node_type": "comparison", "operator": "greater_than_or_equal", "left": field("close"), "right": {"kind": "anchor_market_field", "anchor": "t0", "field": "close"}}, "execution": "close"},
        "exit_rules": [
            {"rule_id": "early_stop", "priority": 10, "active_days": {"start_offset_days": 3, "end_offset_days": 4}, "kind": "intraday_price_trigger", "price_trigger": {"base": "entry_price", "multiplier": .93, "operator": "less_than_or_equal"}, "execution": "intraday"},
            {"rule_id": "technical", "priority": 20, "active_days": {"start_offset_days": 4, "end_offset_days": 6}, "kind": "close_condition", "condition": {"node_type": "comparison", "operator": "less_than", "left": field("close"), "right": {"kind": "entry_price"}}, "execution": "close"},
            {"rule_id": "t6_protection", "priority": 5, "active_days": {"start_offset_days": 6, "end_offset_days": 6}, "kind": "intraday_price_trigger", "price_trigger": {"base": "entry_price", "multiplier": .99, "operator": "less_than_or_equal"}, "execution": "intraday"},
            {"rule_id": "t6_close", "priority": 30, "active_days": {"start_offset_days": 6, "end_offset_days": 6}, "kind": "forced_close", "execution": "close"},
        ],
    }
    model = TimedStrategyDefinition.model_validate(timed)
    assert model.entry.active_day.start_offset_days == 1
    assert TimedStrategyDefinition.model_validate_json(model.model_dump_json()) == model
    parsed_result = TypeAdapter(StrategyInterpretationResult).validate_python({"status": "parsed", "strategy": timed, "assumptions": [], "warnings": []})
    assert parsed_result.strategy.schema_version == "0.2"
    timed["data_requirement"] = "daily_ohlcv"
    with pytest.raises(ValidationError, match="intraday_ohlcv"):
        TimedStrategyDefinition.model_validate(timed)


def test_open_ended_technical_exit_does_not_require_or_create_a_forced_close():
    strategy_without_expiry = {
        "schema_version": "0.2", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": comparison(field("close"), indicator("sma", 20))},
        "entry": {"active_day": {"start_offset_days": 1, "end_offset_days": 1}, "condition": {"node_type": "comparison", "operator": "greater_than_or_equal", "left": field("close"), "right": {"kind": "anchor_market_field", "anchor": "t0", "field": "close"}}, "execution": "close"},
        "exit_rules": [{"rule_id": "technical_from_t4", "priority": 1, "active_days": {"start_offset_days": 4}, "kind": "close_condition", "condition": {"node_type": "comparison", "operator": "less_than", "left": field("close"), "right": {"kind": "entry_price"}}, "execution": "close"}],
    }
    model = TimedStrategyDefinition.model_validate(strategy_without_expiry)
    assert model.exit_rules[0].active_days.end_offset_days is None
    assert all(rule.kind != "forced_close" for rule in model.exit_rules)


def test_timed_dsl_can_preserve_market_index_dif_requirement_without_claiming_single_asset_data():
    timed = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv_with_market_index",
        "anchor": {"name": "t0", "condition": {"node_type": "group", "operator": "and", "conditions": [
            comparison(
                {"kind": "indicator", "indicator": {"indicator": "market_index_macd_line", "index_symbol": "^IXIC", "fast_window": 10, "slow_window": 24, "signal_window": 8}},
                scalar(0),
            ),
            comparison(
                {"kind": "indicator", "indicator": {"indicator": "dmi_adx", "directional_window": 10, "adx_window": 6}},
                scalar(26),
            ),
        ]}, "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 1, "end_offset_days": 1}, "execution": "open"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 5), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "leave_open_excluded", "allow_reentry_after_exit": True},
    }

    parsed = TimedStrategyDefinition.model_validate(timed)
    validate_strategy(parsed)
    assert parsed.entry.execution == "open"
    assert parsed.data_requirement == "daily_ohlcv_with_market_index"


def test_same_exit_priority_is_allowed_only_for_non_overlapping_relative_day_ranges():
    timed = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": comparison(field("close"), indicator("sma", 5)), "constraints": []},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": comparison(field("close"), indicator("sma", 5)), "execution": "close"},
        "exit_rules": [
            {"rule_id": "early", "priority": 1, "active_days": {"start_offset_days": 1, "end_offset_days": 10}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 20), "less_than"), "execution": "close"},
            {"rule_id": "late", "priority": 1, "active_days": {"start_offset_days": 11}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 10), "less_than"), "execution": "close"},
        ],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": True},
    }
    assert TimedStrategyDefinition.model_validate(timed).exit_rules[1].priority == 1
    timed["exit_rules"][1]["active_days"]["start_offset_days"] = 10
    with pytest.raises(ValidationError, match="simultaneously active"):
        TimedStrategyDefinition.model_validate(timed)


def test_v03_supports_ordered_drawdown_running_volume_and_candlestick_exit():
    timed = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": {"node_type": "group", "operator": "and", "conditions": [
            comparison(field("close"), indicator("sma", 5)),
            comparison(indicator("sma", 5), indicator("sma", 10)),
            comparison(indicator("sma", 10), indicator("sma", 20)),
            comparison(indicator("sma", 20), {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "sma", "field": "close", "window": 20}}),
        ]}, "constraints": [{"kind": "ordered_extrema_drawdown_constraint", "lookback_days": 50, "field": "close", "peak_tie_break": "earliest", "trough_tie_break": "earliest", "minimum_peak_to_trough_drawdown": .25, "maximum_anchor_recovery_from_trough": .15}]},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": comparison(field("close"), indicator("sma", 5)), "execution": "close"},
        "exit_rules": [
            {"rule_id": "protect_ma20", "priority": 1, "active_days": {"start_offset_days": 1, "end_offset_days": 10}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 20), "less_than"), "execution": "close"},
            {"rule_id": "manage", "priority": 2, "active_days": {"start_offset_days": 11}, "kind": "close_condition", "condition": {"node_type": "group", "operator": "or", "conditions": [
                comparison(field("close"), indicator("sma", 10), "less_than"),
                {"node_type": "group", "operator": "and", "conditions": [
                    comparison(field("close"), {"kind": "scaled_entry_price", "multiplier": 1.1}, "greater_than"),
                    comparison(field("volume"), {"kind": "anchor_running_maximum", "anchor": "t0", "field": "volume", "tie_break": "earliest"}, "equal"),
                    {"node_type": "group", "operator": "or", "conditions": [
                        {"node_type": "candlestick_pattern", "pattern": "doji", "body_to_open_threshold": .01},
                        {"node_type": "candlestick_pattern", "pattern": "large_bearish", "body_to_open_threshold": .02},
                    ]},
                ]},
            ]}, "execution": "close"},
        ],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    model = TimedStrategyDefinition.model_validate(timed)
    assert model.anchor.constraints[0].peak_tie_break == "earliest"
    assert model.exit_rules[1].condition.node_type == "group"
    assert "成交量 = 自 t0 起至当日的最大成交量" in "\n".join(item.explanation for item in build_review_items(model))
    without_policy = dict(timed); without_policy.pop("lifecycle_policy")
    with pytest.raises(ValidationError, match="lifecycle_policy"):
        TimedStrategyDefinition.model_validate(without_policy)


def test_fixed_t0_entry_can_rely_on_the_anchor_without_a_duplicate_condition():
    timed = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": comparison(field("close"), indicator("sma", 5)), "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 5), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": True},
    }
    parsed = TimedStrategyDefinition.model_validate(timed)
    assert parsed.entry.condition is None


def test_timed_anchor_rejects_price_comparison_against_a_return_indicator():
    timed = {
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only",
        "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": comparison(field("close"), {"kind": "scaled_operand", "operand": indicator("rolling_return", 15), "multiplier": 1.15}), "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": comparison(field("close"), indicator("sma", 5), "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "leave_open_excluded", "allow_reentry_after_exit": True},
    }
    with pytest.raises(SemanticStrategyValidationFailure) as error:
        validate_strategy(TimedStrategyDefinition.model_validate(timed))
    assert error.value.details["issues"][0]["rule"] == "compatible_operands"


@pytest.mark.parametrize(
    ("left", "right", "operator"),
    [
        (field("close"), indicator("rolling_max", 20), "greater_than"),
        (field("close"), indicator("rolling_min", 20), "less_than"),
    ],
)
def test_strict_current_rolling_extremum_comparison_is_rejected_as_impossible(left, right, operator):
    model = StrategyDefinition.model_validate(strategy(comparison(left, right, operator)))

    with pytest.raises(SemanticStrategyValidationFailure) as error:
        validate_strategy(model)

    assert error.value.details["issues"][0]["rule"] == "inclusive_rolling_extremum_strict_comparison"


def test_review_items_are_deterministically_derived_from_validated_dsl():
    model = StrategyDefinition.model_validate(strategy())
    items = build_review_items(model)
    assert [(item.title, item.dsl_path) for item in items] == [("执行频率", "frequency"), ("入场条件", "entry_condition"), ("出场条件", "exit_condition")]
    assert "SMA(20)" in items[1].explanation
