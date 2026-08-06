import numpy as np
import pandas as pd

from alpha_agent.domain.conditions import CrossCondition
from alpha_agent.domain.indicators import ADX, MACDLine, MACDSignal, RollingMaximum, RollingMinimum, SMA
from alpha_agent.domain.operands import IndicatorOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorOperand,
    AnchorRunningVolumeRankOperand,
    TemporalComparisonCondition,
    TemporalCrossCondition,
    TemporalScalarOperand,
    TimedStrategyDefinition,
)
from alpha_agent.research.v03_engine import build_trade_event_audit, _anchor_condition_checks, _anchor_constraints_hold, _indicator_values, _resolve_entry, _static_condition, _temporal_condition, _temporal_operand, run_v03_backtest
from alpha_agent.parser.validation import validate_strategy


def _field(name): return {"kind": "market_field", "field": name}
def _sma(window): return {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": window}}
def _cmp(left, right, operator="greater_than"): return {"node_type": "comparison", "operator": operator, "left": left, "right": right}


def _strategy() -> TimedStrategyDefinition:
    return TimedStrategyDefinition.model_validate({
        "schema_version": "0.3", "symbol": "TEST", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": {"node_type": "group", "operator": "and", "conditions": [_cmp(_field("close"), _sma(5)), _cmp(_sma(5), _sma(10)), _cmp(_sma(10), _sma(20))]}, "constraints": [
            {"kind": "ordered_extrema_drawdown_constraint", "lookback_days": 50, "field": "close", "peak_tie_break": "earliest", "trough_tie_break": "earliest", "minimum_peak_to_trough_drawdown": .25, "maximum_anchor_recovery_from_trough": .15},
            {"kind": "anchor_indicator_change_constraint", "indicator": {"indicator": "sma", "field": "close", "window": 20}, "comparison_offset_days": -1, "minimum_relative_change": 0, "operator": "greater_than"},
        ]},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": _cmp(_field("close"), _sma(5)), "execution": "close"},
        "exit_rules": [
            {"rule_id": "protect_ma20", "priority": 1, "active_days": {"start_offset_days": 1, "end_offset_days": 10}, "kind": "close_condition", "condition": _cmp(_field("close"), _sma(20), "less_than"), "execution": "close"},
            {"rule_id": "high_volume_reversal", "priority": 2, "active_days": {"start_offset_days": 11}, "kind": "close_condition", "condition": {"node_type": "group", "operator": "and", "conditions": [
                _cmp(_field("close"), {"kind": "scaled_entry_price", "multiplier": 1.1}),
                _cmp(_field("volume"), {"kind": "anchor_running_maximum", "anchor": "t0", "field": "volume", "tie_break": "earliest"}, "equal"),
                {"node_type": "group", "operator": "or", "conditions": [
                    {"node_type": "candlestick_pattern", "pattern": "doji", "body_to_open_threshold": .01},
                    {"node_type": "candlestick_pattern", "pattern": "large_bearish", "body_to_open_threshold": .02},
                ]},
            ]}, "execution": "close"},
        ],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    })


def test_v03_engine_uses_earliest_ordered_extrema_and_high_volume_doji_exit():
    close = np.concatenate([np.linspace(100, 70, 20), np.linspace(70, 80, 31), np.linspace(81, 90, 11), [90.2, 90.5, 90.8]])
    open_ = close.copy(); open_[61] = close[61] * 1.004  # doji at the first management day
    volume = np.full(len(close), 100.0); volume[61] = 1_000.0
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": open_, "High": close + 1, "Low": close - 1, "Close": close, "Volume": volume})
    trades, stats = run_v03_backtest(frame, strategy=_strategy(), capital=10_000, compound=False)
    assert len(trades) == 1
    assert trades.iloc[0].reason == "v03:high_volume_reversal"
    assert trades.iloc[0].days_held == 11
    assert trades.iloc[0].event_points[0]["label"].startswith("A：")
    assert trades.iloc[0].event_points[1]["label"].startswith("B：")
    assert any(point["label"] == "C：t0 基准点" for point in trades.iloc[0].event_points)
    assert all(check["passed"] for check in trades.iloc[0].anchor_checks)
    assert {check["left"] for check in trades.iloc[0].anchor_checks if not str(check["left"]).startswith("组合条件")} == {"Close", "SMA(5)", "SMA(10)"}
    assert any(check["left"] == "组合条件（AND）" and check["passed"] for check in trades.iloc[0].anchor_checks)
    assert stats["n_trades"] == 1


def test_v03_event_audit_replays_entry_and_clicked_exit_with_actual_values():
    close = np.concatenate([np.linspace(100, 70, 20), np.linspace(70, 80, 31), np.linspace(81, 90, 11), [90.2, 90.5, 90.8]])
    open_ = close.copy(); open_[61] = close[61] * 1.004
    volume = np.full(len(close), 100.0); volume[61] = 1_000.0
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": open_, "High": close + 1, "Low": close - 1, "Close": close, "Volume": volume})
    trades, _ = run_v03_backtest(frame, strategy=_strategy(), capital=10_000, compound=False)
    trade = trades.iloc[0]

    entry_audit = build_trade_event_audit(frame, strategy=_strategy(), signal_date=trade.signal, entry_date=trade.entry, exit_date=trade.exit, reason=trade.reason, event="entry")
    exit_audit = build_trade_event_audit(frame, strategy=_strategy(), signal_date=trade.signal, entry_date=trade.entry, exit_date=trade.exit, reason=trade.reason, event="exit")

    assert entry_audit["event"] == "entry"
    assert any(row["section"] == "C 点入场条件" and row["passed"] for row in entry_audit["rows"])
    assert any(row["section"] == "C 点结构约束" and row["passed"] for row in entry_audit["rows"])
    assert exit_audit["event"] == "exit"
    assert "high_volume_reversal" in exit_audit["summary"]
    # Exit audits are intentionally complete: users can see both the winning
    # rule and why every other rule was not eligible on the same day.
    assert any(row["section"].startswith("出场规则：high_volume_reversal") and row["passed"] for row in exit_audit["rows"])
    assert any(row["section"].startswith("出场规则：protect_ma20") for row in exit_audit["rows"])
    assert any(
        row["dsl_path"] == "exit_rules.high_volume_reversal.result" and row["passed"]
        for row in exit_audit["rows"]
    )
    assert any(
        row["dsl_path"] == "exit_rules.protect_ma20.result" and not row["passed"]
        for row in exit_audit["rows"]
    )


def test_v03_engine_enters_on_a_qualified_anchor_without_duplicate_entry_condition():
    payload = _strategy().model_dump(mode="json")
    payload["entry"] = {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"}
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = np.concatenate([np.linspace(100, 70, 20), np.linspace(70, 80, 31), np.linspace(81, 90, 15)])
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 100.0})
    trades, _ = run_v03_backtest(frame, strategy=strategy, capital=10_000, compound=False)
    assert len(trades) == 1
    assert trades.iloc[0].signal == trades.iloc[0].entry


def test_v03_engine_allows_false_leaf_inside_a_satisfied_anchor_or_group():
    """Audit snapshots retain false OR branches without rejecting the anchor."""
    payload = _strategy().model_dump(mode="json")
    payload["anchor"]["condition"] = {
        "node_type": "group",
        "operator": "or",
        "conditions": [
            _cmp(_field("close"), {"kind": "scalar", "value": 10_000}),
            _cmp(_field("close"), _sma(5)),
        ],
    }
    payload["anchor"]["constraints"] = []
    payload["entry"] = {
        "mode": "fixed",
        "active_day": {"start_offset_days": 0, "end_offset_days": 0},
        "execution": "close",
    }
    payload["lifecycle_policy"]["allow_reentry_after_exit"] = False
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = np.linspace(10, 30, 30)
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 100.0})

    trades, _ = run_v03_backtest(frame, strategy=strategy, capital=10_000, compound=False)

    assert len(trades) == 1
    assert any(not check["passed"] for check in trades.iloc[0].anchor_checks)
    assert any(check["passed"] for check in trades.iloc[0].anchor_checks)
    assert any(check["left"] == "组合条件（OR）" and check["passed"] for check in trades.iloc[0].anchor_checks)


def test_v03_supports_macd_rolling_trend_count_and_wait_until_entry():
    close = np.array([10, 9, 8, 8.5, 8.2, 8.1, 8.4, 8.3, 8.6, 8.5, 8.8, 8.7, 9.0, 8.9, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2], dtype=float)
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close + .1, "Low": close - .1, "Close": close, "Volume": np.arange(len(close), dtype=float)})
    macd_line = {"indicator": "macd_line", "field": "close", "fast_window": 3, "slow_window": 6, "signal_window": 3}
    macd_signal = {"indicator": "macd_signal", "field": "close", "fast_window": 3, "slow_window": 6, "signal_window": 3}
    assert np.isfinite(_indicator_values(frame, MACDLine.model_validate(macd_line))[-1])
    assert np.isfinite(_indicator_values(frame, MACDSignal.model_validate(macd_signal))[-1])
    assert _indicator_values(frame, RollingMinimum.model_validate({"indicator": "rolling_min", "field": "close", "window": 5}))[-1] == 11.4
    sma = SMA(indicator="sma", field="close", window=5)
    assert _indicator_values(frame, sma) is _indicator_values(frame, sma)
    assert sma in frame.attrs["_alpha_indicator_cache"]
    rolling_count = {"node_type": "rolling_comparison_count", "lookback_days": 5, "minimum_true_count": 3, "comparison": _cmp(_sma(5), {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "sma", "field": "close", "window": 5}})}
    from alpha_agent.domain.conditions import Condition
    from pydantic import TypeAdapter
    assert _static_condition(frame, TypeAdapter(Condition).validate_python(rolling_count), len(frame) - 1)

    payload = _strategy().model_dump(mode="json")
    payload["entry"] = {
        "mode": "wait_until",
        "execution": "close",
        "defer_when": _cmp(_field("close"), {"kind": "lagged_market_field", "offset_days": -1, "field": "close"}, "less_than"),
        "resume_when": _cmp(_field("close"), {"kind": "lagged_market_field", "offset_days": -1, "field": "close"}, "greater_than"),
    }
    strategy = TimedStrategyDefinition.model_validate(payload)
    # A wait-until entry has no branches; semantic validation must inspect its
    # defer/resume predicates instead of assuming every non-fixed entry is
    # conditional.
    validate_strategy(strategy)
    # Anchor at a declining day waits for the first subsequent up-close.
    resolved = _resolve_entry(frame, strategy, 4)
    assert resolved is not None and resolved[0] == 6


def test_v03_supports_adx_prior_window_breakout_and_two_day_ma_exit():
    """Regression coverage for the complete 波段交易策略 V2 primitives."""
    close = np.concatenate([np.linspace(10, 20, 70), np.linspace(20.1, 25, 20)])
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=len(close), freq="B"),
        "Open": close * .998, "High": close * 1.01, "Low": close * .99,
        "Close": close, "Volume": np.linspace(100, 400, len(close)),
    })
    adx = ADX(indicator="adx", window=14)
    # Populate the frame-local indicator cache first: ADX must remain valid
    # when other indicators have already been evaluated on the same frame.
    _indicator_values(frame, SMA(indicator="sma", field="close", window=5))
    assert np.isfinite(_indicator_values(frame, adx)[-1])
    assert _indicator_values(frame, RollingMaximum(indicator="rolling_max", field="close", window=20))[-1] == close[-1]

    payload = _strategy().model_dump(mode="json")
    payload["anchor"]["constraints"] = []
    payload["anchor"]["condition"] = {
        "node_type": "group", "operator": "and", "conditions": [
            _cmp(_field("close"), {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "rolling_max", "field": "close", "window": 20}}),
            _cmp({"kind": "indicator", "indicator": {"indicator": "adx", "window": 14}}, {"kind": "scalar", "value": 22}),
            _cmp({"kind": "indicator", "indicator": {"indicator": "adx", "window": 14}}, {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "adx", "window": 14}}, "greater_than_or_equal"),
        ],
    }
    payload["entry"] = {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"}
    payload["exit_rules"] = [{
        "rule_id": "two_days_below_ma3", "priority": 1, "relative_to": "entry",
        "active_days": {"start_offset_days": 1}, "kind": "close_condition", "execution": "close",
        "condition": {"node_type": "group", "operator": "and", "conditions": [
            _cmp(_field("close"), _sma(3), "less_than"),
            _cmp({"kind": "lagged_market_field", "offset_days": -1, "field": "close"}, {"kind": "lagged_indicator", "offset_days": -1, "indicator": {"indicator": "sma", "field": "close", "window": 3}}, "less_than"),
        ]},
    }]
    strategy = TimedStrategyDefinition.model_validate(payload)
    validate_strategy(strategy)
    assert _static_condition(frame, strategy.anchor.condition, len(frame) - 1)


def test_v03_anchor_cross_requires_current_cross_and_prior_non_crossing_state():
    close = np.array([10.0] * 20 + [12.0, 12.0])
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0})
    cross = CrossCondition(
        node_type="cross",
        operator="cross_above",
        left=IndicatorOperand(kind="indicator", indicator=SMA(indicator="sma", field="close", window=10)),
        right=IndicatorOperand(kind="indicator", indicator=SMA(indicator="sma", field="close", window=20)),
    )
    assert _static_condition(frame, cross, 20)
    assert not _static_condition(frame, cross, 21)
    snapshot = _anchor_condition_checks(frame, cross, 20)
    assert snapshot[0]["passed"] is True
    assert snapshot[0]["prior_left_value"] <= snapshot[0]["prior_right_value"]


def test_v03_temporal_anchor_indicator_resolves_against_t0_and_supports_crosses():
    close = np.array([10.0] * 20 + [12.0, 12.0])
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0})
    anchored_sma = AnchorIndicatorOperand(kind="anchor_indicator", anchor="t0", offset_days=0, indicator=SMA(indicator="sma", field="close", window=10))
    assert _temporal_operand(frame, anchored_sma, index=21, anchor_index=20, entry_price=12.0) == 10.2

    positive_anchor_return = TemporalComparisonCondition(
        node_type="comparison",
        operator="greater_than",
        left=AnchorIndicatorOperand(kind="anchor_indicator", anchor="t0", offset_days=0, indicator={"indicator": "rolling_return", "field": "close", "window": 1}),
        right=TemporalScalarOperand(kind="scalar", value=0),
    )
    assert _temporal_condition(frame, positive_anchor_return, index=20, anchor_index=20, entry_price=12.0)

    cross = TemporalCrossCondition(
        node_type="cross",
        operator="cross_above",
        left={"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 10}},
        right={"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}},
    )
    assert _temporal_condition(frame, cross, index=20, anchor_index=20, entry_price=12.0)


def test_v03_ordered_extrema_honors_latest_tie_break():
    payload = _strategy().model_dump(mode="json")
    payload["anchor"]["constraints"] = [{
        "kind": "ordered_extrema_drawdown_constraint", "lookback_days": 50,
        "field": "close", "peak_tie_break": "latest", "trough_tie_break": "latest",
        "minimum_peak_to_trough_drawdown": .25, "maximum_anchor_recovery_from_trough": .15,
    }]
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = np.array([100.0, 100.0, 70.0, 70.0] + [75.0] * 46)
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0})
    passed, points = _anchor_constraints_hold(frame, strategy, len(frame) - 1)
    assert passed
    assert points[0]["date"] == "2020-01-02"  # Latest tied 100 close.
    assert points[1]["date"] == "2020-01-06"  # Latest tied 70 close after A.


def test_v03_conditional_entry_uses_actual_entry_day_for_exit_window_and_volume_peak():
    payload = {
        "schema_version": "0.3", "symbol": "TEST", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": _cmp(_field("close"), _sma(2)), "constraints": []},
        "entry": {"mode": "conditional", "execution": "close", "branches": [
            {"branch_id": "direct", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": _cmp({"kind": "anchor_indicator", "anchor": "t0", "offset_days": 0, "indicator": {"indicator": "rolling_return", "field": "close", "window": 1}}, {"kind": "scalar", "value": .03}, "less_than_or_equal")},
            {"branch_id": "next_day_pullback", "active_day": {"start_offset_days": 1, "end_offset_days": 1}, "condition": {"node_type": "group", "operator": "and", "conditions": [
                _cmp({"kind": "anchor_indicator", "anchor": "t0", "offset_days": 0, "indicator": {"indicator": "rolling_return", "field": "close", "window": 1}}, {"kind": "scalar", "value": .03}, "greater_than"),
                _cmp(_field("close"), {"kind": "anchor_market_field", "anchor": "t0", "field": "close"}, "less_than"),
            ]}},
        ]},
        "exit_rules": [{"rule_id": "below_entry", "priority": 1, "relative_to": "entry", "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": _cmp(_field("close"), {"kind": "entry_price"}, "less_than"), "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = np.array([10.0, 10.0, 11.0, 10.0, 9.0])
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": [1, 1, 1, 5, 7]})

    trades, _ = run_v03_backtest(frame, strategy=strategy, capital=10_000, compound=False)

    assert trades.iloc[0].entry == "2020-01-06"  # S+1, not the 10% S candle.
    assert trades.iloc[0].exit == "2020-01-07"  # E+1 exit window.


def test_v03_allows_user_defined_doji_threshold_and_omitted_recovery_cap():
    payload = _strategy().model_dump(mode="json")
    payload["anchor"]["constraints"][0].pop("maximum_anchor_recovery_from_trough")
    payload["exit_rules"][1]["condition"]["conditions"][2]["conditions"][0]["body_to_open_threshold"] = .005

    strategy = TimedStrategyDefinition.model_validate(payload)

    assert strategy.anchor.constraints[0].maximum_anchor_recovery_from_trough is None


def test_v03_leaves_sample_end_position_out_of_closed_trade_kpis_and_reports_it():
    payload = _strategy().model_dump(mode="json")
    payload["anchor"]["constraints"] = []
    payload["lifecycle_policy"]["sample_end_open_position"] = "leave_open_excluded"
    payload["exit_rules"] = [{
        "rule_id": "protect_ma20", "priority": 1,
        "active_days": {"start_offset_days": 1, "end_offset_days": 10},
        "kind": "close_condition", "condition": _cmp(_field("close"), _sma(20), "less_than"), "execution": "close",
    }]
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = np.arange(1.0, 61.0)
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(close), freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0})
    trades, stats = run_v03_backtest(frame, strategy=strategy, capital=10_000, compound=True)
    assert trades.empty
    assert stats["n_trades"] == 0
    assert stats["open_position_count"] == 1
    assert stats["open_positions"][0]["period_end_close"] == 60.0
    assert stats["open_positions"][0]["status"] == "open_excluded_from_closed_trade_kpis"


def test_v03_volume_rank_treats_tied_maximum_as_rank_one_and_next_value_as_rank_two():
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=4, freq="B"),
        "Open": [1.0] * 4, "High": [1.0] * 4, "Low": [1.0] * 4, "Close": [1.0] * 4,
        "Volume": [100.0, 100.0, 90.0, 80.0],
    })
    rank = AnchorRunningVolumeRankOperand(kind="anchor_running_volume_rank", anchor="t0", field="volume")
    assert _temporal_operand(frame, rank, index=1, anchor_index=0, entry_index=0, entry_price=1.0) == 1
    assert _temporal_operand(frame, rank, index=2, anchor_index=0, entry_index=0, entry_price=1.0) == 3


def test_v03_supports_scaled_current_bar_operand_for_explicit_bearish_body_rule():
    """``Open × 0.975 > Close`` is the typed 2.5% bearish-body predicate."""
    condition = TemporalComparisonCondition.model_validate({
        "node_type": "comparison",
        "operator": "greater_than",
        "left": {
            "kind": "scaled_operand",
            "operand": {"kind": "market_field", "field": "open"},
            "multiplier": 0.975,
        },
        "right": {"kind": "market_field", "field": "close"},
    })
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=1),
        "Open": [100.0], "High": [101.0], "Low": [96.0], "Close": [97.0], "Volume": [100.0],
    })

    assert _temporal_condition(frame, condition, index=0, anchor_index=0, entry_index=0, entry_price=100.0)


def test_v03_persistent_state_activates_before_same_day_state_gated_exit():
    payload = {
        "schema_version": "0.3", "symbol": "TEST", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": _cmp(_field("close"), _sma(1), "greater_than_or_equal"), "constraints": []},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": _cmp(_field("close"), _sma(1), "greater_than_or_equal"), "execution": "close"},
        "persistent_states": [{
            "state_id": "high_state", "relative_to": "entry", "active_days": {"start_offset_days": 1},
            "activate_when": _cmp(_field("close"), {"kind": "scaled_entry_price", "multiplier": 1.1}, "greater_than"),
        }],
        "exit_rules": [{
            "rule_id": "high_state_doji", "priority": 1, "relative_to": "entry", "active_days": {"start_offset_days": 1},
            "requires_state": "high_state", "kind": "close_condition",
            "condition": {"node_type": "candlestick_pattern", "pattern": "doji", "body_to_open_threshold": .005}, "execution": "close",
        }],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    strategy = TimedStrategyDefinition.model_validate(payload)
    close = [10.0, 12.0, 11.0]
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=3, freq="B"), "Open": close, "High": close, "Low": close, "Close": close, "Volume": [100.0, 200.0, 100.0]})

    trades, _ = run_v03_backtest(frame, strategy=strategy, capital=10_000, compound=False)

    assert trades.iloc[0].exit == "2020-01-02"
    assert trades.iloc[0].reason == "v03:high_state_doji"
