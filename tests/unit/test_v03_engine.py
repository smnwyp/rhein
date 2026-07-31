import numpy as np
import pandas as pd

from alpha_agent.domain.conditions import CrossCondition
from alpha_agent.domain.indicators import SMA
from alpha_agent.domain.operands import IndicatorOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorOperand,
    TemporalComparisonCondition,
    TemporalCrossCondition,
    TemporalScalarOperand,
    TimedStrategyDefinition,
)
from alpha_agent.research.v03_engine import _anchor_condition_checks, _anchor_constraints_hold, _static_condition, _temporal_condition, _temporal_operand, run_v03_backtest


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
    assert {check["left"] for check in trades.iloc[0].anchor_checks} == {"Close", "SMA(5)", "SMA(10)"}
    assert stats["n_trades"] == 1


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
