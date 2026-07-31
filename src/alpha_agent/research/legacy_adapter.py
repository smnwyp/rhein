"""Compile the explicitly supported v0.2 timed DSL subset to Rhein's deterministic engine."""
from __future__ import annotations

from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup
from alpha_agent.domain.indicators import RSI, RollingMeanVolume, RollingReturn, SMA
from alpha_agent.domain.operands import IndicatorOperand, MarketFieldOperand, ScalarOperand
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.errors import UnsupportedStrategyFeature
from rhein.backtest import DEFAULT_STRATEGY


def _leaves(condition: Condition) -> list[ComparisonCondition]:
    if isinstance(condition, ComparisonCondition):
        return [condition]
    if isinstance(condition, ConditionGroup) and condition.operator == "and":
        return [leaf for child in condition.conditions for leaf in _leaves(child)]
    raise UnsupportedStrategyFeature("timed backtest currently requires an AND-only anchor condition tree")


def compile_timed_strategy(strategy: TimedStrategyDefinition, *, approximate_intraday_with_daily_low: bool) -> dict:
    """Return engine parameters or a typed error; never drop an unrecognised rule."""
    params = dict(DEFAULT_STRATEGY)
    params.update({
        "use_signal_band": False, "use_baseline_rsi": False,
        "use_baseline_close_above_fast_sma": False, "use_baseline_fast_above_slow_sma": False,
        "use_baseline_close_above_sma20": False, "use_baseline_volume_sma": False,
        "use_baseline_bullish_candle": False, "use_baseline_sma20_rising": False,
        "use_baseline_prior_low": False, "use_baseline_max_rise": False,
        "use_entry_close_vs_t0": True, "use_early_stop": False,
        "use_exit_below_entry": False, "use_exit_below_sma": False, "use_forced_exit": False,
        "cost_bps": 0.0,
    })
    for leaf in _leaves(strategy.anchor.condition):
        left, right = leaf.left, leaf.right
        if isinstance(left, IndicatorOperand) and isinstance(left.indicator, RollingReturn) and isinstance(right, ScalarOperand):
            if leaf.operator in ("greater_than_or_equal", "greater_than"):
                params.update(use_signal_band=True, band_lo=right.value)
            elif leaf.operator in ("less_than_or_equal", "less_than"):
                params.update(use_signal_band=True, band_hi=right.value)
            else: raise UnsupportedStrategyFeature("unsupported rolling-return comparison")
        elif isinstance(left, IndicatorOperand) and isinstance(left.indicator, RSI) and isinstance(right, ScalarOperand) and leaf.operator in ("less_than", "less_than_or_equal"):
            params.update(use_baseline_rsi=True, baseline_rsi_period=left.indicator.window, baseline_rsi_max=right.value)
        elif isinstance(left, MarketFieldOperand) and left.field == "close" and isinstance(right, IndicatorOperand) and isinstance(right.indicator, SMA):
            if right.indicator.window == 20:
                params["use_baseline_close_above_sma20"] = True
            else:
                params.update(use_baseline_close_above_fast_sma=True, entry_trend_fast_sma=right.indicator.window)
        elif isinstance(left, IndicatorOperand) and isinstance(left.indicator, SMA) and isinstance(right, IndicatorOperand) and isinstance(right.indicator, SMA):
            params.update(use_baseline_fast_above_slow_sma=True, entry_trend_fast_sma=left.indicator.window, entry_trend_slow_sma=right.indicator.window)
        elif isinstance(left, IndicatorOperand) and isinstance(left.indicator, RollingMeanVolume) and isinstance(right, IndicatorOperand) and isinstance(right.indicator, RollingMeanVolume):
            params.update(use_baseline_volume_sma=True, entry_volume_fast_window=left.indicator.window, entry_volume_slow_window=right.indicator.window)
        elif isinstance(left, MarketFieldOperand) and left.field == "close" and isinstance(right, MarketFieldOperand) and right.field == "open":
            params["use_baseline_bullish_candle"] = True
        else:
            raise UnsupportedStrategyFeature("anchor condition cannot yet be compiled to the deterministic daily engine", details={"condition": leaf.model_dump(mode="json")})
    for constraint in strategy.anchor.constraints:
        if constraint.kind == "rolling_low_anchor_constraint":
            if constraint.reference_field != "low":
                raise UnsupportedStrategyFeature("the current daily backtest only supports rolling-low constraints based on Low", details={"reference_field": constraint.reference_field})
            params.update(use_baseline_prior_low=constraint.low_must_precede_anchor, use_baseline_max_rise=True, baseline_lookback=constraint.lookback_days, baseline_max_rise=constraint.maximum_anchor_close_gain)
        elif constraint.kind == "anchor_indicator_change_constraint":
            if not isinstance(constraint.indicator, SMA) or constraint.indicator.window != 20 or constraint.comparison_offset_days != -2:
                raise UnsupportedStrategyFeature("only the MA20(t0) versus MA20(t0-2) anchor slope is currently executable")
            params["use_baseline_sma20_rising"] = True
    if strategy.entry.active_day.start_offset_days != strategy.entry.active_day.end_offset_days:
        raise UnsupportedStrategyFeature("entry must occur on one explicit relative day")
    params["entry_lag"] = strategy.entry.active_day.start_offset_days
    for rule in strategy.exit_rules:
        if rule.kind == "intraday_price_trigger":
            if not approximate_intraday_with_daily_low:
                raise UnsupportedStrategyFeature("intraday exit requires intraday data or explicit daily-low approximation")
            if rule.active_days.start_offset_days == rule.active_days.end_offset_days:
                params.update(use_forced_exit=True, forced_exit_day=rule.active_days.start_offset_days, use_forced_exit_intraday_protection=True, forced_exit_intraday_stop_pct=1 - rule.price_trigger.multiplier)
            else:
                params.update(use_early_stop=True, hard_stop_days=tuple(range(rule.active_days.start_offset_days, rule.active_days.end_offset_days + 1)), stop_pct=1 - rule.price_trigger.multiplier, stop_intraday=True)
        elif rule.kind == "forced_close":
            if rule.active_days.start_offset_days != rule.active_days.end_offset_days:
                raise UnsupportedStrategyFeature("forced close needs one explicit day")
            params.update(use_forced_exit=True, forced_exit_day=rule.active_days.start_offset_days)
        else:
            # The engine supports these standard close exits once early-stop days have passed.
            params["use_exit_below_entry"] = True
    return params
