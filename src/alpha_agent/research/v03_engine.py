"""Deterministic daily execution for the supported Strategy DSL v0.3 subset."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup, CrossCondition
from alpha_agent.domain.indicators import EMA, RSI, RollingMeanVolume, RollingReturn, SMA
from alpha_agent.domain.operands import IndicatorOperand, LaggedIndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorChangeConstraint, AnchorIndicatorOperand, AnchorMarketOperand, AnchorRunningMaximumOperand,
    CandlestickPatternCondition, CurrentIndicatorOperand, CurrentMarketOperand, EntryPriceOperand,
    ExitRule, OrderedExtremaDrawdownConstraint, ScaledEntryPriceOperand, TemporalComparisonCondition,
    TemporalCrossCondition,
    TemporalCondition, TemporalConditionGroup, TemporalOperand, TemporalScalarOperand, TimedStrategyDefinition,
)
from alpha_agent.errors import UnsupportedStrategyFeature
from rhein.engine.kpis import calculate_kpis


def _indicator_values(df: pd.DataFrame, indicator: object) -> np.ndarray:
    close = df["Close"]
    if isinstance(indicator, SMA): return close.rolling(indicator.window).mean().to_numpy()
    if isinstance(indicator, EMA): return close.ewm(span=indicator.window, adjust=False, min_periods=indicator.window).mean().to_numpy()
    if isinstance(indicator, RollingReturn): return (close / close.shift(indicator.window) - 1).to_numpy()
    if isinstance(indicator, RollingMeanVolume): return df["Volume"].rolling(indicator.window).mean().to_numpy()
    if isinstance(indicator, RSI):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / indicator.window, adjust=False, min_periods=indicator.window).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / indicator.window, adjust=False, min_periods=indicator.window).mean()
        return (100 - 100 / (1 + gain / loss)).to_numpy()
    raise UnsupportedStrategyFeature("v0.3 daily engine does not support this indicator", details={"indicator": type(indicator).__name__})


def _compare(left: float, right: float, operator: str) -> bool:
    if not (np.isfinite(left) and np.isfinite(right)): return False
    return {
        "greater_than": left > right, "less_than": left < right,
        "greater_than_or_equal": left >= right, "less_than_or_equal": left <= right,
        "equal": left == right,
    }[operator]


def _extremum_index(values: np.ndarray, *, kind: str, tie_break: str) -> int:
    """Select a finite-window extremum with an explicit deterministic tie rule."""
    selector = np.argmax if kind == "maximum" else np.argmin
    if tie_break == "earliest":
        return int(selector(values))
    # Reverse first so numpy's first match is the original window's latest one.
    return len(values) - 1 - int(selector(values[::-1]))


def _static_operand(df: pd.DataFrame, operand: Operand, index: int) -> float:
    if isinstance(operand, MarketFieldOperand): return float(df[operand.field.capitalize()].iloc[index])
    if isinstance(operand, IndicatorOperand): return float(_indicator_values(df, operand.indicator)[index])
    if isinstance(operand, LaggedIndicatorOperand):
        lagged_index = index + operand.offset_days
        return float(_indicator_values(df, operand.indicator)[lagged_index]) if lagged_index >= 0 else float("nan")
    if isinstance(operand, ScalarOperand): return float(operand.value)
    if isinstance(operand, ScaledOperand): return _static_operand(df, operand.operand, index) * operand.multiplier
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot evaluate this anchor operand")


def _static_condition(df: pd.DataFrame, condition: Condition, index: int) -> bool:
    if isinstance(condition, ComparisonCondition):
        return _compare(_static_operand(df, condition.left, index), _static_operand(df, condition.right, index), condition.operator)
    if isinstance(condition, CrossCondition):
        if index < 1:
            return False
        left, right = _static_operand(df, condition.left, index), _static_operand(df, condition.right, index)
        prior_left, prior_right = _static_operand(df, condition.left, index - 1), _static_operand(df, condition.right, index - 1)
        if not all(np.isfinite(value) for value in (left, right, prior_left, prior_right)):
            return False
        if condition.operator == "cross_above":
            return left > right and prior_left <= prior_right
        return left < right and prior_left >= prior_right
    if isinstance(condition, ConditionGroup):
        outcomes = [_static_condition(df, child, index) for child in condition.conditions]
        return all(outcomes) if condition.operator == "and" else any(outcomes)
    raise UnsupportedStrategyFeature("v0.3 daily engine supports comparison/group anchor conditions only")


def _anchor_operand_label(operand: Operand) -> str:
    if isinstance(operand, MarketFieldOperand): return operand.field.capitalize()
    if isinstance(operand, IndicatorOperand): return f"{operand.indicator.indicator.upper()}({operand.indicator.window})"
    if isinstance(operand, LaggedIndicatorOperand): return f"{operand.indicator.indicator.upper()}({operand.indicator.window})[t{operand.offset_days:+d}]"
    if isinstance(operand, ScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, ScaledOperand): return f"{_anchor_operand_label(operand.operand)} × {operand.multiplier:g}"
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot describe this anchor operand")


def _anchor_condition_checks(df: pd.DataFrame, condition: Condition, index: int, path: str = "anchor.condition") -> list[dict[str, object]]:
    """Materialize every anchor comparison used for one accepted C/t0 point."""
    if isinstance(condition, ConditionGroup):
        checks: list[dict[str, object]] = []
        for child_index, child in enumerate(condition.conditions):
            checks.extend(_anchor_condition_checks(df, child, index, f"{path}.conditions[{child_index}]"))
        return checks
    if isinstance(condition, ComparisonCondition):
        left, right = _static_operand(df, condition.left, index), _static_operand(df, condition.right, index)
        return [{
            "dsl_path": path,
            "left": _anchor_operand_label(condition.left),
            "left_value": round(left, 6) if np.isfinite(left) else None,
            "operator": condition.operator,
            "right": _anchor_operand_label(condition.right),
            "right_value": round(right, 6) if np.isfinite(right) else None,
            "passed": _compare(left, right, condition.operator),
        }]
    if isinstance(condition, CrossCondition):
        if index < 1:
            return [{"dsl_path": path, "left": _anchor_operand_label(condition.left), "left_value": None, "operator": condition.operator, "right": _anchor_operand_label(condition.right), "right_value": None, "prior_left_value": None, "prior_right_value": None, "passed": False}]
        left, right = _static_operand(df, condition.left, index), _static_operand(df, condition.right, index)
        prior_left, prior_right = _static_operand(df, condition.left, index - 1), _static_operand(df, condition.right, index - 1)
        return [{
            "dsl_path": path,
            "left": _anchor_operand_label(condition.left),
            "left_value": round(left, 6) if np.isfinite(left) else None,
            "operator": condition.operator,
            "right": _anchor_operand_label(condition.right),
            "right_value": round(right, 6) if np.isfinite(right) else None,
            "prior_left_value": round(prior_left, 6) if np.isfinite(prior_left) else None,
            "prior_right_value": round(prior_right, 6) if np.isfinite(prior_right) else None,
            "passed": _static_condition(df, condition, index),
        }]
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot materialize this anchor condition")


def _temporal_operand(df: pd.DataFrame, operand: TemporalOperand, *, index: int, anchor_index: int, entry_price: float) -> float:
    if isinstance(operand, CurrentMarketOperand): return float(df[operand.field.capitalize()].iloc[index])
    if isinstance(operand, CurrentIndicatorOperand): return float(_indicator_values(df, operand.indicator)[index])
    if isinstance(operand, TemporalScalarOperand): return float(operand.value)
    if isinstance(operand, AnchorMarketOperand): return float(df[operand.field.capitalize()].iloc[anchor_index])
    if isinstance(operand, AnchorIndicatorOperand):
        anchored_index = anchor_index + operand.offset_days
        return float(_indicator_values(df, operand.indicator)[anchored_index]) if anchored_index >= 0 else float("nan")
    if isinstance(operand, EntryPriceOperand): return entry_price
    if isinstance(operand, ScaledEntryPriceOperand): return entry_price * operand.multiplier
    if isinstance(operand, AnchorRunningMaximumOperand): return float(df["Volume"].iloc[anchor_index:index + 1].max())
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot evaluate this temporal operand")


def _temporal_condition(df: pd.DataFrame, condition: TemporalCondition, *, index: int, anchor_index: int, entry_price: float) -> bool:
    if isinstance(condition, TemporalComparisonCondition):
        return _compare(
            _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_price=entry_price),
            _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_price=entry_price),
            condition.operator,
        )
    if isinstance(condition, CandlestickPatternCondition):
        open_, close = float(df["Open"].iloc[index]), float(df["Close"].iloc[index])
        if not np.isfinite(open_) or open_ == 0 or not np.isfinite(close): return False
        body_ratio = abs(close - open_) / open_
        if condition.pattern == "doji": return body_ratio <= condition.body_to_open_threshold
        return close < open_ and (open_ - close) / open_ > condition.body_to_open_threshold
    if isinstance(condition, TemporalCrossCondition):
        if index < 1:
            return False
        left = _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_price=entry_price)
        right = _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_price=entry_price)
        prior_left = _temporal_operand(df, condition.left, index=index - 1, anchor_index=anchor_index, entry_price=entry_price)
        prior_right = _temporal_operand(df, condition.right, index=index - 1, anchor_index=anchor_index, entry_price=entry_price)
        if not all(np.isfinite(value) for value in (left, right, prior_left, prior_right)):
            return False
        if condition.operator == "cross_above":
            return left > right and prior_left <= prior_right
        return left < right and prior_left >= prior_right
    if isinstance(condition, TemporalConditionGroup):
        outcomes = [_temporal_condition(df, child, index=index, anchor_index=anchor_index, entry_price=entry_price) for child in condition.conditions]
        return all(outcomes) if condition.operator == "and" else any(outcomes)
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot evaluate this temporal condition")


def _anchor_constraints_hold(df: pd.DataFrame, strategy: TimedStrategyDefinition, index: int) -> tuple[bool, list[dict[str, object]]]:
    """Return both eligibility and the named structural points used to prove it."""
    event_points: list[dict[str, object]] = []
    ordered_constraint_count = 0
    for constraint in strategy.anchor.constraints:
        if isinstance(constraint, OrderedExtremaDrawdownConstraint):
            start = max(0, index - constraint.lookback_days + 1)
            window = df["Close"].iloc[start:index + 1].to_numpy(dtype=float)
            if len(window) < constraint.lookback_days or not np.isfinite(window).all(): return False, []
            peak_relative = _extremum_index(window, kind="maximum", tie_break=constraint.peak_tie_break)
            if peak_relative >= len(window) - 1: return False, []
            trough_tail = window[peak_relative + 1:]
            trough_relative = peak_relative + 1 + _extremum_index(
                trough_tail,
                kind="minimum",
                tie_break=constraint.trough_tie_break,
            )
            peak, trough, anchor_close = window[peak_relative], window[trough_relative], window[-1]
            if peak <= 0 or trough <= 0: return False, []
            drawdown = (peak - trough) / peak
            recovery = (anchor_close - trough) / trough
            if drawdown < constraint.minimum_peak_to_trough_drawdown or recovery > constraint.maximum_anchor_recovery_from_trough:
                return False, []
            ordered_constraint_count += 1
            suffix = "" if ordered_constraint_count == 1 else str(ordered_constraint_count)
            event_points.extend([
                {"event_id": f"ordered_extrema_{ordered_constraint_count}_peak", "label": f"A{suffix}：窗口最高收盘", "date": str(pd.Timestamp(df["Date"].iloc[start + peak_relative]).date()), "price": round(float(peak), 4)},
                {"event_id": f"ordered_extrema_{ordered_constraint_count}_trough", "label": f"B{suffix}：A 后最低收盘", "date": str(pd.Timestamp(df["Date"].iloc[start + trough_relative]).date()), "price": round(float(trough), 4)},
            ])
        elif isinstance(constraint, AnchorIndicatorChangeConstraint):
            prior = index + constraint.comparison_offset_days
            if prior < 0: return False, []
            values = _indicator_values(df, constraint.indicator)
            previous, current = values[prior], values[index]
            if not (np.isfinite(previous) and previous != 0 and np.isfinite(current)): return False, []
            change = current / previous - 1
            if constraint.operator == "greater_than":
                if change <= constraint.minimum_relative_change: return False, []
            elif change < constraint.minimum_relative_change:
                return False, []
        else:
            raise UnsupportedStrategyFeature("v0.3 daily engine cannot execute this anchor constraint", details={"constraint": constraint.kind})
    return True, event_points


def _rule_active(rule: ExitRule, relative_day: int) -> bool:
    return relative_day >= rule.active_days.start_offset_days and (rule.active_days.end_offset_days is None or relative_day <= rule.active_days.end_offset_days)


def run_v03_backtest(df: pd.DataFrame, *, strategy: TimedStrategyDefinition, capital: float, compound: bool, cost_bps: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Run v0.3 exactly at daily closes; never approximate its stateful rules."""
    if strategy.schema_version != "0.3" or strategy.lifecycle_policy is None:
        raise UnsupportedStrategyFeature("v0.3 daily engine requires a v0.3 strategy with lifecycle_policy")
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df): raise ValueError("OHLCV input is missing required columns")
    equity, trades, open_positions, index = capital, [], [], 0
    n = len(df)
    while index < n:
        if not _static_condition(df, strategy.anchor.condition, index):
            index += 1
            continue
        anchor_checks = _anchor_condition_checks(df, strategy.anchor.condition, index)
        if not all(bool(check["passed"]) for check in anchor_checks):
            raise AssertionError("accepted anchor point has a failed materialized condition")
        constraints_hold, event_points = _anchor_constraints_hold(df, strategy, index)
        if not constraints_hold:
            index += 1
            continue
        entry_index = index + strategy.entry.active_day.start_offset_days
        if entry_index >= n: break
        entry_price = float(df["Close"].iloc[entry_index])
        if not _temporal_condition(df, strategy.entry.condition, index=entry_index, anchor_index=index, entry_price=entry_price):
            index += 1
            continue
        exit_index, reason = None, None
        for day in range(entry_index + 1, n):
            matching = [rule for rule in strategy.exit_rules if _rule_active(rule, day - index) and rule.condition is not None and _temporal_condition(df, rule.condition, index=day, anchor_index=index, entry_price=entry_price)]
            if matching:
                winner = min(matching, key=lambda rule: rule.priority)
                exit_index, reason = day, f"v03:{winner.rule_id}"
                break
        if exit_index is None:
            if strategy.lifecycle_policy.sample_end_open_position == "leave_open_excluded":
                final_index = n - 1
                final_close = float(df["Close"].iloc[final_index])
                open_positions.append({
                    "signal": str(pd.Timestamp(df["Date"].iloc[index]).date()),
                    "entry": str(pd.Timestamp(df["Date"].iloc[entry_index]).date()),
                    "entry_px": round(entry_price, 4),
                    "period_end": str(pd.Timestamp(df["Date"].iloc[final_index]).date()),
                    "period_end_close": round(final_close, 4),
                    "unrealized_return_pct": round((final_close / entry_price - 1) * 100, 3),
                    "days_held": final_index - entry_index,
                    "status": "open_excluded_from_closed_trade_kpis",
                })
                break
            exit_index, reason = n - 1, "v03:sample_end_force_close"
        exit_price = float(df["Close"].iloc[exit_index])
        net_return = exit_price / entry_price - 1 - 2 * cost_bps / 10_000
        stake = equity if compound else capital
        pnl = stake * net_return
        equity += pnl
        signal_date = str(pd.Timestamp(df["Date"].iloc[index]).date())
        entry_date = str(pd.Timestamp(df["Date"].iloc[entry_index]).date())
        exit_date = str(pd.Timestamp(df["Date"].iloc[exit_index]).date())
        event_points.extend([
            {"event_id": "anchor", "label": "C：t0 基准点", "date": signal_date, "price": round(float(df["Close"].iloc[index]), 4)},
            {"event_id": "entry", "label": "入场", "date": entry_date, "price": round(entry_price, 4)},
            {"event_id": "exit", "label": f"出场：{reason}", "date": exit_date, "price": round(exit_price, 4)},
        ])
        trades.append({"signal": signal_date, "entry": entry_date, "exit": exit_date, "days_held": exit_index - entry_index, "entry_px": round(entry_price, 4), "exit_px": round(exit_price, 4), "ret_pct": round(net_return * 100, 3), "pnl_eur": round(pnl, 2), "reason": reason, "event_points": event_points, "anchor_checks": anchor_checks})
        if not strategy.lifecycle_policy.allow_reentry_after_exit: break
        index = exit_index + 1
    trade_frame = pd.DataFrame(trades)
    stats = calculate_kpis(trade_frame, capital, compound)
    stats["open_position_count"] = len(open_positions)
    stats["open_positions"] = open_positions
    return trade_frame, stats
