"""Deterministic daily execution for the supported Strategy DSL v0.3 subset."""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd

from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup, CrossCondition, RollingComparisonCountCondition, RollingCrossCountCondition
from alpha_agent.domain.indicators import ADX, DMIADX, EMA, MACDLine, MACDPercentLine, MACDSignal, MarketIndexMACDLine, RSI, RollingMaximum, RollingMeanVolume, RollingMinimum, RollingReturn, SMA
from alpha_agent.domain.operands import IndicatorOperand, LaggedIndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorChangeConstraint, AnchorIndicatorOperand, AnchorMarketOperand, AnchorRunningMaximumOperand, AnchorRunningVolumeRankOperand,
    CandlestickPatternCondition, CurrentIndicatorOperand, CurrentMarketOperand, LaggedIndicatorOperand as TemporalLaggedIndicatorOperand, LaggedMarketOperand, EntryPriceOperand,
    EntryRunningMaximumOperand, ExitRule, OrderedExtremaDrawdownConstraint, RollingLowAnchorConstraint,
    RollingVolumeRankOperand, ScaledEntryPriceOperand, TemporalComparisonCondition, TemporalScaledOperand,
    TemporalCrossCondition,
    TemporalCondition, TemporalConditionGroup, TemporalOperand, TemporalScalarOperand, TimedStrategyDefinition,
)
from alpha_agent.errors import UnsupportedStrategyFeature
from rhein.engine.kpis import calculate_kpis


def attach_market_index_context(df: pd.DataFrame, market_index: pd.DataFrame) -> pd.DataFrame:
    """Return an asset frame with its external market index aligned by date.

    The alignment is deliberately exact: silently forward-filling a missing
    index close would turn a data-quality problem into a different trading
    rule.  The aligned series lives only in dataframe ``attrs`` for the
    duration of deterministic execution and is never persisted in a strategy.
    """
    if "Date" not in df or not {"Date", "Close"}.issubset(market_index):
        raise ValueError("asset and market-index inputs must contain Date; market index must contain Close")
    asset_dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    index_frame = market_index.loc[:, ["Date", "Close"]].copy()
    index_frame["Date"] = pd.to_datetime(index_frame["Date"], errors="coerce").dt.normalize()
    index_close = pd.to_numeric(index_frame["Close"], errors="coerce")
    by_date = pd.Series(index_close.to_numpy(dtype=float), index=index_frame["Date"])
    by_date = by_date[~by_date.index.duplicated(keep="last")]
    aligned = by_date.reindex(asset_dates)
    # ``reindex(asset_dates)`` uses dates as the resulting Series index,
    # while ``asset_dates`` retains the input frame's RangeIndex.  Select by
    # position to keep the missing-date audit aligned to the asset rows.
    missing = asset_dates.loc[aligned.isna().to_numpy()]
    if len(missing):
        raise UnsupportedStrategyFeature(
            "configured market-index series is missing trading dates required by this asset",
            details={"missing_date_count": int(len(missing)), "sample_missing_dates": [str(value.date()) for value in missing[:5]]},
        )
    result = df.copy()
    result.attrs["_alpha_market_index_close"] = aligned.to_numpy(dtype=float)
    result.attrs["_alpha_market_index_symbol"] = market_index.attrs.get("_alpha_market_index_symbol")
    return result


def _indicator_values(df: pd.DataFrame, indicator: object) -> np.ndarray:
    # An anchor tree evaluates several conditions for every trading day.  The
    # same SMA/MACD series must be calculated once per input frame, not once
    # per condition/day.  ``attrs`` keeps this deterministic cache local to
    # the in-memory backtest frame and out of persisted strategy artifacts.
    cache = df.attrs.setdefault("_alpha_indicator_cache", {})
    try:
        if indicator in cache:
            return cache[indicator]
    except TypeError:
        # All supported DSL indicator models are frozen/hashable.  Preserve a
        # safe fallback for callers supplying an unsupported object so the
        # normal typed error below remains authoritative.
        cache = None
    close = df["Close"]
    if isinstance(indicator, SMA):
        values = close.rolling(indicator.window).mean().to_numpy()
    elif isinstance(indicator, EMA):
        values = close.ewm(span=indicator.window, adjust=False, min_periods=indicator.window).mean().to_numpy()
    elif isinstance(indicator, RollingReturn):
        values = (close / close.shift(indicator.window) - 1).to_numpy()
    elif isinstance(indicator, RollingMinimum):
        values = df[indicator.field.capitalize()].rolling(indicator.window).min().to_numpy()
    elif isinstance(indicator, RollingMaximum):
        values = df[indicator.field.capitalize()].rolling(indicator.window).max().to_numpy()
    elif isinstance(indicator, RollingMeanVolume): values = df["Volume"].rolling(indicator.window).mean().to_numpy()
    elif isinstance(indicator, ADX):
        high, low = df["High"], df["Low"]
        previous_close = close.shift(1)
        # Do not use ``pd.concat`` here.  This frame intentionally stores the
        # indicator cache in ``attrs``; pandas tries to compare every input
        # object's attrs during concat, and numpy cache values make that
        # comparison ambiguous.  Numpy keeps the Wilder calculation pure and
        # independent of cache implementation details.
        true_range = pd.Series(
            np.maximum.reduce([
                (high - low).to_numpy(dtype=float),
                (high - previous_close).abs().to_numpy(dtype=float),
                (low - previous_close).abs().to_numpy(dtype=float),
            ]),
            index=df.index,
        )
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        alpha = 1 / indicator.window
        atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=indicator.window).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=indicator.window).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=indicator.window).mean() / atr
        directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        values = directional_index.ewm(alpha=alpha, adjust=False, min_periods=indicator.window).mean().to_numpy()
    elif isinstance(indicator, DMIADX):
        # Chinese charting DMI(N, M): rolling sums for DI, then MA(DX, M)
        # for ADX. Keep it distinct from Wilder ADX(N); the source language
        # explicitly supplies two parameters and they are not interchangeable.
        high, low = df["High"], df["Low"]
        previous_close = close.shift(1)
        # Keep this numpy-based for the same reason as the Wilder ADX branch:
        # an index-gated frame carries array-valued attrs, and pandas concat
        # attempts to compare those attrs while finalising its result.
        true_range = pd.Series(
            np.maximum.reduce([
                (high - low).to_numpy(dtype=float),
                (high - previous_close).abs().to_numpy(dtype=float),
                (low - previous_close).abs().to_numpy(dtype=float),
            ]),
            index=df.index,
        ).fillna(high - low)
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_sum = true_range.rolling(indicator.directional_window, min_periods=1).sum()
        plus_di = 100 * plus_dm.rolling(indicator.directional_window, min_periods=1).sum() / tr_sum
        minus_di = 100 * minus_dm.rolling(indicator.directional_window, min_periods=1).sum() / tr_sum
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        values = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0).rolling(indicator.adx_window, min_periods=1).mean().to_numpy()
    elif isinstance(indicator, RSI):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / indicator.window, adjust=False, min_periods=indicator.window).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / indicator.window, adjust=False, min_periods=indicator.window).mean()
        values = (100 - 100 / (1 + gain / loss)).to_numpy()
    elif isinstance(indicator, (MACDLine, MACDSignal, MACDPercentLine)):
        fast = close.ewm(span=indicator.fast_window, adjust=False, min_periods=indicator.slow_window).mean()
        slow = close.ewm(span=indicator.slow_window, adjust=False, min_periods=indicator.slow_window).mean()
        line = fast - slow
        if isinstance(indicator, MACDSignal):
            values = line.ewm(span=indicator.signal_window, adjust=False, min_periods=indicator.signal_window).mean().to_numpy()
        elif isinstance(indicator, MACDPercentLine):
            values = (line / close.replace(0, np.nan) * 100).to_numpy()
        else:
            values = line.to_numpy()
    elif isinstance(indicator, MarketIndexMACDLine):
        index_close = df.attrs.get("_alpha_market_index_close")
        if index_close is None:
            raise UnsupportedStrategyFeature(
                "v0.3 daily engine needs the configured market-index series before it can evaluate market-index DIF",
                details={"index_symbol": indicator.index_symbol},
            )
        configured_symbol = df.attrs.get("_alpha_market_index_symbol")
        if configured_symbol != indicator.index_symbol:
            raise UnsupportedStrategyFeature(
                "configured market-index series does not match the strategy's declared index symbol",
                details={"required_symbol": indicator.index_symbol, "configured_symbol": configured_symbol},
            )
        index_close = np.asarray(index_close, dtype=float)
        if len(index_close) != len(df):
            raise UnsupportedStrategyFeature(
                "configured market-index series is not aligned to the asset trading dates",
                details={"index_symbol": indicator.index_symbol, "asset_rows": len(df), "index_rows": len(index_close)},
            )
        series = pd.Series(index_close, index=df.index)
        fast = series.ewm(span=indicator.fast_window, adjust=False, min_periods=indicator.slow_window).mean()
        slow = series.ewm(span=indicator.slow_window, adjust=False, min_periods=indicator.slow_window).mean()
        values = (fast - slow).to_numpy()
    else:
        raise UnsupportedStrategyFeature("v0.3 daily engine does not support this indicator", details={"indicator": type(indicator).__name__})
    if cache is not None:
        cache[indicator] = values
    return values


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
    if isinstance(condition, RollingComparisonCountCondition):
        start = index - condition.lookback_days + 1
        if start < 0:
            return False
        passed = sum(
            _static_condition(df, condition.comparison, candidate_index)
            for candidate_index in range(start, index + 1)
        )
        return passed >= condition.minimum_true_count
    if isinstance(condition, RollingCrossCountCondition):
        start = index - condition.lookback_days + 1
        if start < 1:
            return False
        cross = CrossCondition(node_type="cross", operator=condition.operator, left=condition.left, right=condition.right)
        return sum(_static_condition(df, cross, candidate) for candidate in range(start, index + 1)) >= condition.minimum_true_count
    raise UnsupportedStrategyFeature("v0.3 daily engine supports comparison/group anchor conditions only")


def _anchor_operand_label(operand: Operand) -> str:
    if isinstance(operand, MarketFieldOperand): return operand.field.capitalize()
    if isinstance(operand, IndicatorOperand): return _indicator_label(operand.indicator)
    if isinstance(operand, LaggedIndicatorOperand): return f"{_indicator_label(operand.indicator)}[t{operand.offset_days:+d}]"
    if isinstance(operand, ScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, ScaledOperand): return f"{_anchor_operand_label(operand.operand)} × {operand.multiplier:g}"
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot describe this anchor operand")


def _indicator_label(indicator: object) -> str:
    if isinstance(indicator, (MACDLine, MACDSignal, MACDPercentLine)):
        if isinstance(indicator, MACDPercentLine):
            component = indicator.label or "归一化 EMA 差值"
        else:
            component = "慢线" if isinstance(indicator, MACDSignal) else "快线"
        windows = f"{indicator.fast_window},{indicator.slow_window}"
        if indicator.signal_window is not None:
            windows += f",{indicator.signal_window}"
        return f"{component}({windows})" if isinstance(indicator, MACDPercentLine) else f"MACD{component}({windows})"
    if isinstance(indicator, DMIADX):
        return f"DMI ADX({indicator.directional_window},{indicator.adx_window})"
    if isinstance(indicator, MarketIndexMACDLine):
        return f"指数 {indicator.index_symbol} DIF({indicator.fast_window},{indicator.slow_window},{indicator.signal_window})"
    return f"{indicator.indicator.upper()}({indicator.window})"  # type: ignore[union-attr]


def _anchor_condition_checks(df: pd.DataFrame, condition: Condition, index: int, path: str = "anchor.condition") -> list[dict[str, object]]:
    """Materialize every anchor comparison used for one accepted C/t0 point."""
    if isinstance(condition, ConditionGroup):
        checks: list[dict[str, object]] = []
        for child_index, child in enumerate(condition.conditions):
            checks.extend(_anchor_condition_checks(df, child, index, f"{path}.conditions[{child_index}]"))
        # A false leaf inside an OR is expected when another branch is true.
        # Persist the group result alongside leaves so audit UIs never imply
        # that every printed leaf is independently required for entry.
        checks.append({
            "dsl_path": path,
            "left": f"组合条件（{condition.operator.upper()}）",
            "left_value": None,
            "operator": "group_result",
            "right": "全部子条件" if condition.operator == "and" else "任一子条件",
            "right_value": None,
            "passed": _static_condition(df, condition, index),
        })
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
    if isinstance(condition, RollingComparisonCountCondition):
        start = index - condition.lookback_days + 1
        count = sum(_static_condition(df, condition.comparison, day) for day in range(max(0, start), index + 1))
        return [{
            "dsl_path": path,
            "left": f"{condition.lookback_days} 日内满足次数",
            "left_value": count,
            "operator": "greater_than_or_equal",
            "right": str(condition.minimum_true_count),
            "right_value": condition.minimum_true_count,
            "passed": start >= 0 and count >= condition.minimum_true_count,
        }]
    if isinstance(condition, RollingCrossCountCondition):
        start = index - condition.lookback_days + 1
        cross = CrossCondition(node_type="cross", operator=condition.operator, left=condition.left, right=condition.right)
        count = sum(_static_condition(df, cross, day) for day in range(max(1, start), index + 1))
        return [{
            "dsl_path": path,
            "left": f"{condition.lookback_days} 日内{condition.operator}次数",
            "left_value": count,
            "operator": "greater_than_or_equal",
            "right": str(condition.minimum_true_count),
            "right_value": condition.minimum_true_count,
            "passed": start >= 1 and count >= condition.minimum_true_count,
        }]
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot materialize this anchor condition")


def _temporal_operand(df: pd.DataFrame, operand: TemporalOperand, *, index: int, anchor_index: int, entry_price: float, entry_index: int | None = None) -> float:
    if isinstance(operand, CurrentMarketOperand): return float(df[operand.field.capitalize()].iloc[index])
    if isinstance(operand, LaggedMarketOperand):
        lagged_index = index + operand.offset_days
        return float(df[operand.field.capitalize()].iloc[lagged_index]) if lagged_index >= 0 else float("nan")
    if isinstance(operand, CurrentIndicatorOperand): return float(_indicator_values(df, operand.indicator)[index])
    if isinstance(operand, TemporalLaggedIndicatorOperand):
        lagged_index = index + operand.offset_days
        return float(_indicator_values(df, operand.indicator)[lagged_index]) if lagged_index >= 0 else float("nan")
    if isinstance(operand, TemporalScalarOperand): return float(operand.value)
    if isinstance(operand, AnchorMarketOperand): return float(df[operand.field.capitalize()].iloc[anchor_index])
    if isinstance(operand, AnchorIndicatorOperand):
        anchored_index = anchor_index + operand.offset_days
        return float(_indicator_values(df, operand.indicator)[anchored_index]) if anchored_index >= 0 else float("nan")
    if isinstance(operand, EntryPriceOperand): return entry_price
    if isinstance(operand, ScaledEntryPriceOperand): return entry_price * operand.multiplier
    if isinstance(operand, TemporalScaledOperand):
        return _temporal_operand(
            df,
            operand.operand,
            index=index,
            anchor_index=anchor_index,
            entry_index=entry_index,
            entry_price=entry_price,
        ) * operand.multiplier
    if isinstance(operand, AnchorRunningMaximumOperand): return float(df["Volume"].iloc[anchor_index:index + 1].max())
    if isinstance(operand, EntryRunningMaximumOperand):
        start = anchor_index if entry_index is None else entry_index
        return float(df["Volume"].iloc[start:index + 1].max())
    if isinstance(operand, AnchorRunningVolumeRankOperand):
        current = float(df["Volume"].iloc[index])
        values = df["Volume"].iloc[anchor_index:index + 1].to_numpy(dtype=float)
        if not np.isfinite(current) or not np.isfinite(values).all():
            return float("nan")
        # Competition rank: 100, 100, 90 are ranks 1, 1, 3. Thus an equal
        # maximum or second-highest volume satisfies rank <= 2 as specified.
        return float(1 + np.count_nonzero(values > current))
    if isinstance(operand, RollingVolumeRankOperand):
        start = index - operand.window + 1
        if start < 0:
            return float("nan")
        current = float(df["Volume"].iloc[index])
        values = df["Volume"].iloc[start:index + 1].to_numpy(dtype=float)
        if not np.isfinite(current) or not np.isfinite(values).all():
            return float("nan")
        return float(1 + np.count_nonzero(values > current))
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot evaluate this temporal operand")


def _temporal_condition(df: pd.DataFrame, condition: TemporalCondition, *, index: int, anchor_index: int, entry_price: float, entry_index: int | None = None) -> bool:
    if isinstance(condition, TemporalComparisonCondition):
        return _compare(
            _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price),
            _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price),
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
        left = _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        right = _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        prior_left = _temporal_operand(df, condition.left, index=index - 1, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        prior_right = _temporal_operand(df, condition.right, index=index - 1, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        if not all(np.isfinite(value) for value in (left, right, prior_left, prior_right)):
            return False
        if condition.operator == "cross_above":
            return left > right and prior_left <= prior_right
        return left < right and prior_left >= prior_right
    if isinstance(condition, TemporalConditionGroup):
        outcomes = [_temporal_condition(df, child, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price) for child in condition.conditions]
        return all(outcomes) if condition.operator == "and" else any(outcomes)
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot evaluate this temporal condition")


def _temporal_operand_label(operand: TemporalOperand) -> str:
    """Human-readable labels for an auditable temporal-condition snapshot."""
    if isinstance(operand, CurrentMarketOperand): return operand.field.capitalize()
    if isinstance(operand, LaggedMarketOperand): return f"{operand.field.capitalize()}[t{operand.offset_days:+d}]"
    if isinstance(operand, CurrentIndicatorOperand): return _indicator_label(operand.indicator)
    if isinstance(operand, TemporalLaggedIndicatorOperand): return f"{_indicator_label(operand.indicator)}[t{operand.offset_days:+d}]"
    if isinstance(operand, TemporalScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, AnchorMarketOperand): return f"t0 {operand.field.capitalize()}"
    if isinstance(operand, AnchorIndicatorOperand): return f"t0{operand.offset_days:+d} {_indicator_label(operand.indicator)}"
    if isinstance(operand, EntryPriceOperand): return "实际入场价"
    if isinstance(operand, ScaledEntryPriceOperand): return f"实际入场价 × {operand.multiplier:g}"
    if isinstance(operand, TemporalScaledOperand): return f"{_temporal_operand_label(operand.operand)} × {operand.multiplier:g}"
    if isinstance(operand, AnchorRunningMaximumOperand): return "t0 至今成交量最大值"
    if isinstance(operand, EntryRunningMaximumOperand): return "入场至今成交量最大值"
    if isinstance(operand, AnchorRunningVolumeRankOperand): return "t0 至今成交量排名"
    if isinstance(operand, RollingVolumeRankOperand): return f"{operand.window} 日成交量排名"
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot describe this temporal operand")


def _audit_row(
    *,
    section: str,
    dsl_path: str,
    left: str,
    left_value: float | int | None,
    operator: str,
    right: str,
    right_value: float | int | None,
    passed: bool,
    prior_left_value: float | int | None = None,
    prior_right_value: float | int | None = None,
) -> dict[str, object]:
    return {
        "section": section,
        "dsl_path": dsl_path,
        "left": left,
        "left_value": round(float(left_value), 6) if left_value is not None and np.isfinite(left_value) else None,
        "operator": operator,
        "right": right,
        "right_value": round(float(right_value), 6) if right_value is not None and np.isfinite(right_value) else None,
        "prior_left_value": round(float(prior_left_value), 6) if prior_left_value is not None and np.isfinite(prior_left_value) else None,
        "prior_right_value": round(float(prior_right_value), 6) if prior_right_value is not None and np.isfinite(prior_right_value) else None,
        "passed": passed,
    }


def _temporal_condition_checks(
    df: pd.DataFrame,
    condition: TemporalCondition,
    *,
    index: int,
    anchor_index: int,
    entry_index: int,
    entry_price: float,
    section: str,
    path: str,
) -> list[dict[str, object]]:
    """Materialize actual values for each temporal condition leaf and group."""
    if isinstance(condition, TemporalConditionGroup):
        rows: list[dict[str, object]] = []
        for child_index, child in enumerate(condition.conditions):
            rows.extend(_temporal_condition_checks(
                df, child, index=index, anchor_index=anchor_index, entry_index=entry_index,
                entry_price=entry_price, section=section, path=f"{path}.conditions[{child_index}]",
            ))
        group_passed = _temporal_condition(
            df, condition, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price,
        )
        rows.append(_audit_row(
            section=section, dsl_path=path, left=f"组合条件（{condition.operator.upper()}）", left_value=None,
            operator="结果", right="所有子条件" if condition.operator == "and" else "任一子条件", right_value=None,
            passed=group_passed,
        ))
        return rows
    if isinstance(condition, TemporalComparisonCondition):
        left = _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        right = _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        return [_audit_row(
            section=section, dsl_path=path, left=_temporal_operand_label(condition.left), left_value=left,
            operator=condition.operator, right=_temporal_operand_label(condition.right), right_value=right,
            passed=_compare(left, right, condition.operator),
        )]
    if isinstance(condition, TemporalCrossCondition):
        left = _temporal_operand(df, condition.left, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        right = _temporal_operand(df, condition.right, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
        prior_left = _temporal_operand(df, condition.left, index=index - 1, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price) if index else float("nan")
        prior_right = _temporal_operand(df, condition.right, index=index - 1, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price) if index else float("nan")
        return [_audit_row(
            section=section, dsl_path=path, left=_temporal_operand_label(condition.left), left_value=left,
            operator=condition.operator, right=_temporal_operand_label(condition.right), right_value=right,
            prior_left_value=prior_left, prior_right_value=prior_right,
            passed=_temporal_condition(df, condition, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price),
        )]
    if isinstance(condition, CandlestickPatternCondition):
        open_, close = float(df["Open"].iloc[index]), float(df["Close"].iloc[index])
        if not np.isfinite(open_) or open_ == 0 or not np.isfinite(close):
            body_ratio = float("nan")
        else:
            body_ratio = abs(close - open_) / open_
        if condition.pattern == "doji":
            return [_audit_row(
                section=section, dsl_path=path, left="|收盘−开盘|/开盘", left_value=body_ratio,
                operator="less_than_or_equal", right="十字星实体阈值", right_value=condition.body_to_open_threshold,
                passed=_temporal_condition(df, condition, index=index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price),
            )]
        bearish_ratio = (open_ - close) / open_ if np.isfinite(open_) and open_ else float("nan")
        return [
            _audit_row(section=section, dsl_path=f"{path}.bearish", left="收盘", left_value=close, operator="less_than", right="开盘", right_value=open_, passed=close < open_),
            _audit_row(section=section, dsl_path=f"{path}.body", left="(开盘−收盘)/开盘", left_value=bearish_ratio, operator="greater_than", right="大阴线实体阈值", right_value=condition.body_to_open_threshold, passed=bearish_ratio > condition.body_to_open_threshold),
        ]
    raise UnsupportedStrategyFeature("v0.3 daily engine cannot materialize this temporal condition")


def _anchor_constraint_checks(df: pd.DataFrame, strategy: TimedStrategyDefinition, index: int) -> list[dict[str, object]]:
    """Show the numerical evidence for structural C/t0 constraints."""
    rows: list[dict[str, object]] = []
    for constraint_index, constraint in enumerate(strategy.anchor.constraints):
        path = f"anchor.constraints[{constraint_index}]"
        if isinstance(constraint, RollingLowAnchorConstraint):
            start, end = index - constraint.lookback_days, index + 1 if constraint.include_anchor else index
            values = df[constraint.reference_field.capitalize()].iloc[max(0, start):end].to_numpy(dtype=float)
            low = float(np.min(values)) if start >= 0 and len(values) and np.isfinite(values).all() else float("nan")
            gain = (float(df["Close"].iloc[index]) - low) / low if np.isfinite(low) and low else float("nan")
            rows.append(_audit_row(
                section="C 点结构约束", dsl_path=path, left=f"C 点相对前 {constraint.lookback_days} 日最低{constraint.reference_field}涨幅", left_value=gain,
                operator="less_than_or_equal", right="允许上限", right_value=constraint.maximum_anchor_close_gain,
                passed=np.isfinite(gain) and gain <= constraint.maximum_anchor_close_gain,
            ))
        elif isinstance(constraint, AnchorIndicatorChangeConstraint):
            prior = index + constraint.comparison_offset_days
            values = _indicator_values(df, constraint.indicator)
            change = values[index] / values[prior] - 1 if prior >= 0 and np.isfinite(values[index]) and np.isfinite(values[prior]) and values[prior] else float("nan")
            rows.append(_audit_row(
                section="C 点结构约束", dsl_path=path, left=f"{_indicator_label(constraint.indicator)} 相对 t{constraint.comparison_offset_days:+d}变化", left_value=change,
                operator=constraint.operator, right="最小变化", right_value=constraint.minimum_relative_change,
                passed=np.isfinite(change) and _compare(change, constraint.minimum_relative_change, constraint.operator),
            ))
        elif isinstance(constraint, OrderedExtremaDrawdownConstraint):
            start = index - constraint.lookback_days + 1
            values = df["Close"].iloc[max(0, start):index + 1].to_numpy(dtype=float)
            drawdown = recovery = float("nan")
            if start >= 0 and len(values) == constraint.lookback_days and np.isfinite(values).all():
                peak_relative = _extremum_index(values, kind="maximum", tie_break=constraint.peak_tie_break)
                if peak_relative < len(values) - 1:
                    trough_relative = peak_relative + 1 + _extremum_index(values[peak_relative + 1:], kind="minimum", tie_break=constraint.trough_tie_break)
                    peak, trough = values[peak_relative], values[trough_relative]
                    drawdown = (peak - trough) / peak if peak else float("nan")
                    recovery = (values[-1] - trough) / trough if trough else float("nan")
            rows.append(_audit_row(section="C 点结构约束", dsl_path=f"{path}.drawdown", left="A→B 回撤", left_value=drawdown, operator="greater_than_or_equal", right="最小回撤", right_value=constraint.minimum_peak_to_trough_drawdown, passed=np.isfinite(drawdown) and drawdown >= constraint.minimum_peak_to_trough_drawdown))
            if constraint.maximum_anchor_recovery_from_trough is not None:
                rows.append(_audit_row(section="C 点结构约束", dsl_path=f"{path}.recovery", left="C 点相对 B 点反弹", left_value=recovery, operator="less_than_or_equal", right="最大反弹", right_value=constraint.maximum_anchor_recovery_from_trough, passed=np.isfinite(recovery) and recovery <= constraint.maximum_anchor_recovery_from_trough))
    return rows


def _state_values_at(df: pd.DataFrame, strategy: TimedStrategyDefinition, *, anchor_index: int, entry_index: int, entry_price: float, index: int) -> dict[str, bool]:
    """Replay only persistent-state transitions up to one known event day."""
    values = {state.state_id: False for state in strategy.persistent_states}
    for day in range(entry_index + 1, index + 1):
        for state in strategy.persistent_states:
            if values[state.state_id] or not _state_active(state, anchor_index=anchor_index, entry_index=entry_index, day=day):
                continue
            values[state.state_id] = _temporal_condition(df, state.activate_when, index=day, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price)
    return values


def _exit_rule_audit_rows(
    df: pd.DataFrame,
    rule: ExitRule,
    *,
    anchor_index: int,
    entry_index: int,
    entry_price: float,
    exit_index: int,
    states: dict[str, bool],
) -> list[dict[str, object]]:
    """Explain one exit rule at the actual exit date.

    The backtest selects the lowest-priority-number rule only after checking
    its activation window, persistent-state gate, and condition.  This helper
    writes those exact gates out for *every* rule, including rules which were
    not eligible that day.  It prevents the audit view from mistaking an
    unlisted rule for a rule that was silently ignored.
    """
    section = f"出场规则：{rule.rule_id}（优先级 {rule.priority}）"
    relative_day = exit_index - (entry_index if rule.relative_to == "entry" else anchor_index)
    active = _rule_active(rule, relative_day)
    rows = [
        _audit_row(
            section=section,
            dsl_path=f"exit_rules.{rule.rule_id}.active_days",
            left="相对持仓日",
            left_value=relative_day,
            operator="范围",
            right=f"{rule.active_days.start_offset_days} 至 {rule.active_days.end_offset_days if rule.active_days.end_offset_days is not None else '∞'}",
            right_value=None,
            passed=active,
        )
    ]
    state_gate = True
    if rule.requires_state is not None:
        state_value = states.get(rule.requires_state, False)
        state_gate = state_gate and state_value
        rows.append(_audit_row(
            section=section,
            dsl_path=f"exit_rules.{rule.rule_id}.requires_state",
            left=f"状态 {rule.requires_state}",
            left_value=int(state_value),
            operator="equal",
            right="已激活",
            right_value=1,
            passed=state_value,
        ))
    if rule.forbids_state is not None:
        state_value = states.get(rule.forbids_state, False)
        state_gate = state_gate and not state_value
        rows.append(_audit_row(
            section=section,
            dsl_path=f"exit_rules.{rule.rule_id}.forbids_state",
            left=f"状态 {rule.forbids_state}",
            left_value=int(state_value),
            operator="equal",
            right="未激活",
            right_value=0,
            passed=not state_value,
        ))

    if rule.condition is not None:
        rows.extend(_temporal_condition_checks(
            df,
            rule.condition,
            index=exit_index,
            anchor_index=anchor_index,
            entry_index=entry_index,
            entry_price=entry_price,
            section=section,
            path=f"exit_rules.{rule.rule_id}.condition",
        ))
        condition_passed = _temporal_condition(
            df,
            rule.condition,
            index=exit_index,
            anchor_index=anchor_index,
            entry_index=entry_index,
            entry_price=entry_price,
        )
    elif rule.kind == "forced_close":
        # A forced close has no technical trigger.  Its activation window and
        # state gate are the complete execution rule.
        condition_passed = True
        rows.append(_audit_row(
            section=section,
            dsl_path=f"exit_rules.{rule.rule_id}.condition",
            left="强制平仓规则",
            left_value=1,
            operator="equal",
            right="无额外技术条件",
            right_value=1,
            passed=True,
        ))
    else:
        # v0.3's daily engine rejects intraday triggers before this point.
        # Keep the audit honest rather than pretending it has calculated an
        # unsupported price trigger from daily OHLC data.
        condition_passed = False
        rows.append(_audit_row(
            section=section,
            dsl_path=f"exit_rules.{rule.rule_id}.condition",
            left="条件",
            left_value=0,
            operator="结果",
            right="当前日线引擎不可审计",
            right_value=1,
            passed=False,
        ))

    rule_passed = active and state_gate and condition_passed
    rows.append(_audit_row(
        section=section,
        dsl_path=f"exit_rules.{rule.rule_id}.result",
        left="规则最终结果",
        left_value=int(rule_passed),
        operator="结果",
        right="触发出场",
        right_value=1,
        passed=rule_passed,
    ))
    return rows


def build_trade_event_audit(
    df: pd.DataFrame,
    *,
    strategy: TimedStrategyDefinition,
    signal_date: str,
    entry_date: str,
    exit_date: str,
    reason: str,
    event: Literal["entry", "exit"],
    market_index: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build an evidence table for a clicked v0.3 entry or exit marker.

    It deliberately reuses the deterministic execution primitives rather than
    trusting an LLM summary or a chart-derived approximation.
    """
    if strategy.data_requirement == "daily_ohlcv_with_market_index":
        if market_index is None:
            raise UnsupportedStrategyFeature(
                "this strategy's event audit requires the configured market-index series",
                details={"data_requirement": strategy.data_requirement},
            )
        df = attach_market_index_context(df, market_index)
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    def locate(date: str) -> int:
        matches = df.index[dates == pd.Timestamp(date).normalize()]
        if not len(matches):
            raise ValueError(f"trade date is not present in source data: {date}")
        return int(matches[0])

    anchor_index, entry_index, exit_index = locate(signal_date), locate(entry_date), locate(exit_date)
    entry_price = _entry_price_at(df, strategy, entry_index)
    if event == "entry":
        rows = [{**row, "section": "C 点入场条件"} for row in _anchor_condition_checks(df, strategy.anchor.condition, anchor_index)]
        rows.extend(_anchor_constraint_checks(df, strategy, anchor_index))
        if strategy.entry.mode == "wait_until":
            assert strategy.entry.defer_when is not None and strategy.entry.resume_when is not None
            rows.extend(_temporal_condition_checks(df, strategy.entry.defer_when, index=anchor_index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price, section="延后买入判断（t0）", path="entry.defer_when"))
            if entry_index > anchor_index:
                rows.extend(_temporal_condition_checks(df, strategy.entry.resume_when, index=entry_index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price, section="实际入场确认", path="entry.resume_when"))
        elif strategy.entry.mode == "conditional":
            assert strategy.entry.branches is not None
            for branch_index, branch in enumerate(strategy.entry.branches):
                if anchor_index + branch.active_day.start_offset_days == entry_index:
                    rows.extend(_temporal_condition_checks(df, branch.condition, index=entry_index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price, section="实际入场确认", path=f"entry.branches[{branch_index}].condition"))
        execution_name = "开盘价" if strategy.entry.execution == "open" else "收盘价"
        return {"event": "entry", "event_date": entry_date, "summary": f"t0 为 {signal_date}；实际入场为 {entry_date}，按{execution_name}成交 {entry_price:.4f}。", "rows": rows}

    rule_id = reason.removeprefix("v03:")
    if rule_id == "sample_end_force_close":
        rows = [_audit_row(
            section="样本末尾处理",
            dsl_path="lifecycle_policy.sample_end_open_position",
            left="样本结束",
            left_value=1,
            operator="结果",
            right="按 lifecycle_policy 强制平仓",
            right_value=1,
            passed=True,
        )]
        return {"event": "exit", "event_date": exit_date, "summary": "该笔交易因样本结束按 lifecycle_policy 强制平仓，不是技术条件触发。", "rows": rows}
    winner = next((candidate for candidate in strategy.exit_rules if candidate.rule_id == rule_id), None)
    if winner is None:
        return {"event": "exit", "event_date": exit_date, "summary": f"未找到出场原因 {reason} 对应的 DSL 规则。", "rows": []}
    states = _state_values_at(df, strategy, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price, index=exit_index)
    rows: list[dict[str, object]] = []
    for rule in sorted(strategy.exit_rules, key=lambda candidate: candidate.priority):
        rows.extend(_exit_rule_audit_rows(
            df,
            rule,
            anchor_index=anchor_index,
            entry_index=entry_index,
            entry_price=entry_price,
            exit_index=exit_index,
            states=states,
        ))
    return {
        "event": "exit",
        "event_date": exit_date,
        "summary": (
            f"{exit_date} 因规则 {winner.rule_id}（优先级 {winner.priority}）按收盘价出场。"
            "下表同时核对全部出场规则；只有“规则最终结果”为通过的规则具备触发资格。"
        ),
        "rows": rows,
    }


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
            if drawdown < constraint.minimum_peak_to_trough_drawdown:
                return False, []
            if constraint.maximum_anchor_recovery_from_trough is not None and recovery > constraint.maximum_anchor_recovery_from_trough:
                return False, []
            ordered_constraint_count += 1
            suffix = "" if ordered_constraint_count == 1 else str(ordered_constraint_count)
            event_points.extend([
                {"event_id": f"ordered_extrema_{ordered_constraint_count}_peak", "label": f"A{suffix}：窗口最高收盘", "date": str(pd.Timestamp(df["Date"].iloc[start + peak_relative]).date()), "price": round(float(peak), 4)},
                {"event_id": f"ordered_extrema_{ordered_constraint_count}_trough", "label": f"B{suffix}：A 后最低收盘", "date": str(pd.Timestamp(df["Date"].iloc[start + trough_relative]).date()), "price": round(float(trough), 4)},
            ])
        elif isinstance(constraint, RollingLowAnchorConstraint):
            start = index - constraint.lookback_days
            end = index + 1 if constraint.include_anchor else index
            if start < 0 or end <= start:
                return False, []
            column = constraint.reference_field.capitalize()
            values = df[column].iloc[start:end].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                return False, []
            low_relative = _extremum_index(values, kind="minimum", tie_break=constraint.tie_break)
            low_index = start + low_relative
            if constraint.low_must_precede_anchor and low_index >= index:
                return False, []
            low, anchor_close = values[low_relative], float(df["Close"].iloc[index])
            if low <= 0 or (anchor_close - low) / low > constraint.maximum_anchor_close_gain:
                return False, []
            event_points.append({"event_id": "rolling_low", "label": f"L{constraint.lookback_days}：窗口最低{constraint.reference_field}", "date": str(pd.Timestamp(df["Date"].iloc[low_index]).date()), "price": round(float(low), 4)})
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


def _state_active(state, *, anchor_index: int, entry_index: int, day: int) -> bool:
    relative_day = day - (entry_index if state.relative_to == "entry" else anchor_index)
    return relative_day >= state.active_days.start_offset_days and (
        state.active_days.end_offset_days is None or relative_day <= state.active_days.end_offset_days
    )


def _entry_price_at(df: pd.DataFrame, strategy: TimedStrategyDefinition, index: int) -> float:
    return float(df["Open" if strategy.entry.execution == "open" else "Close"].iloc[index])


def _resolve_entry(df: pd.DataFrame, strategy: TimedStrategyDefinition, anchor_index: int) -> tuple[int, float] | None:
    """Resolve fixed or conditional E-day entry without guessing a branch."""
    if strategy.entry.mode == "fixed":
        assert strategy.entry.active_day is not None
        candidates = [(strategy.entry.active_day.start_offset_days, strategy.entry.condition)]
    elif strategy.entry.mode == "conditional":
        assert strategy.entry.branches is not None
        candidates = [(branch.active_day.start_offset_days, branch.condition) for branch in strategy.entry.branches]
    else:
        assert strategy.entry.defer_when is not None and strategy.entry.resume_when is not None
        anchor_price = _entry_price_at(df, strategy, anchor_index)
        if not _temporal_condition(df, strategy.entry.defer_when, index=anchor_index, anchor_index=anchor_index, entry_index=anchor_index, entry_price=anchor_price):
            return anchor_index, anchor_price
        for entry_index in range(anchor_index + 1, len(df)):
            entry_price = _entry_price_at(df, strategy, entry_index)
            if _temporal_condition(df, strategy.entry.resume_when, index=entry_index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price):
                return entry_index, entry_price
        return None
    for offset, condition in sorted(candidates, key=lambda item: item[0]):
        entry_index = anchor_index + offset
        if entry_index >= len(df):
            continue
        entry_price = _entry_price_at(df, strategy, entry_index)
        # An absent/non-positive daily Open or Close cannot be treated as a
        # filled order. For a fixed next-day entry this means the candidate is
        # abandoned, exactly as the formula source specifies; the scanner can
        # then evaluate a new anchor on later bars.
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        # An omitted fixed-entry condition means “enter whenever the anchor
        # qualifies”. It is not a permissive fallback for branch entries.
        if condition is None or _temporal_condition(df, condition, index=entry_index, anchor_index=anchor_index, entry_index=entry_index, entry_price=entry_price):
            return entry_index, entry_price
    return None


def run_v03_backtest(
    df: pd.DataFrame,
    *,
    strategy: TimedStrategyDefinition,
    capital: float,
    compound: bool,
    cost_bps: float = 0.0,
    market_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run v0.3 exactly at daily closes; never approximate its stateful rules."""
    if strategy.schema_version != "0.3" or strategy.lifecycle_policy is None:
        raise UnsupportedStrategyFeature("v0.3 daily engine requires a v0.3 strategy with lifecycle_policy")
    if strategy.data_requirement not in {"daily_ohlcv", "daily_ohlcv_with_market_index"}:
        raise UnsupportedStrategyFeature(
            "v0.3 daily engine does not support this strategy data requirement",
            details={"data_requirement": strategy.data_requirement},
        )
    if strategy.data_requirement == "daily_ohlcv_with_market_index":
        if market_index is None:
            raise UnsupportedStrategyFeature(
                "v0.3 daily engine needs the configured market-index series before it can execute this strategy",
                details={"data_requirement": strategy.data_requirement},
            )
        df = attach_market_index_context(df, market_index)
    elif market_index is not None:
        # Avoid attaching unrelated data to ordinary daily strategies.  This
        # keeps their calculation path and cached indicators unchanged.
        df = df.copy()
    if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df):
        raise UnsupportedStrategyFeature(
            "v0.3 daily engine requires daily OHLCV columns",
        )
    equity, trades, open_positions, index = capital, [], [], 0
    n = len(df)
    while index < n:
        if not _static_condition(df, strategy.anchor.condition, index):
            index += 1
            continue
        anchor_checks = _anchor_condition_checks(df, strategy.anchor.condition, index)
        # ``anchor_checks`` is an audit trail, not an additional eligibility
        # gate.  In particular, an accepted ``OR`` group deliberately has
        # false leaves whenever another branch is true.  Requiring every
        # flattened leaf to pass turned valid strategies into a group-wide
        # failure (and hid the real result behind an AssertionError).
        # Eligibility is authoritatively decided above by the recursively
        # evaluated condition tree, which preserves AND/OR semantics.
        constraints_hold, event_points = _anchor_constraints_hold(df, strategy, index)
        if not constraints_hold:
            index += 1
            continue
        resolved_entry = _resolve_entry(df, strategy, index)
        if resolved_entry is None:
            index += 1
            continue
        entry_index, entry_price = resolved_entry
        exit_index, reason = None, None
        state_values = {state.state_id: False for state in strategy.persistent_states}
        # A close entry cannot be exited on that same close.  A next-day open
        # entry, however, is already held throughout that bar, so a source
        # rule may legitimately sell at the same day's close.  Treating both
        # cases as ``entry_index + 1`` silently drops that open-to-close risk
        # window.
        first_exit_day = entry_index if strategy.entry.execution == "open" else entry_index + 1
        for day in range(first_exit_day, n):
            # State transitions happen before exits, so a state entered today
            # can intentionally activate an exit rule on the same close.
            for state in strategy.persistent_states:
                if state_values[state.state_id] or not _state_active(state, anchor_index=index, entry_index=entry_index, day=day):
                    continue
                state_values[state.state_id] = _temporal_condition(
                    df, state.activate_when, index=day, anchor_index=index,
                    entry_index=entry_index, entry_price=entry_price,
                )
            matching = [
                rule
                for rule in strategy.exit_rules
                if _rule_active(rule, day - (entry_index if rule.relative_to == "entry" else index))
                and (rule.requires_state is None or state_values[rule.requires_state])
                and (rule.forbids_state is None or not state_values[rule.forbids_state])
                and rule.condition is not None
                and _temporal_condition(df, rule.condition, index=day, anchor_index=index, entry_index=entry_index, entry_price=entry_price)
            ]
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
