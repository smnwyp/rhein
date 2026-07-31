"""Deterministic, user-facing explanation of a validated strategy DSL."""
from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup, CrossCondition
from alpha_agent.domain.indicators import EMA, RSI, RollingMeanVolume, RollingReturn, SMA
from alpha_agent.domain.operands import IndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorOperand, AnchorMarketOperand, CurrentIndicatorOperand, CurrentMarketOperand,
    EntryPriceOperand, ExitRule, TemporalComparisonCondition, TemporalCondition, TemporalConditionGroup,
    TemporalCrossCondition, TemporalOperand, TemporalScalarOperand, TimedStrategyDefinition,
)
from alpha_agent.domain.strategy import AnyStrategyDefinition, StrategyDefinition


@dataclass(frozen=True)
class ReviewItem:
    title: str
    dsl_path: str
    explanation: str


def _indicator(indicator: object) -> str:
    if isinstance(indicator, SMA): return f"收盘 SMA({indicator.window})"
    if isinstance(indicator, EMA): return f"收盘 EMA({indicator.window})"
    if isinstance(indicator, RSI): return f"RSI({indicator.window})"
    if isinstance(indicator, RollingReturn): return f"{indicator.window} 日滚动收益"
    if isinstance(indicator, RollingMeanVolume): return f"成交量 MA({indicator.window})"
    raise TypeError(f"unknown indicator: {type(indicator).__name__}")


def _operand(operand: Operand) -> str:
    if isinstance(operand, MarketFieldOperand): return {"open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价", "volume": "成交量"}[operand.field]
    if isinstance(operand, IndicatorOperand): return _indicator(operand.indicator)
    if isinstance(operand, ScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, ScaledOperand): return f"{_operand(operand.operand)} × {operand.multiplier:g}"
    raise TypeError(f"unknown operand: {type(operand).__name__}")


def _temporal_operand(operand: TemporalOperand) -> str:
    if isinstance(operand, CurrentMarketOperand): return _operand(MarketFieldOperand(kind="market_field", field=operand.field))
    if isinstance(operand, CurrentIndicatorOperand): return _indicator(operand.indicator)
    if isinstance(operand, TemporalScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, AnchorMarketOperand): return f"{operand.anchor} 日{_operand(MarketFieldOperand(kind='market_field', field=operand.field))}"
    if isinstance(operand, EntryPriceOperand): return "入场价"
    if isinstance(operand, AnchorIndicatorOperand): return f"{operand.anchor}{operand.offset_days:+d} 日{_indicator(operand.indicator)}"
    raise TypeError(f"unknown temporal operand: {type(operand).__name__}")


_OPERATOR = {"greater_than": ">", "less_than": "<", "greater_than_or_equal": "≥", "less_than_or_equal": "≤", "cross_above": "上穿", "cross_below": "下穿"}


def _condition(condition: Condition) -> str:
    if isinstance(condition, ComparisonCondition): return f"{_operand(condition.left)} {_OPERATOR[condition.operator]} {_operand(condition.right)}"
    if isinstance(condition, CrossCondition): return f"{_operand(condition.left)} {_OPERATOR[condition.operator]} {_operand(condition.right)}"
    if isinstance(condition, ConditionGroup):
        joiner = " 且 " if condition.operator == "and" else " 或 "
        return "(" + joiner.join(_condition(item) for item in condition.conditions) + ")"
    raise TypeError(f"unknown condition: {type(condition).__name__}")


def _temporal_condition(condition: TemporalCondition) -> str:
    if isinstance(condition, TemporalComparisonCondition): return f"{_temporal_operand(condition.left)} {_OPERATOR[condition.operator]} {_temporal_operand(condition.right)}"
    if isinstance(condition, TemporalCrossCondition): return f"{_temporal_operand(condition.left)} {_OPERATOR[condition.operator]} {_temporal_operand(condition.right)}"
    if isinstance(condition, TemporalConditionGroup):
        joiner = " 且 " if condition.operator == "and" else " 或 "
        return "(" + joiner.join(_temporal_condition(item) for item in condition.conditions) + ")"
    raise TypeError(f"unknown temporal condition: {type(condition).__name__}")


def _range(start: int, end: int | None) -> str:
    return f"t0+{start}" if end == start else (f"t0+{start} 起" if end is None else f"t0+{start} 至 t0+{end}")


def _exit(rule: ExitRule) -> str:
    window = _range(rule.active_days.start_offset_days, rule.active_days.end_offset_days)
    if rule.kind == "close_condition": return f"{window}：若{_temporal_condition(rule.condition)}，按收盘价退出。"  # type: ignore[arg-type]
    if rule.kind == "intraday_price_trigger":
        trigger = rule.price_trigger
        return f"{window}：盘中价格 {_OPERATOR[trigger.operator]} 入场价 × {trigger.multiplier:g} 时退出（需要分时数据）。"  # type: ignore[union-attr]
    return f"{window}：按收盘价强制平仓。"


def build_review_items(strategy: AnyStrategyDefinition) -> list[ReviewItem]:
    """Translate validated fields only; it never claims source-language equivalence."""
    payload = strategy.model_dump(mode="json")
    if payload.get("schema_version") == "0.1":
        strategy = StrategyDefinition.model_validate(payload)
        return [
            ReviewItem("执行频率", "frequency", "日线（1d）：每个条件以一个交易日的 OHLCV 数据计算。"),
            ReviewItem("入场条件", "entry_condition", _condition(strategy.entry_condition)),
            ReviewItem("出场条件", "exit_condition", _condition(strategy.exit_condition)),
        ]
    strategy = TimedStrategyDefinition.model_validate(payload)
    items = [
        ReviewItem("执行频率", "frequency", "日线（1d）：t0、t1、t2… 均为相对 t0 的交易日。"),
        ReviewItem("基准点 t0", "anchor.condition", _condition(strategy.anchor.condition)),
    ]
    for index, constraint in enumerate(strategy.anchor.constraints):
        if constraint.kind == "rolling_low_anchor_constraint":
            field = "最低价（Low）" if constraint.reference_field == "low" else "最低收盘价（Close）"
            explanation = f"t0 前 {constraint.lookback_days} 日至 t0 的{field}必须早于 t0；t0 收盘相对该低点涨幅 ≤ {constraint.maximum_anchor_close_gain:.2%}。"
        else:
            explanation = f"t0 的 {_indicator(constraint.indicator)} 相对 t0{constraint.comparison_offset_days:+d} 至少变化 {constraint.minimum_relative_change:.2%}。"
        items.append(ReviewItem("t0 附加约束", f"anchor.constraints[{index}]", explanation))
    items.append(ReviewItem("入场", "entry", f"{_range(strategy.entry.active_day.start_offset_days, strategy.entry.active_day.end_offset_days)}：若{_temporal_condition(strategy.entry.condition)}，按收盘价入场。"))
    for index, rule in sorted(enumerate(strategy.exit_rules), key=lambda item: item[1].priority):
        items.append(ReviewItem(f"出场规则（优先级 {rule.priority}）", f"exit_rules[{index}]", _exit(rule)))
    return items
