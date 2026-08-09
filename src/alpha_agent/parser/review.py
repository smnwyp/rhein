"""Deterministic, user-facing explanation of a validated strategy DSL."""
from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup, CrossCondition, RollingComparisonCountCondition, RollingCrossCountCondition
from alpha_agent.domain.indicators import ADX, DMIADX, EMA, MACDLine, MACDPercentLine, MACDPercentSignal, MACDSignal, MarketIndexMACDLine, RSI, RollingMaximum, RollingMeanVolume, RollingMinimum, RollingReturn, SMA
from alpha_agent.domain.operands import IndicatorOperand, LaggedIndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.sequence import (
    AnchorIndicatorOperand, AnchorMarketOperand, AnchorRunningMaximumOperand, AnchorRunningVolumeRankOperand, CandlestickPatternCondition,
    CurrentIndicatorOperand, CurrentMarketOperand, LaggedIndicatorOperand as TemporalLaggedIndicatorOperand, LaggedMarketOperand, EntryPriceOperand, ExitRule, OrderedExtremaDrawdownConstraint, RollingVolumeRankOperand, ScaledEntryPriceOperand, TemporalScaledOperand,
    EntryRunningMaximumOperand,
    TemporalComparisonCondition, TemporalCondition, TemporalConditionGroup, TemporalCrossCondition, TemporalOperand,
    TemporalScalarOperand, TimedStrategyDefinition,
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
    if isinstance(indicator, RollingMinimum): return f"{indicator.field} {indicator.window} 日滚动最低值"
    if isinstance(indicator, RollingMaximum): return f"{indicator.field} {indicator.window} 日滚动最高值"
    if isinstance(indicator, RollingMeanVolume): return f"成交量 MA({indicator.window})"
    if isinstance(indicator, ADX): return f"ADX({indicator.window})"
    if isinstance(indicator, DMIADX): return f"DMI ADX({indicator.directional_window},{indicator.adx_window})"
    if isinstance(indicator, MarketIndexMACDLine): return f"指数 {indicator.index_symbol} DIF({indicator.fast_window},{indicator.slow_window},{indicator.signal_window})"
    if isinstance(indicator, (MACDPercentLine, MACDPercentSignal)):
        windows = f"{indicator.fast_window},{indicator.slow_window}"
        if indicator.signal_window is not None:
            windows += f",{indicator.signal_window}"
        name = indicator.label or '归一化 EMA 差值'
        return f"{name}{' 信号线' if isinstance(indicator, MACDPercentSignal) else ''}({windows})"
    if isinstance(indicator, MACDSignal): return f"MACD 慢线({indicator.fast_window},{indicator.slow_window},{indicator.signal_window})"
    if isinstance(indicator, MACDLine): return f"MACD 快线({indicator.fast_window},{indicator.slow_window},{indicator.signal_window})"
    raise TypeError(f"unknown indicator: {type(indicator).__name__}")


def _operand(operand: Operand) -> str:
    if isinstance(operand, MarketFieldOperand): return {"open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价", "volume": "成交量"}[operand.field]
    if isinstance(operand, IndicatorOperand): return _indicator(operand.indicator)
    if isinstance(operand, LaggedIndicatorOperand): return f"{_indicator(operand.indicator)}（当日{operand.offset_days:+d}）"
    if isinstance(operand, ScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, ScaledOperand): return f"{_operand(operand.operand)} × {operand.multiplier:g}"
    raise TypeError(f"unknown operand: {type(operand).__name__}")


def _temporal_operand(operand: TemporalOperand) -> str:
    if isinstance(operand, CurrentMarketOperand): return _operand(MarketFieldOperand(kind="market_field", field=operand.field))
    if isinstance(operand, LaggedMarketOperand): return f"前 {abs(operand.offset_days)} 日{_operand(MarketFieldOperand(kind='market_field', field=operand.field))}"
    if isinstance(operand, CurrentIndicatorOperand): return _indicator(operand.indicator)
    if isinstance(operand, TemporalLaggedIndicatorOperand): return f"前 {abs(operand.offset_days)} 日{_indicator(operand.indicator)}"
    if isinstance(operand, TemporalScalarOperand): return f"{operand.value:g}"
    if isinstance(operand, AnchorMarketOperand): return f"{operand.anchor} 日{_operand(MarketFieldOperand(kind='market_field', field=operand.field))}"
    if isinstance(operand, EntryPriceOperand): return "入场价"
    if isinstance(operand, ScaledEntryPriceOperand): return f"入场价 × {operand.multiplier:g}"
    if isinstance(operand, TemporalScaledOperand): return f"{_temporal_operand(operand.operand)} × {operand.multiplier:g}"
    if isinstance(operand, AnchorIndicatorOperand): return f"{operand.anchor}{operand.offset_days:+d} 日{_indicator(operand.indicator)}"
    if isinstance(operand, AnchorRunningMaximumOperand): return f"自 t0 起至当日的最大{_operand(MarketFieldOperand(kind='market_field', field=operand.field))}（并列取最早）"
    if isinstance(operand, EntryRunningMaximumOperand): return f"自实际入场日起至当日的最大{_operand(MarketFieldOperand(kind='market_field', field=operand.field))}（并列取{('最晚' if operand.tie_break == 'latest' else '最早')}）"
    if isinstance(operand, AnchorRunningVolumeRankOperand): return "自 t0 起至当日成交量的并列排名"
    if isinstance(operand, RollingVolumeRankOperand): return f"最近 {operand.window} 日成交量的并列排名"
    raise TypeError(f"unknown temporal operand: {type(operand).__name__}")


_OPERATOR = {"greater_than": ">", "less_than": "<", "greater_than_or_equal": "≥", "less_than_or_equal": "≤", "equal": "=", "cross_above": "上穿", "cross_below": "下穿"}


def _condition(condition: Condition) -> str:
    if isinstance(condition, ComparisonCondition): return f"{_operand(condition.left)} {_OPERATOR[condition.operator]} {_operand(condition.right)}"
    if isinstance(condition, CrossCondition): return f"{_operand(condition.left)} {_OPERATOR[condition.operator]} {_operand(condition.right)}"
    if isinstance(condition, RollingComparisonCountCondition): return f"最近 {condition.lookback_days} 日中，{_condition(condition.comparison)} 至少成立 {condition.minimum_true_count} 次"
    if isinstance(condition, RollingCrossCountCondition): return f"最近 {condition.lookback_days} 日中，{_operand(condition.left)} {_OPERATOR[condition.operator]} {_operand(condition.right)} 至少出现 {condition.minimum_true_count} 次"
    if isinstance(condition, ConditionGroup):
        joiner = " 且 " if condition.operator == "and" else " 或 "
        return "(" + joiner.join(_condition(item) for item in condition.conditions) + ")"
    raise TypeError(f"unknown condition: {type(condition).__name__}")


def _temporal_condition(condition: TemporalCondition) -> str:
    if isinstance(condition, TemporalComparisonCondition): return f"{_temporal_operand(condition.left)} {_OPERATOR[condition.operator]} {_temporal_operand(condition.right)}"
    if isinstance(condition, TemporalCrossCondition): return f"{_temporal_operand(condition.left)} {_OPERATOR[condition.operator]} {_temporal_operand(condition.right)}"
    if isinstance(condition, CandlestickPatternCondition):
        if condition.pattern == "doji": return f"十字星：|收盘价 - 开盘价| / 开盘价 ≤ {condition.body_to_open_threshold:.2%}"
        return f"大阴线：收盘价 < 开盘价，且 (开盘价 - 收盘价) / 开盘价 > {condition.body_to_open_threshold:.2%}"
    if isinstance(condition, TemporalConditionGroup):
        joiner = " 且 " if condition.operator == "and" else " 或 "
        return "(" + joiner.join(_temporal_condition(item) for item in condition.conditions) + ")"
    raise TypeError(f"unknown temporal condition: {type(condition).__name__}")


def _range(start: int, end: int | None) -> str:
    return f"t0+{start}" if end == start else (f"t0+{start} 起" if end is None else f"t0+{start} 至 t0+{end}")


def _exit(rule: ExitRule) -> str:
    prefix = "实际入场日 E" if rule.relative_to == "entry" else "t0"
    start, end = rule.active_days.start_offset_days, rule.active_days.end_offset_days
    window = f"{prefix}+{start}" if end == start else (f"{prefix}+{start} 起" if end is None else f"{prefix}+{start} 至 {prefix}+{end}")
    gate = f"且状态“{rule.requires_state}”已进入" if rule.requires_state else (f"且状态“{rule.forbids_state}”尚未进入" if rule.forbids_state else "")
    if rule.kind == "close_condition": return f"{window}：若{_temporal_condition(rule.condition)}{gate}，按收盘价退出。"  # type: ignore[arg-type]
    if rule.kind == "intraday_price_trigger":
        trigger = rule.price_trigger
        return f"{window}：盘中价格 {_OPERATOR[trigger.operator]} 入场价 × {trigger.multiplier:g} 时退出（需要分时数据）。"  # type: ignore[union-attr]
    return f"{window}：按收盘价强制平仓。"


def _entry_execution_name(execution: str) -> str:
    return "开盘价" if execution == "open" else "收盘价"


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
            window = f"t0 前 {constraint.lookback_days} 日至 t0" if constraint.include_anchor else f"t0 前 {constraint.lookback_days} 日（不含 t0）"
            explanation = f"{window}的{field}（并列取{('最晚' if constraint.tie_break == 'latest' else '最早')}）必须早于 t0；t0 收盘相对该低点涨幅 ≤ {constraint.maximum_anchor_close_gain:.2%}。"
        elif constraint.kind == "anchor_indicator_change_constraint":
            comparator = "大于" if constraint.operator == "greater_than" else "不低于"
            explanation = f"t0 的 {_indicator(constraint.indicator)} 相对 t0{constraint.comparison_offset_days:+d} 的变化{comparator} {constraint.minimum_relative_change:.2%}。"
        else:
            assert isinstance(constraint, OrderedExtremaDrawdownConstraint)
            recovery = "" if constraint.maximum_anchor_recovery_from_trough is None else f"，且 t0 相对 B 的反弹 ≤ {constraint.maximum_anchor_recovery_from_trough:.2%}"
            explanation = f"在截至 t0 的 {constraint.lookback_days} 日收盘价窗口中，A 为{('最晚' if constraint.peak_tie_break == 'latest' else '最早')}最高收盘价，B 为 A 后{('最晚' if constraint.trough_tie_break == 'latest' else '最早')}最低收盘价；A→B 回撤 ≥ {constraint.minimum_peak_to_trough_drawdown:.2%}{recovery}。"
        items.append(ReviewItem("t0 附加约束", f"anchor.constraints[{index}]", explanation))
    if strategy.entry.mode == "fixed":
        assert strategy.entry.active_day is not None
        timing = _range(strategy.entry.active_day.start_offset_days, strategy.entry.active_day.end_offset_days)
        price = _entry_execution_name(strategy.entry.execution)
        explanation = f"{timing}：t0 条件全部成立后，按{price}入场。" if strategy.entry.condition is None else f"{timing}：若{_temporal_condition(strategy.entry.condition)}，按{price}入场。"
        items.append(ReviewItem("入场", "entry", explanation))
    elif strategy.entry.mode == "conditional":
        assert strategy.entry.branches is not None
        price = _entry_execution_name(strategy.entry.execution)
        branches = "；".join(f"t0+{branch.active_day.start_offset_days}（{branch.branch_id}）：若{_temporal_condition(branch.condition)}，按{price}入场" for branch in strategy.entry.branches)
        items.append(ReviewItem("条件入场", "entry.branches", branches + "。未命中任何分支则放弃该候选点。"))
    elif strategy.entry.mode == "wait_until":
        assert strategy.entry.defer_when is not None and strategy.entry.resume_when is not None
        items.append(ReviewItem("延后入场", "entry", f"若 t0 当日{_temporal_condition(strategy.entry.defer_when)}，则不入场；自下一交易日起，首次{_temporal_condition(strategy.entry.resume_when)}时按{_entry_execution_name(strategy.entry.execution)}入场。"))
    else:
        assert strategy.entry.observation_day is not None and strategy.entry.confirmed_entry_day is not None and strategy.entry.wait_when is not None
        observation = _range(strategy.entry.observation_day.start_offset_days, strategy.entry.observation_day.end_offset_days)
        confirmed = _range(strategy.entry.confirmed_entry_day.start_offset_days, strategy.entry.confirmed_entry_day.end_offset_days)
        items.append(ReviewItem("次日观察与确认入场", "entry", f"{observation}：若{_temporal_condition(strategy.entry.wait_when)}，则观望；该日收盘重新满足完整 t0 条件时，{confirmed}按开盘价入场。若未触发观望，则 {observation} 按开盘价入场；重新确认失败即放弃该候选点。"))
    for index, state in enumerate(strategy.persistent_states):
        prefix = "实际入场日 E" if state.relative_to == "entry" else "t0"
        window = _range(state.active_days.start_offset_days, state.active_days.end_offset_days).replace("t0", prefix)
        items.append(ReviewItem(f"持久状态：{state.state_id}", f"persistent_states[{index}]", f"{window}：当{_temporal_condition(state.activate_when)}时进入；进入后持续有效，直至平仓。"))
    for index, rule in sorted(enumerate(strategy.exit_rules), key=lambda item: item[1].priority):
        items.append(ReviewItem(f"出场规则（优先级 {rule.priority}）", f"exit_rules[{index}]", _exit(rule)))
    if strategy.lifecycle_policy is not None:
        policy = strategy.lifecycle_policy
        sample_end = "样本最后一个交易日按收盘价强制平仓" if policy.sample_end_open_position == "force_close" else "样本结束未平仓头寸不计入已平仓交易"
        reentry = "卖出后允许继续寻找下一次 t0" if policy.allow_reentry_after_exit else "卖出后不再寻找下一次 t0"
        items.append(ReviewItem("完整样本执行政策", "lifecycle_policy", f"{sample_end}；{reentry}。"))
    return items
