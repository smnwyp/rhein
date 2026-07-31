"""Deterministic domain semantics, separate from schema parsing."""
from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup
from alpha_agent.domain.indicators import RSI, RollingMeanVolume, RollingReturn
from alpha_agent.domain.operands import IndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.sequence import (
    AnchorMarketOperand,
    CurrentMarketOperand,
    TemporalComparisonCondition,
    TemporalCondition,
    TemporalConditionGroup,
    TimedStrategyDefinition,
)
from alpha_agent.domain.strategy import AnyStrategyDefinition, StrategyDefinition
from alpha_agent.errors import SemanticStrategyValidationFailure
def _issue(path: str, rule: str, message: str) -> dict[str, str]: return {"path": path, "rule": rule, "message": message}
def _series_dimension(o: MarketFieldOperand | IndicatorOperand) -> str:
    if isinstance(o, MarketFieldOperand): return "volume" if o.field == "volume" else "price"
    if isinstance(o.indicator, RollingMeanVolume): return "volume"
    if isinstance(o.indicator, RSI): return "rsi"
    if isinstance(o.indicator, RollingReturn): return "return"
    return "price"
def _dimension(o: Operand) -> str:
    if isinstance(o, ScalarOperand): return "scalar"
    return _series_dimension(o.operand) if isinstance(o, ScaledOperand) else _series_dimension(o)
def _walk(c: Condition, path: str, issues: list[dict[str, str]]) -> None:
    if isinstance(c, ConditionGroup):
        for i, child in enumerate(c.conditions): _walk(child, f"{path}.conditions[{i}]", issues)
    elif isinstance(c, ComparisonCondition):
        l, r = _dimension(c.left), _dimension(c.right)
        if "scalar" not in (l, r) and l != r: issues.append(_issue(path, "compatible_operands", f"cannot compare {l} with {r}"))
        for scalar, other in ((c.left, c.right), (c.right, c.left)):
            if isinstance(scalar, ScalarOperand) and isinstance(other, IndicatorOperand) and isinstance(other.indicator, RSI) and not 0 <= scalar.value <= 100: issues.append(_issue(path, "rsi_threshold_range", "RSI thresholds must be between 0 and 100"))


def _walk_timed_condition(condition: TemporalCondition, path: str, issues: list[dict[str, str]], *, evaluated_on_anchor_day: bool) -> None:
    if isinstance(condition, TemporalConditionGroup):
        for index, child in enumerate(condition.conditions):
            _walk_timed_condition(child, f"{path}.conditions[{index}]", issues, evaluated_on_anchor_day=evaluated_on_anchor_day)
        return
    if not isinstance(condition, TemporalComparisonCondition) or not evaluated_on_anchor_day:
        return
    pairs = ((condition.left, condition.right), (condition.right, condition.left))
    for current, anchor in pairs:
        if not (isinstance(current, CurrentMarketOperand) and isinstance(anchor, AnchorMarketOperand) and current.field == anchor.field):
            continue
        if condition.operator in {"greater_than", "less_than"}:
            issues.append(_issue(
                path,
                "anchor_day_identity_comparison",
                f"{current.field}[t0] cannot be strictly {condition.operator} than itself on the anchor day",
            ))
        return
def validate_strategy(strategy: AnyStrategyDefinition) -> None:
    if isinstance(strategy, TimedStrategyDefinition):
        issues: list[dict[str, str]] = []
        _walk_timed_condition(
            strategy.entry.condition,
            "entry.condition",
            issues,
            evaluated_on_anchor_day=strategy.entry.active_day.start_offset_days == 0,
        )
        if issues:
            raise SemanticStrategyValidationFailure(issues)
        return
    issues: list[dict[str, str]] = []; _walk(strategy.entry_condition, "entry_condition", issues); _walk(strategy.exit_condition, "exit_condition", issues)
    if issues: raise SemanticStrategyValidationFailure(issues)
