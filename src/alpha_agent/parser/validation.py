"""Deterministic domain semantics, separate from schema parsing."""
from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup
from alpha_agent.domain.indicators import RSI, RollingMeanVolume, RollingReturn
from alpha_agent.domain.operands import IndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
from alpha_agent.domain.strategy import StrategyDefinition
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
        if len(c.conditions) < 2: issues.append(_issue(path, "minimum_group_arity", "condition groups require at least two conditions"))
        for i, child in enumerate(c.conditions): _walk(child, f"{path}.conditions[{i}]", issues)
    elif isinstance(c, ComparisonCondition):
        l, r = _dimension(c.left), _dimension(c.right)
        if "scalar" not in (l, r) and l != r: issues.append(_issue(path, "compatible_operands", f"cannot compare {l} with {r}"))
        for scalar, other in ((c.left, c.right), (c.right, c.left)):
            if isinstance(scalar, ScalarOperand) and isinstance(other, IndicatorOperand) and isinstance(other.indicator, RSI) and not 0 <= scalar.value <= 100: issues.append(_issue(path, "rsi_threshold_range", "RSI thresholds must be between 0 and 100"))
def validate_strategy(strategy: StrategyDefinition) -> None:
    issues: list[dict[str, str]] = []; _walk(strategy.entry_condition, "entry_condition", issues); _walk(strategy.exit_condition, "exit_condition", issues)
    if issues: raise SemanticStrategyValidationFailure(issues)
