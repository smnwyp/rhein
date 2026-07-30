"""Deterministic business-rule validation of a valid condition schema."""

from alpha_agent.domain.conditions import Condition, ConditionGroup, IndicatorValue, MarketField, Scalar, ScaledSeries
from alpha_agent.domain.indicators import RSI, RollingMeanVolume, RollingReturn
from alpha_agent.domain.strategy import StrategyDefinition
from alpha_agent.errors import StrategyValidationError


def _dimension(operand: object) -> str:
    if isinstance(operand, Scalar):
        return "scalar"
    if isinstance(operand, ScaledSeries):
        return _dimension(operand.series)
    if isinstance(operand, MarketField):
        return "volume" if operand.field == "volume" else "price"
    if isinstance(operand, IndicatorValue):
        indicator = operand.indicator
        if isinstance(indicator, RollingMeanVolume):
            return "volume"
        if isinstance(indicator, (RSI, RollingReturn)):
            return "oscillator"
        return "price"
    raise TypeError("unknown operand")


def _validate_condition(condition: Condition, path: str, issues: list[str]) -> None:
    if isinstance(condition, ConditionGroup):
        if len(condition.conditions) < 2:
            issues.append(f"{path}: boolean groups require at least two conditions")
        for index, child in enumerate(condition.conditions):
            _validate_condition(child, f"{path}.conditions[{index}]", issues)
        return
    if condition.kind != "comparison":
        return
    left_dimension, right_dimension = _dimension(condition.left), _dimension(condition.right)
    if "scalar" not in (left_dimension, right_dimension) and left_dimension != right_dimension:
        issues.append(f"{path}: incompatible operands {left_dimension} and {right_dimension}")
    for operand in (condition.left, condition.right):
        if isinstance(operand, Scalar):
            other = condition.right if operand is condition.left else condition.left
            if isinstance(other, IndicatorValue) and isinstance(other.indicator, RSI) and not 0 <= operand.value <= 100:
                issues.append(f"{path}: RSI thresholds must be between 0 and 100")


def validate_strategy(strategy: StrategyDefinition) -> None:
    issues: list[str] = []
    _validate_condition(strategy.entry, "entry", issues)
    _validate_condition(strategy.exit, "exit", issues)
    if issues:
        raise StrategyValidationError(issues)
