"""Deterministic domain semantics, separate from schema parsing."""
from alpha_agent.domain.conditions import ComparisonCondition, Condition, ConditionGroup, RollingComparisonCountCondition
from alpha_agent.domain.indicators import ADX, RSI, RollingMaximum, RollingMeanVolume, RollingMinimum, RollingReturn
from alpha_agent.domain.operands import IndicatorOperand, LaggedIndicatorOperand, MarketFieldOperand, Operand, ScalarOperand, ScaledOperand
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
def _series_dimension(o: MarketFieldOperand | IndicatorOperand | LaggedIndicatorOperand) -> str:
    if isinstance(o, MarketFieldOperand): return "volume" if o.field == "volume" else "price"
    if isinstance(o.indicator, RollingMeanVolume): return "volume"
    if isinstance(o.indicator, RSI): return "rsi"
    if isinstance(o.indicator, RollingReturn): return "return"
    if isinstance(o.indicator, ADX): return "adx"
    return "price"
def _dimension(o: Operand) -> str:
    if isinstance(o, ScalarOperand): return "scalar"
    return _series_dimension(o.operand) if isinstance(o, ScaledOperand) else _series_dimension(o)
def _walk(c: Condition, path: str, issues: list[dict[str, str]]) -> None:
    if isinstance(c, ConditionGroup):
        for i, child in enumerate(c.conditions): _walk(child, f"{path}.conditions[{i}]", issues)
    elif isinstance(c, RollingComparisonCountCondition):
        _walk(c.comparison, f"{path}.comparison", issues)
    elif isinstance(c, ComparisonCondition):
        l, r = _dimension(c.left), _dimension(c.right)
        if "scalar" not in (l, r) and l != r: issues.append(_issue(path, "compatible_operands", f"cannot compare {l} with {r}"))
        for scalar, other in ((c.left, c.right), (c.right, c.left)):
            if isinstance(scalar, ScalarOperand) and isinstance(other, IndicatorOperand) and isinstance(other.indicator, RSI) and not 0 <= scalar.value <= 100: issues.append(_issue(path, "rsi_threshold_range", "RSI thresholds must be between 0 and 100"))
        # A rolling maximum/minimum is inclusive of the current bar.  Thus a
        # strict close > current rolling_max (or close < current rolling_min)
        # is mathematically impossible and silently turns every anchor false.
        # A "prior N days" breakout must use lagged_indicator(offset=-1).
        impossible_extremum_comparison = (
            (isinstance(c.left, MarketFieldOperand) and c.left.field == "close" and isinstance(c.right, IndicatorOperand) and isinstance(c.right.indicator, RollingMaximum) and c.operator == "greater_than")
            or (isinstance(c.right, MarketFieldOperand) and c.right.field == "close" and isinstance(c.left, IndicatorOperand) and isinstance(c.left.indicator, RollingMaximum) and c.operator == "less_than")
            or (isinstance(c.left, MarketFieldOperand) and c.left.field == "close" and isinstance(c.right, IndicatorOperand) and isinstance(c.right.indicator, RollingMinimum) and c.operator == "less_than")
            or (isinstance(c.right, MarketFieldOperand) and c.right.field == "close" and isinstance(c.left, IndicatorOperand) and isinstance(c.left.indicator, RollingMinimum) and c.operator == "greater_than")
        )
        if impossible_extremum_comparison:
            issues.append(_issue(
                path,
                "inclusive_rolling_extremum_strict_comparison",
                "a strict close comparison against a current inclusive rolling extremum is impossible; use a prior-bar lagged_indicator when the source says prior days",
            ))


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
        # Timed strategies have a static C/t0 anchor tree as well as temporal
        # entry/exit rules.  Validate both; otherwise a price-vs-return error
        # can be accepted and silently produce a zero-trade backtest.
        _walk(strategy.anchor.condition, "anchor.condition", issues)
        if strategy.entry.mode == "fixed":
            if strategy.entry.active_day is None:
                issues.append(_issue("entry.active_day", "fixed_entry_active_day", "fixed entry requires one active day"))
                raise SemanticStrategyValidationFailure(issues)
            if strategy.entry.condition is not None:
                _walk_timed_condition(
                    strategy.entry.condition,
                    "entry.condition",
                    issues,
                    evaluated_on_anchor_day=strategy.entry.active_day.start_offset_days == 0,
                )
        elif strategy.entry.mode == "conditional":
            if strategy.entry.branches is None:
                issues.append(_issue("entry.branches", "conditional_entry_branches", "conditional entry requires branches"))
                raise SemanticStrategyValidationFailure(issues)
            for index, branch in enumerate(strategy.entry.branches):
                _walk_timed_condition(
                    branch.condition,
                    f"entry.branches[{index}].condition",
                    issues,
                    evaluated_on_anchor_day=branch.active_day.start_offset_days == 0,
                )
        elif strategy.entry.mode == "wait_until":
            # The deferral predicate is evaluated on the anchor day.  The
            # resume predicate is deliberately evaluated on later candidate
            # days, so it must not be subjected to anchor-day identity checks.
            if strategy.entry.defer_when is None or strategy.entry.resume_when is None:
                issues.append(_issue("entry", "wait_until_predicates", "wait_until entry requires defer_when and resume_when"))
                raise SemanticStrategyValidationFailure(issues)
            _walk_timed_condition(strategy.entry.defer_when, "entry.defer_when", issues, evaluated_on_anchor_day=True)
            _walk_timed_condition(strategy.entry.resume_when, "entry.resume_when", issues, evaluated_on_anchor_day=False)
        else:
            issues.append(_issue("entry.mode", "supported_entry_mode", f"unsupported entry mode: {strategy.entry.mode}"))
        if issues:
            raise SemanticStrategyValidationFailure(issues)
        return
    issues: list[dict[str, str]] = []; _walk(strategy.entry_condition, "entry_condition", issues); _walk(strategy.exit_condition, "exit_condition", issues)
    if issues: raise SemanticStrategyValidationFailure(issues)
