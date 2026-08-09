"""Typed recursive condition trees."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, PositiveInt, model_validator
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.operands import Operand, SeriesOperand

class ComparisonCondition(DSLModel):
    node_type: Literal["comparison"]
    operator: Literal["greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal"]
    left: Operand
    right: Operand
class CrossCondition(DSLModel):
    node_type: Literal["cross"]
    operator: Literal["cross_above", "cross_below"]
    left: SeriesOperand
    right: SeriesOperand
class RollingComparisonCountCondition(DSLModel):
    """Count bars where one explicit persistent comparison holds in a window."""
    node_type: Literal["rolling_comparison_count"]
    lookback_days: PositiveInt
    minimum_true_count: PositiveInt
    comparison: ComparisonCondition

    @model_validator(mode="after")
    def count_fits_window(self) -> "RollingComparisonCountCondition":
        if self.minimum_true_count > self.lookback_days:
            raise ValueError("minimum_true_count cannot exceed lookback_days")
        return self


class RollingCrossCountCondition(DSLModel):
    """Count discrete cross events in a trailing window.

    This preserves formula expressions such as ``COUNT(CROSS(MA20, MA60), 5)
    >= 1`` without weakening a cross event into a persistent MA relationship.
    """

    node_type: Literal["rolling_cross_count"]
    lookback_days: PositiveInt
    minimum_true_count: PositiveInt
    operator: Literal["cross_above", "cross_below"]
    left: SeriesOperand
    right: SeriesOperand

    @model_validator(mode="after")
    def count_fits_window(self) -> "RollingCrossCountCondition":
        if self.minimum_true_count > self.lookback_days:
            raise ValueError("minimum_true_count cannot exceed lookback_days")
        return self
class ConditionGroup(DSLModel):
    node_type: Literal["group"]
    operator: Literal["and", "or"]
    conditions: list[Condition] = Field(min_length=1)
Condition = Annotated[ComparisonCondition | CrossCondition | RollingComparisonCountCondition | RollingCrossCountCondition | ConditionGroup, Field(discriminator="node_type")]
ConditionGroup.model_rebuild()
