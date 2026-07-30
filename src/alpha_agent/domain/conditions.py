"""Typed recursive condition trees."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field
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
class ConditionGroup(DSLModel):
    node_type: Literal["group"]
    operator: Literal["and", "or"]
    conditions: list[Condition] = Field(min_length=1)
Condition = Annotated[ComparisonCondition | CrossCondition | ConditionGroup, Field(discriminator="node_type")]
ConditionGroup.model_rebuild()
