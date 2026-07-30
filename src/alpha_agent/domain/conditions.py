"""Typed operands and recursive condition trees."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PositiveFloat

from alpha_agent.domain.indicators import DSLModel, Indicator


class MarketField(DSLModel):
    kind: Literal["market_field"]
    field: Literal["open", "high", "low", "close", "volume"]


class IndicatorValue(DSLModel):
    kind: Literal["indicator"]
    indicator: Indicator


class Scalar(DSLModel):
    kind: Literal["scalar"]
    value: float


class ScaledSeries(DSLModel):
    kind: Literal["scaled_series"]
    series: TimeSeriesOperand
    factor: PositiveFloat


TimeSeriesOperand = Annotated[MarketField | IndicatorValue, Field(discriminator="kind")]
ComparableOperand = Annotated[MarketField | IndicatorValue | Scalar | ScaledSeries, Field(discriminator="kind")]


class Comparison(DSLModel):
    kind: Literal["comparison"]
    operator: Literal["greater_than", "less_than"]
    left: ComparableOperand
    right: ComparableOperand


class Cross(DSLModel):
    kind: Literal["cross"]
    operator: Literal["cross_above", "cross_below"]
    left: TimeSeriesOperand
    right: TimeSeriesOperand


class ConditionGroup(DSLModel):
    kind: Literal["group"]
    operator: Literal["and", "or"]
    conditions: list[Condition] = Field(min_length=1)


Condition = Annotated[Comparison | Cross | ConditionGroup, Field(discriminator="kind")]
ConditionGroup.model_rebuild()
ScaledSeries.model_rebuild()
