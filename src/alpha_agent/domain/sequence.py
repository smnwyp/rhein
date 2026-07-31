"""Explicit stateful, relative-day strategy semantics for DSL v0.2.

This remains a strategy *definition* only.  It deliberately does not simulate
orders or approximate intraday paths from daily candles.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, model_validator

from alpha_agent.domain.conditions import Condition
from alpha_agent.domain.indicators import DSLModel, IndicatorDefinition, MarketFieldName


ComparisonOperator = Literal["greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal"]


class RelativeDayRange(DSLModel):
    """An inclusive offset range from the named t0 anchor day."""

    anchor: Literal["t0"] = "t0"
    start_offset_days: int = Field(ge=0)
    end_offset_days: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "RelativeDayRange":
        if self.end_offset_days is not None and self.end_offset_days < self.start_offset_days:
            raise ValueError("end_offset_days must be at least start_offset_days")
        return self


class CurrentMarketOperand(DSLModel):
    kind: Literal["market_field"]
    field: MarketFieldName


class CurrentIndicatorOperand(DSLModel):
    kind: Literal["indicator"]
    indicator: IndicatorDefinition


class TemporalScalarOperand(DSLModel):
    kind: Literal["scalar"]
    value: FiniteFloat


class AnchorMarketOperand(DSLModel):
    kind: Literal["anchor_market_field"]
    anchor: Literal["t0"]
    field: MarketFieldName


class EntryPriceOperand(DSLModel):
    kind: Literal["entry_price"]


class AnchorIndicatorOperand(DSLModel):
    kind: Literal["anchor_indicator"]
    anchor: Literal["t0"]
    offset_days: int
    indicator: IndicatorDefinition


TemporalOperand = Annotated[
    CurrentMarketOperand | CurrentIndicatorOperand | TemporalScalarOperand | AnchorMarketOperand | EntryPriceOperand | AnchorIndicatorOperand,
    Field(discriminator="kind"),
]
TemporalSeriesOperand = Annotated[
    CurrentMarketOperand | CurrentIndicatorOperand | AnchorMarketOperand | AnchorIndicatorOperand,
    Field(discriminator="kind"),
]


class TemporalComparisonCondition(DSLModel):
    node_type: Literal["comparison"]
    operator: ComparisonOperator
    left: TemporalOperand
    right: TemporalOperand


class TemporalCrossCondition(DSLModel):
    node_type: Literal["cross"]
    operator: Literal["cross_above", "cross_below"]
    left: TemporalSeriesOperand
    right: TemporalSeriesOperand


class TemporalConditionGroup(DSLModel):
    node_type: Literal["group"]
    operator: Literal["and", "or"]
    conditions: list[TemporalCondition] = Field(min_length=1)


TemporalCondition = Annotated[
    TemporalComparisonCondition | TemporalCrossCondition | TemporalConditionGroup,
    Field(discriminator="node_type"),
]
TemporalConditionGroup.model_rebuild()


class RollingLowAnchorConstraint(DSLModel):
    """Anchor-day close relative to the lowest low from t0-lookback through t0."""

    kind: Literal["rolling_low_anchor_constraint"]
    lookback_days: int = Field(gt=0)
    reference_field: Literal["low", "close"] = "low"
    low_must_precede_anchor: bool = True
    maximum_anchor_close_gain: Annotated[FiniteFloat, Field(ge=0)]


class AnchorIndicatorChangeConstraint(DSLModel):
    """Compare an indicator at t0 with the same indicator on an earlier anchor offset."""

    kind: Literal["anchor_indicator_change_constraint"]
    indicator: IndicatorDefinition
    comparison_offset_days: int = Field(lt=0)
    minimum_relative_change: Annotated[FiniteFloat, Field(ge=0)]


AnchorConstraint = Annotated[
    RollingLowAnchorConstraint | AnchorIndicatorChangeConstraint,
    Field(discriminator="kind"),
]


class AnchorDefinition(DSLModel):
    name: Literal["t0"]
    condition: Condition
    constraints: list[AnchorConstraint] = Field(default_factory=list)


class EntryRule(DSLModel):
    active_day: RelativeDayRange
    condition: TemporalCondition
    execution: Literal["close"]

    @model_validator(mode="after")
    def entry_must_be_one_day(self) -> "EntryRule":
        if self.active_day.end_offset_days != self.active_day.start_offset_days:
            raise ValueError("entry active_day must name exactly one relative day")
        return self


class EntryPriceTrigger(DSLModel):
    base: Literal["entry_price"]
    multiplier: Annotated[FiniteFloat, Field(gt=0)]
    operator: Literal["less_than_or_equal", "greater_than_or_equal"]


class ExitRule(DSLModel):
    rule_id: str = Field(min_length=1)
    priority: int = Field(gt=0)
    active_days: RelativeDayRange
    kind: Literal["close_condition", "intraday_price_trigger", "forced_close"]
    condition: TemporalCondition | None = None
    price_trigger: EntryPriceTrigger | None = None
    execution: Literal["close", "intraday"]

    @model_validator(mode="after")
    def validate_kind_payload(self) -> "ExitRule":
        if self.kind == "close_condition" and (self.condition is None or self.price_trigger is not None or self.execution != "close"):
            raise ValueError("close_condition requires condition only and close execution")
        if self.kind == "intraday_price_trigger" and (self.price_trigger is None or self.condition is not None or self.execution != "intraday"):
            raise ValueError("intraday_price_trigger requires price_trigger only and intraday execution")
        if self.kind == "forced_close" and (self.condition is not None or self.price_trigger is not None or self.execution != "close"):
            raise ValueError("forced_close cannot have a trigger and requires close execution")
        return self


class TimedStrategyDefinition(DSLModel):
    schema_version: Literal["0.2"]
    strategy_name: str | None = Field(default=None, min_length=1)
    symbol: str = Field(min_length=1)
    frequency: Literal["1d"]
    direction: Literal["long_only"]
    position_mode: Literal["fully_invested_or_flat"]
    data_requirement: Literal["daily_ohlcv", "intraday_ohlcv"]
    anchor: AnchorDefinition
    entry: EntryRule
    exit_rules: list[ExitRule] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_exit_rules(self) -> "TimedStrategyDefinition":
        if len({rule.rule_id for rule in self.exit_rules}) != len(self.exit_rules):
            raise ValueError("exit rule IDs must be unique")
        if len({rule.priority for rule in self.exit_rules}) != len(self.exit_rules):
            raise ValueError("exit rule priorities must be unique")
        if any(rule.execution == "intraday" for rule in self.exit_rules) and self.data_requirement != "intraday_ohlcv":
            raise ValueError("intraday exit rules require intraday_ohlcv data")
        return self
