"""Explicit stateful, relative-day strategy semantics for DSL v0.2/v0.3.

This remains a strategy *definition* only.  It deliberately does not simulate
orders or approximate intraday paths from daily candles.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, model_validator

from alpha_agent.domain.conditions import Condition
from alpha_agent.domain.indicators import DSLModel, IndicatorDefinition, MarketFieldName


ComparisonOperator = Literal["greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal", "equal"]


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


class ScaledEntryPriceOperand(DSLModel):
    """A fixed multiple of the entry close, e.g. entry price × 1.10."""

    kind: Literal["scaled_entry_price"]
    multiplier: Annotated[FiniteFloat, Field(gt=0)]


class AnchorIndicatorOperand(DSLModel):
    kind: Literal["anchor_indicator"]
    anchor: Literal["t0"]
    offset_days: int
    indicator: IndicatorDefinition


class AnchorRunningMaximumOperand(DSLModel):
    """Maximum value from the anchor day through the current relative day."""

    kind: Literal["anchor_running_maximum"]
    anchor: Literal["t0"]
    field: Literal["volume"]
    tie_break: Literal["earliest", "latest"] = "earliest"


class EntryRunningMaximumOperand(DSLModel):
    """Maximum value from the actual entry day through the current day."""

    kind: Literal["entry_running_maximum"]
    field: Literal["volume"]
    tie_break: Literal["earliest", "latest"] = "earliest"


TemporalOperand = Annotated[
    CurrentMarketOperand | CurrentIndicatorOperand | TemporalScalarOperand | AnchorMarketOperand | EntryPriceOperand | ScaledEntryPriceOperand | AnchorIndicatorOperand | AnchorRunningMaximumOperand | EntryRunningMaximumOperand,
    Field(discriminator="kind"),
]
TemporalSeriesOperand = Annotated[
    CurrentMarketOperand | CurrentIndicatorOperand | AnchorMarketOperand | AnchorIndicatorOperand | AnchorRunningMaximumOperand | EntryRunningMaximumOperand,
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


class CandlestickPatternCondition(DSLModel):
    """A daily pattern whose formula is explicit instead of generated code."""

    node_type: Literal["candlestick_pattern"]
    pattern: Literal["doji", "large_bearish"]
    body_to_open_threshold: Annotated[FiniteFloat, Field(gt=0, lt=1)]


class TemporalConditionGroup(DSLModel):
    node_type: Literal["group"]
    operator: Literal["and", "or"]
    conditions: list[TemporalCondition] = Field(min_length=1)


TemporalCondition = Annotated[
    TemporalComparisonCondition | TemporalCrossCondition | CandlestickPatternCondition | TemporalConditionGroup,
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
    include_anchor: bool = True
    tie_break: Literal["earliest", "latest"] = "earliest"


class AnchorIndicatorChangeConstraint(DSLModel):
    """Compare an indicator at t0 with the same indicator on an earlier anchor offset."""

    kind: Literal["anchor_indicator_change_constraint"]
    indicator: IndicatorDefinition
    comparison_offset_days: int = Field(lt=0)
    minimum_relative_change: Annotated[FiniteFloat, Field(ge=0)]
    operator: Literal["greater_than", "greater_than_or_equal"] = "greater_than_or_equal"


class OrderedExtremaDrawdownConstraint(DSLModel):
    """Close-high A followed by close-low B in a lookback ending on t0."""

    kind: Literal["ordered_extrema_drawdown_constraint"]
    lookback_days: int = Field(gt=1)
    field: Literal["close"]
    peak_tie_break: Literal["earliest", "latest"]
    trough_tie_break: Literal["earliest", "latest"]
    minimum_peak_to_trough_drawdown: Annotated[FiniteFloat, Field(gt=0, lt=1)]
    maximum_anchor_recovery_from_trough: Annotated[FiniteFloat, Field(ge=0, lt=1)] | None = None


AnchorConstraint = Annotated[
    RollingLowAnchorConstraint | AnchorIndicatorChangeConstraint | OrderedExtremaDrawdownConstraint,
    Field(discriminator="kind"),
]


class AnchorDefinition(DSLModel):
    name: Literal["t0"]
    condition: Condition
    constraints: list[AnchorConstraint] = Field(default_factory=list)


class EntryBranch(DSLModel):
    """One deterministic candidate for an actual entry day."""

    branch_id: str = Field(min_length=1)
    active_day: RelativeDayRange
    condition: TemporalCondition

    @model_validator(mode="after")
    def branch_must_be_one_day(self) -> "EntryBranch":
        if self.active_day.end_offset_days != self.active_day.start_offset_days:
            raise ValueError("entry branch active_day must name exactly one relative day")
        return self


class EntryRule(DSLModel):
    """Either one fixed entry or explicit conditional entry-day branches."""

    mode: Literal["fixed", "conditional"] = "fixed"
    active_day: RelativeDayRange | None = None
    condition: TemporalCondition | None = None
    execution: Literal["close"]
    branches: list[EntryBranch] | None = None

    @model_validator(mode="after")
    def entry_must_be_one_day(self) -> "EntryRule":
        if self.mode == "fixed":
            if self.active_day is None or self.condition is None or self.branches is not None:
                raise ValueError("fixed entry requires active_day and condition only")
            if self.active_day.end_offset_days != self.active_day.start_offset_days:
                raise ValueError("entry active_day must name exactly one relative day")
        else:
            if self.active_day is not None or self.condition is not None or not self.branches:
                raise ValueError("conditional entry requires one or more branches only")
            offsets = [branch.active_day.start_offset_days for branch in self.branches]
            if len(offsets) != len(set(offsets)):
                raise ValueError("conditional entry branches must use distinct relative days")
        return self


class EntryPriceTrigger(DSLModel):
    base: Literal["entry_price"]
    multiplier: Annotated[FiniteFloat, Field(gt=0)]
    operator: Literal["less_than_or_equal", "greater_than_or_equal"]


class ExitRule(DSLModel):
    rule_id: str = Field(min_length=1)
    priority: int = Field(gt=0)
    active_days: RelativeDayRange
    relative_to: Literal["anchor", "entry"] = "anchor"
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


class StrategyLifecyclePolicy(DSLModel):
    """Explicit choices required for deterministic full-sample execution."""

    sample_end_open_position: Literal["force_close", "leave_open_excluded"]
    allow_reentry_after_exit: bool


class TimedStrategyDefinition(DSLModel):
    schema_version: Literal["0.2", "0.3"]
    strategy_name: str | None = Field(default=None, min_length=1)
    symbol: str = Field(min_length=1)
    frequency: Literal["1d"]
    direction: Literal["long_only"]
    position_mode: Literal["fully_invested_or_flat"]
    data_requirement: Literal["daily_ohlcv", "intraday_ohlcv"]
    anchor: AnchorDefinition
    entry: EntryRule
    exit_rules: list[ExitRule] = Field(min_length=1)
    lifecycle_policy: StrategyLifecyclePolicy | None = None

    @model_validator(mode="after")
    def validate_exit_rules(self) -> "TimedStrategyDefinition":
        if len({rule.rule_id for rule in self.exit_rules}) != len(self.exit_rules):
            raise ValueError("exit rule IDs must be unique")
        for left_index, left_rule in enumerate(self.exit_rules):
            for right_rule in self.exit_rules[left_index + 1:]:
                if left_rule.priority != right_rule.priority:
                    continue
                left_end = left_rule.active_days.end_offset_days
                right_end = right_rule.active_days.end_offset_days
                left_ends_before_right = left_end is not None and left_end < right_rule.active_days.start_offset_days
                right_ends_before_left = right_end is not None and right_end < left_rule.active_days.start_offset_days
                if not left_ends_before_right and not right_ends_before_left:
                    raise ValueError("simultaneously active exit rules must have unique priorities")
        if any(rule.execution == "intraday" for rule in self.exit_rules) and self.data_requirement != "intraday_ohlcv":
            raise ValueError("intraday exit rules require intraday_ohlcv data")
        if self.schema_version == "0.3" and self.lifecycle_policy is None:
            raise ValueError("v0.3 strategies require an explicit lifecycle_policy")
        if self.schema_version == "0.3" and self.data_requirement != "daily_ohlcv":
            raise ValueError("v0.3 currently defines daily-close execution only")
        if self.schema_version == "0.3" and any(rule.kind == "intraday_price_trigger" for rule in self.exit_rules):
            raise ValueError("v0.3 does not permit intraday price triggers")
        return self
