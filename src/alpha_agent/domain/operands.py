"""Explicit operands preserve current, calculated, scalar, and scaled values."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, FiniteFloat
from alpha_agent.domain.indicators import DSLModel, IndicatorDefinition, MarketFieldName

class MarketFieldOperand(DSLModel): kind: Literal["market_field"]; field: MarketFieldName
class IndicatorOperand(DSLModel): kind: Literal["indicator"]; indicator: IndicatorDefinition
class LaggedIndicatorOperand(DSLModel):
    """An indicator at a prior bar, used by an anchor-day comparison.

    This is deliberately distinct from a timed `anchor_indicator`: an anchor
    condition is evaluated *on* t0, so a negative offset here means a prior
    market bar (for example MA20[t] > MA20[t-1]).
    """

    kind: Literal["lagged_indicator"]
    offset_days: int = Field(lt=0)
    indicator: IndicatorDefinition
class ScalarOperand(DSLModel): kind: Literal["scalar"]; value: FiniteFloat
SeriesOperand = Annotated[MarketFieldOperand | IndicatorOperand | LaggedIndicatorOperand, Field(discriminator="kind")]
class ScaledOperand(DSLModel):
    kind: Literal["scaled_operand"]
    operand: SeriesOperand
    multiplier: Annotated[FiniteFloat, Field(gt=0)]
Operand = Annotated[MarketFieldOperand | IndicatorOperand | LaggedIndicatorOperand | ScalarOperand | ScaledOperand, Field(discriminator="kind")]
