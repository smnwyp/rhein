"""Explicit operands preserve current, calculated, scalar, and scaled values."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, FiniteFloat
from alpha_agent.domain.indicators import DSLModel, IndicatorDefinition, MarketFieldName

class MarketFieldOperand(DSLModel): kind: Literal["market_field"]; field: MarketFieldName
class IndicatorOperand(DSLModel): kind: Literal["indicator"]; indicator: IndicatorDefinition
class ScalarOperand(DSLModel): kind: Literal["scalar"]; value: FiniteFloat
SeriesOperand = Annotated[MarketFieldOperand | IndicatorOperand, Field(discriminator="kind")]
class ScaledOperand(DSLModel):
    kind: Literal["scaled_operand"]
    operand: SeriesOperand
    multiplier: Annotated[FiniteFloat, Field(gt=0)]
Operand = Annotated[MarketFieldOperand | IndicatorOperand | ScalarOperand | ScaledOperand, Field(discriminator="kind")]
