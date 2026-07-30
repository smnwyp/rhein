"""Indicator definitions for Strategy DSL v0.1."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

class DSLModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

MarketFieldName = Literal["open", "high", "low", "close", "volume"]
PriceFieldName = Literal["open", "high", "low", "close"]
class SMA(DSLModel): indicator: Literal["sma"]; field: PriceFieldName; window: PositiveInt
class EMA(DSLModel): indicator: Literal["ema"]; field: PriceFieldName; window: PositiveInt
class RSI(DSLModel): indicator: Literal["rsi"]; field: PriceFieldName; window: PositiveInt
class RollingReturn(DSLModel): indicator: Literal["rolling_return"]; field: PriceFieldName; window: PositiveInt
class RollingMeanVolume(DSLModel): indicator: Literal["rolling_mean"]; field: Literal["volume"]; window: PositiveInt
IndicatorDefinition = Annotated[SMA | EMA | RSI | RollingReturn | RollingMeanVolume, Field(discriminator="indicator")]
