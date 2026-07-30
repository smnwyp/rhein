"""Supported indicator definitions."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

PriceField = Literal["open", "high", "low", "close"]


class DSLModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SMA(DSLModel):
    type: Literal["sma"]
    source: PriceField
    window: PositiveInt


class EMA(DSLModel):
    type: Literal["ema"]
    source: PriceField
    window: PositiveInt


class RSI(DSLModel):
    type: Literal["rsi"]
    source: PriceField
    window: PositiveInt


class RollingReturn(DSLModel):
    type: Literal["rolling_return"]
    source: PriceField
    window: PositiveInt


class RollingMeanVolume(DSLModel):
    type: Literal["rolling_mean_volume"]
    window: PositiveInt


Indicator = Annotated[SMA | EMA | RSI | RollingReturn | RollingMeanVolume, Field(discriminator="type")]
