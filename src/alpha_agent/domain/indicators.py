"""Indicator definitions for Strategy DSL v0.1."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

class DSLModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

MarketFieldName = Literal["open", "high", "low", "close", "volume"]
PriceFieldName = Literal["open", "high", "low", "close"]
class SMA(DSLModel): indicator: Literal["sma"]; field: PriceFieldName; window: PositiveInt
class EMA(DSLModel): indicator: Literal["ema"]; field: PriceFieldName; window: PositiveInt
class RSI(DSLModel): indicator: Literal["rsi"]; field: PriceFieldName; window: PositiveInt
class RollingReturn(DSLModel): indicator: Literal["rolling_return"]; field: PriceFieldName; window: PositiveInt
class RollingMinimum(DSLModel): indicator: Literal["rolling_min"]; field: PriceFieldName; window: PositiveInt
class RollingMeanVolume(DSLModel): indicator: Literal["rolling_mean"]; field: Literal["volume"]; window: PositiveInt


class MACDLine(DSLModel):
    """MACD fast line: EMA(fast) - EMA(slow)."""

    indicator: Literal["macd_line"]
    field: Literal["close"] = "close"
    fast_window: PositiveInt
    slow_window: PositiveInt
    signal_window: PositiveInt

    @model_validator(mode="after")
    def fast_must_precede_slow(self) -> "MACDLine":
        if self.fast_window >= self.slow_window:
            raise ValueError("MACD fast_window must be less than slow_window")
        return self


class MACDSignal(MACDLine):
    """EMA(signal_window) of the MACD fast line."""

    indicator: Literal["macd_signal"]


IndicatorDefinition = Annotated[SMA | EMA | RSI | RollingReturn | RollingMinimum | RollingMeanVolume | MACDLine | MACDSignal, Field(discriminator="indicator")]
