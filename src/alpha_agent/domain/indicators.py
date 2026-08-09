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
class RollingMaximum(DSLModel): indicator: Literal["rolling_max"]; field: PriceFieldName; window: PositiveInt
class RollingMeanVolume(DSLModel): indicator: Literal["rolling_mean"]; field: Literal["volume"]; window: PositiveInt


class ADX(DSLModel):
    """Wilder Average Directional Index calculated from daily H/L/C."""

    indicator: Literal["adx"]
    window: PositiveInt


class DMIADX(DSLModel):
    """Chinese-formula DMI ADX with explicit directional and smoothing windows."""

    indicator: Literal["dmi_adx"]
    directional_window: PositiveInt
    adx_window: PositiveInt


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


class MACDPercentLine(MACDLine):
    """Formula-language DIF normalised by the current close and expressed in percent.

    ``(EMA(close, fast) - EMA(close, slow)) / close * 100`` is not the same
    quantity as the usual absolute-price MACD fast line.  Keeping it as a
    separate indicator prevents percentage thresholds from being silently
    reinterpreted as price units. ``signal_window`` preserves the declared
    MACD tuple although it does not change the DIF calculation itself.
    """

    indicator: Literal["macd_percent_line"]


class MarketIndexMACDLine(DSLModel):
    """DIF line calculated from a named external market-index close series."""

    indicator: Literal["market_index_macd_line"]
    index_symbol: str = Field(min_length=1)
    field: Literal["close"] = "close"
    fast_window: PositiveInt
    slow_window: PositiveInt
    signal_window: PositiveInt

    @model_validator(mode="after")
    def fast_must_precede_slow(self) -> "MarketIndexMACDLine":
        if self.fast_window >= self.slow_window:
            raise ValueError("market-index MACD fast_window must be less than slow_window")
        return self


IndicatorDefinition = Annotated[SMA | EMA | RSI | RollingReturn | RollingMinimum | RollingMaximum | RollingMeanVolume | ADX | DMIADX | MACDLine | MACDSignal | MACDPercentLine | MarketIndexMACDLine, Field(discriminator="indicator")]
