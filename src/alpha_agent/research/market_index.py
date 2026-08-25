"""Deterministic loading and validation of a shared market-index time series.

The index is data, not an inference made by the interpreter or backtest.  A
strategy that declares ``daily_ohlcv_with_market_index`` must receive this
series explicitly; otherwise the engine refuses to run rather than silently
removing the market gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from alpha_agent.errors import AlphaAgentError
from rhein.data.ohlc import load_ohlc_dates


class MarketIndexDataError(AlphaAgentError):
    """The configured external index could not be loaded safely."""

    code = "market_index_data_unavailable"


_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _cache_path(cache_directory: Path, symbol: str) -> Path:
    # Symbols such as ^IXIC are not portable filenames on every platform.
    return cache_directory / f"{symbol.replace('^', 'INDEX_').replace('/', '_')}.csv"


def required_market_index_symbols(strategy_payload: object) -> set[str]:
    """Find declared index symbols without guessing from natural language."""
    symbols: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("indicator") == "market_index_macd_line" and isinstance(value.get("index_symbol"), str):
                symbols.add(value["index_symbol"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(strategy_payload)
    return symbols


def _normalise_index_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance returns one Ticker level even for a single symbol.  The
        # first level is the OHLC field name we need.
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    if "Date" not in frame.columns:
        frame = frame.reset_index()
    missing = set(_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise MarketIndexDataError(
            "configured market-index data is missing required OHLCV columns",
            details={"source": source, "missing_columns": sorted(missing)},
        )
    result = frame.loc[:, _REQUIRED_COLUMNS].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    for column in _REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Date", "Close"]).drop_duplicates("Date", keep="last").sort_values("Date")
    if result.empty:
        raise MarketIndexDataError(
            "configured market-index data contains no usable trading days",
            details={"source": source},
        )
    # Market-index volume is not currently used by DSL v0.3, but make the
    # generic OHLCV contract stable even when a provider omits it.
    result["Volume"] = result["Volume"].fillna(0.0)
    return result.reset_index(drop=True)


def _covers(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
    return not dates.empty and dates.min() <= start and dates.max() >= end


def load_market_index(
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_directory: Path,
) -> pd.DataFrame:
    """Return an OHLCV index series spanning the requested inclusive dates.

    Cache reads are deterministic.  A network request happens only when the
    local cache cannot cover the requested period, and the downloaded result
    is persisted for later group runs.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        raise MarketIndexDataError("market-index date range is invalid", details={"start": str(start), "end": str(end)})
    path = _cache_path(cache_directory, symbol)
    if path.exists():
        try:
            cached = _normalise_index_frame(pd.read_csv(path), source=str(path))
            if _covers(cached, start, end):
                cached.attrs["_alpha_market_index_symbol"] = symbol
                return cached
        except (OSError, pd.errors.ParserError, MarketIndexDataError):
            # Re-fetch a malformed or stale cache rather than letting it
            # masquerade as a valid external series.
            pass
    try:
        import yfinance as yf

        downloaded = yf.download(
            symbol,
            start=start.date().isoformat(),
            # yfinance treats end as exclusive.  Calendar +1 handles a
            # Friday end date without needing a market-calendar dependency.
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
        )
        fresh = _normalise_index_frame(downloaded, source=f"yfinance:{symbol}")
    except MarketIndexDataError:
        raise
    except Exception as error:
        raise MarketIndexDataError(
            "could not download the configured market-index series",
            details={"symbol": symbol, "exception_type": type(error).__name__},
        ) from error
    if not _covers(fresh, start, end):
        raise MarketIndexDataError(
            "downloaded market-index series does not cover the selected group date range",
            details={
                "symbol": symbol,
                "requested_start": str(start.date()),
                "requested_end": str(end.date()),
                "available_start": str(fresh["Date"].min().date()),
                "available_end": str(fresh["Date"].max().date()),
            },
        )
    cache_directory.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, index=False)
    fresh.attrs["_alpha_market_index_symbol"] = symbol
    return fresh


def load_market_index_for_files(
    paths: Iterable[Path],
    *,
    symbol: str,
    cache_directory: Path,
) -> pd.DataFrame:
    """Load one index series which covers every asset file in a group."""
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for path in paths:
        try:
            dates = load_ohlc_dates(path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise MarketIndexDataError(
                "could not inspect the selected group's asset date range",
                details={"path": str(path), "exception_type": type(error).__name__},
            ) from error
        if not dates.empty:
            starts.append(dates.min().normalize())
            ends.append(dates.max().normalize())
    if not starts:
        raise MarketIndexDataError("selected group has no usable asset dates")
    return load_market_index(
        symbol=symbol,
        start=min(starts),
        end=max(ends),
        cache_directory=cache_directory,
    )
