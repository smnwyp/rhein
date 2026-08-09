"""Deterministic sector-emergence research for daily OHLCV data.

This module is intentionally outside ``alpha_agent``.  It consumes explicit
OHLCV files and an explicit industry map; it neither parses natural language
nor changes a saved strategy or its DSL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from rhein.data.ohlc import load_ohlc


AdjustmentPolicy = Literal["adjusted", "unknown_conservative_drop_outliers"]


class TrendTrackingError(ValueError):
    """A diagnosed input or configuration error for the trend workflow."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class TrendTrackingRun:
    """Immutable, machine-readable record of a deterministic screen run."""

    entry_threshold: float
    exit_threshold: float
    adjustment_policy: AdjustmentPolicy
    transaction_cost: float = 0.001
    portfolio_policy: str = "equal_weight_daily_rebalanced"


@dataclass(frozen=True)
class SectorBacktestResult:
    """Deterministic output for a threshold-locked sector backtest."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    market_benchmark: pd.DataFrame
    momentum_benchmark: pd.DataFrame
    kpis: dict[str, float | int | None]
    open_positions: pd.DataFrame


@dataclass(frozen=True)
class FixedHoldingBacktestResult:
    """Completed trades from a diagnostic that deliberately has no exit signal."""

    trades: pd.DataFrame
    candidate_count: int
    skipped_overlapping_candidates: int
    skipped_incomplete_candidates: int


@dataclass(frozen=True)
class CalibrationSegment:
    """Chronological front segment used by every calibration-only experiment."""

    start: pd.Timestamp
    end: pd.Timestamp
    validation_start: pd.Timestamp
    cleaned_daily: pd.DataFrame
    sector_daily: pd.DataFrame


_SYMBOL_COLUMNS = ("代码", "symbol", "Symbol", "ticker", "Ticker")
_SECTOR_COLUMNS = ("行业板块", "sector", "Sector", "industry", "Industry")
_REQUIRED_STOCK_COLUMNS = {"Date", "Symbol", "Open", "Close", "High", "Low", "Volume"}


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    found = next((name for name in candidates if name in frame.columns), None)
    if found is None:
        raise TrendTrackingError(
            "industry_map_missing_column",
            f"行业对照表缺少{label}列。",
            details={"accepted_columns": list(candidates), "actual_columns": list(frame.columns)},
        )
    return found


def normalize_industry_map(source: pd.DataFrame) -> pd.DataFrame:
    """Validate a dataframe supplied by a file, upload, or another adapter."""
    symbol_column = _resolve_column(source, _SYMBOL_COLUMNS, "代码")
    sector_column = _resolve_column(source, _SECTOR_COLUMNS, "行业板块")
    mapping = source.loc[:, [symbol_column, sector_column]].rename(
        columns={symbol_column: "Symbol", sector_column: "Sector"}
    ).copy()
    mapping["Symbol"] = mapping["Symbol"].astype(str).str.strip().str.upper()
    mapping["Sector"] = mapping["Sector"].astype(str).str.strip()
    mapping = mapping[(mapping["Symbol"] != "") & (mapping["Sector"] != "")]
    if mapping.empty:
        raise TrendTrackingError("industry_map_empty", "行业对照表没有有效的代码—板块映射。")
    conflicts = mapping.groupby("Symbol")["Sector"].nunique()
    conflicting_symbols = conflicts[conflicts > 1].index.tolist()
    if conflicting_symbols:
        raise TrendTrackingError(
            "industry_map_conflict", "同一代码映射到多个行业板块，无法确定性聚合。",
            details={"sample_symbols": conflicting_symbols[:10], "count": len(conflicting_symbols)},
        )
    return mapping.drop_duplicates("Symbol").sort_values("Symbol").reset_index(drop=True)


def load_industry_map(path: Path) -> pd.DataFrame:
    """Load and strictly validate a symbol-to-sector mapping CSV."""
    try:
        source = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise TrendTrackingError(
            "industry_map_read_failed", "无法读取行业对照表。", details={"path": str(path), "error": str(error)}
        ) from error
    return normalize_industry_map(source)


def load_stock_daily(paths: Iterable[Path]) -> pd.DataFrame:
    """Load repository OHLC files into the canonical stock-daily schema."""
    records: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            frame = load_ohlc(path).loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            frame.insert(1, "Symbol", path.stem.upper())
            records.append(frame)
        except Exception as error:  # report every source issue; do not silently skip it
            failures.append({"path": str(path), "error": str(error)})
    if failures:
        raise TrendTrackingError(
            "ohlcv_load_failed", "部分日线文件无法读取，未执行不完整的板块计算。", details={"failures": failures}
        )
    if not records:
        raise TrendTrackingError("ohlcv_empty", "没有可用于板块计算的日线文件。")
    return pd.concat(records, ignore_index=True)


def clean_stock_daily(
    stock_daily: pd.DataFrame,
    industry_map: pd.DataFrame,
    *,
    adjustment_policy: AdjustmentPolicy,
) -> pd.DataFrame:
    """Apply the PRD data guards and produce deterministic stock observations."""
    missing_stock = _REQUIRED_STOCK_COLUMNS - set(stock_daily.columns)
    missing_map = {"Symbol", "Sector"} - set(industry_map.columns)
    if missing_stock or missing_map:
        raise TrendTrackingError(
            "trend_input_schema_invalid",
            "日线或行业映射字段不完整。",
            details={"missing_stock_columns": sorted(missing_stock), "missing_map_columns": sorted(missing_map)},
        )
    if adjustment_policy not in {"adjusted", "unknown_conservative_drop_outliers"}:
        raise TrendTrackingError("adjustment_policy_invalid", "复权状态必须明确为已复权或未知保守模式。")
    frame = stock_daily.loc[:, ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.upper()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[frame[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1)]
    if not invalid.empty:
        raise TrendTrackingError(
            "ohlcv_invalid_values", "日线包含无法解析的日期或 OHLCV 数值。",
            details={"invalid_row_count": len(invalid), "sample_symbols": invalid["Symbol"].head(10).tolist()},
        )
    duplicate = frame.duplicated(["Date", "Symbol"], keep=False)
    if duplicate.any():
        raise TrendTrackingError(
            "ohlcv_duplicate_observation", "同一代码在同一交易日有多条日线记录。",
            details={"duplicate_count": int(duplicate.sum())},
        )
    mapping = industry_map.loc[:, ["Symbol", "Sector"]].copy()
    mapping["Symbol"] = mapping["Symbol"].astype(str).str.strip().str.upper()
    mapping["Sector"] = mapping["Sector"].astype(str).str.strip()
    if mapping.duplicated("Symbol", keep=False).any():
        raise TrendTrackingError("industry_map_not_unique", "行业对照表的代码必须唯一。")
    frame = frame.merge(mapping, on="Symbol", how="inner", validate="many_to_one")
    if frame.empty:
        raise TrendTrackingError("industry_map_no_overlap", "行业对照表与当前日线标的没有交集。")
    frame = frame.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    frame = frame[(frame["Volume"] > 0) & (frame["Close"] >= 1.0)].copy()
    frame["上市序号"] = frame.groupby("Symbol").cumcount()
    frame = frame[frame["上市序号"] >= 60].copy()
    frame["涨跌幅"] = frame.groupby("Symbol", sort=False)["Close"].pct_change()
    frame = frame.dropna(subset=["涨跌幅"])
    if adjustment_policy == "unknown_conservative_drop_outliers":
        frame = frame[frame["涨跌幅"].abs() <= 0.40].copy()
    if frame.empty:
        raise TrendTrackingError("cleaned_daily_empty", "应用清洗规则后没有可计算的日线记录。")
    return frame.sort_values(["Date", "Sector", "Symbol"]).reset_index(drop=True)


def aggregate_sector_daily(cleaned_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PRD-defined equal-weight sector metrics and market benchmark."""
    required = {"Date", "Symbol", "Sector", "Open", "Close", "Volume", "涨跌幅"}
    missing = required - set(cleaned_daily.columns)
    if missing:
        raise TrendTrackingError("cleaned_daily_schema_invalid", "清洗层输出缺少板块聚合所需字段。", details={"missing": sorted(missing)})
    frame = cleaned_daily.copy()
    frame["成交额"] = frame["Volume"] * frame["Close"]
    sector = (
        frame.groupby(["Date", "Sector"], as_index=False)
        .agg(
            板块涨跌幅=("涨跌幅", "mean"),
            上涨占比=("涨跌幅", lambda values: float((values > 0).mean())),
            板块成交额=("成交额", "sum"),
            个股数=("涨跌幅", "size"),
        )
    )
    sector = sector[sector["个股数"] >= 20].copy()
    if sector.empty:
        raise TrendTrackingError(
            "sector_sample_too_small", "没有板块在任一交易日达到至少 20 只股票的样本要求。",
            details={"minimum_constituents": 20},
        )
    market = frame.groupby("Date", as_index=False)["涨跌幅"].mean().rename(columns={"涨跌幅": "大盘涨跌幅"})
    sector = sector.merge(market, on="Date", how="inner", validate="many_to_one")
    sector["超额收益"] = sector["板块涨跌幅"] - sector["大盘涨跌幅"]
    return sector.sort_values(["Sector", "Date"]).reset_index(drop=True)


def _volume_score(ratio: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=ratio.index, dtype=float)
    score.loc[ratio <= 1] = 0.0
    middle = (ratio > 1) & (ratio <= 1.5)
    score.loc[middle] = (ratio.loc[middle] - 1) / 0.5 * 100
    score.loc[(ratio > 1.5) & (ratio <= 2)] = 100.0
    score.loc[ratio > 2] = 20.0
    return score


def score_sector(sector_daily: pd.DataFrame) -> pd.DataFrame:
    """Score one sector using only current and preceding observations."""
    sector = sector_daily.sort_values("Date").copy()
    sector["暴涨上限"] = sector["超额收益"].rolling(250, min_periods=120).quantile(0.98).shift(1)
    credit = pd.Series(np.nan, index=sector.index, dtype=float)
    usable = sector["暴涨上限"].notna()
    credit.loc[usable & (sector["超额收益"] > sector["暴涨上限"])] = -0.5
    credit.loc[usable & (sector["超额收益"] > 0) & (sector["超额收益"] <= sector["暴涨上限"])] = 1.0
    credit.loc[usable & (sector["超额收益"] <= 0)] = 0.0
    sector["单日记分"] = credit
    sector["稳赢计数"] = credit.rolling(20, min_periods=20).sum()
    sector["稳赢天数分"] = (sector["稳赢计数"] / 12).clip(0, 1) * 100
    recent_breadth = sector["上涨占比"].rolling(5, min_periods=5).mean()
    prior_breadth = recent_breadth.shift(5)
    sector["扩散差"] = recent_breadth - prior_breadth
    sector["广度扩散分"] = (sector["扩散差"] / 0.10).clip(0, 1) * 100
    recent_turnover = sector["板块成交额"].rolling(5, min_periods=5).mean()
    prior_turnover = recent_turnover.shift(5)
    sector["量能比"] = recent_turnover / prior_turnover.replace(0, np.nan)
    sector["量能爬坡分"] = _volume_score(sector["量能比"])
    sector["萌芽分"] = 0.4 * sector["稳赢天数分"] + 0.4 * sector["广度扩散分"] + 0.2 * sector["量能爬坡分"]
    return sector


def score_all_sectors(sector_daily: pd.DataFrame) -> pd.DataFrame:
    """Score all sectors, then rank each day without looking across dates."""
    required = {"Date", "Sector", "超额收益", "上涨占比", "板块成交额"}
    missing = required - set(sector_daily.columns)
    if missing:
        raise TrendTrackingError("sector_daily_schema_invalid", "板块日度表缺少打分字段。", details={"missing": sorted(missing)})
    scored_parts: list[pd.DataFrame] = []
    # A short explicit loop preserves the grouping column across pandas 2.x
    # and 3.x.  ``groupby.apply`` changes its inclusion/index behaviour across
    # those versions, which must never silently turn sector identities into
    # row indices in a research result.
    for sector_name, sector_frame in sector_daily.groupby("Sector", sort=False):
        part = score_sector(sector_frame).copy()
        part["Sector"] = sector_name
        scored_parts.append(part)
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["当日排名分位"] = scored.groupby("Date")["萌芽分"].rank(ascending=False, pct=True)
    return scored.sort_values(["Date", "Sector"]).reset_index(drop=True)


def generate_sector_signals(scored: pd.DataFrame, *, entry_threshold: float, exit_threshold: float) -> pd.DataFrame:
    """Generate causal entry/exit events and a per-sector holding state."""
    if not (np.isfinite(entry_threshold) and np.isfinite(exit_threshold) and exit_threshold < entry_threshold):
        raise TrendTrackingError("thresholds_invalid", "离场阈值必须是明确数值且严格低于进场阈值。")
    required = {"Date", "Sector", "萌芽分", "广度扩散分", "当日排名分位", "超额收益", "暴涨上限", "上涨占比"}
    missing = required - set(scored.columns)
    if missing:
        raise TrendTrackingError("scored_schema_invalid", "打分层输出缺少信号字段。", details={"missing": sorted(missing)})
    output: list[pd.DataFrame] = []
    for _, sector in scored.groupby("Sector", sort=False):
        item = sector.sort_values("Date").copy()
        qualified = (item["萌芽分"] >= entry_threshold) & (item["广度扩散分"] > 0) & (item["当日排名分位"] <= 0.30)
        item["进场条件成立"] = qualified
        item["连续三日进场条件"] = qualified & qualified.shift(1, fill_value=False) & qualified.shift(2, fill_value=False)
        breadth_zero = item["广度扩散分"] <= 0
        item["广度连续归零"] = breadth_zero.rolling(5, min_periods=5).sum().eq(5)
        # A pulse on t-1 is only known to be a pulse-and-breadth-decline
        # pattern at t, so the exit event is correctly stamped at t.
        item["脉冲后广度下降"] = (
            (item["超额收益"].shift(1) > item["暴涨上限"].shift(1))
            & (item["上涨占比"] < item["上涨占比"].shift(1))
        ).fillna(False)
        item["离场条件成立"] = (
            (item["萌芽分"] < exit_threshold) | item["广度连续归零"] | item["脉冲后广度下降"]
        )
        states: list[str] = []
        entries: list[bool] = []
        exits: list[bool] = []
        reasons: list[str] = []
        held = False
        for _, row in item.iterrows():
            entry = bool(row["连续三日进场条件"]) and not held
            exit_ = bool(row["离场条件成立"]) and held
            # An already-held sector exits when its mechanism is broken; a
            # same-day fresh entry cannot override an exit because it was not held.
            if exit_:
                state, held = "触发离场", False
            elif entry:
                state, held = "新触发进场", True
            elif held:
                state = "持有中"
            else:
                state = "无"
            reason_parts: list[str] = []
            if entry:
                reason_parts.append("连续三日萌芽分、广度扩散与排名条件成立")
            if exit_:
                if row["萌芽分"] < exit_threshold:
                    reason_parts.append("萌芽分跌破离场阈值")
                if bool(row["广度连续归零"]):
                    reason_parts.append("广度扩散分连续五日为零")
                if bool(row["脉冲后广度下降"]):
                    reason_parts.append("前日暴涨且今日广度下降")
            states.append(state)
            entries.append(entry)
            exits.append(exit_)
            reasons.append("；".join(reason_parts))
        item["进场"] = entries
        item["离场"] = exits
        item["状态"] = states
        item["信号原因"] = reasons
        output.append(item)
    return pd.concat(output, ignore_index=True).sort_values(["Date", "Sector"]).reset_index(drop=True)


def generate_entry_candidates(scored: pd.DataFrame, *, entry_threshold: float) -> pd.DataFrame:
    """Return the three-day entry condition without introducing an exit rule.

    This is intentionally diagnostic-only.  Fixed-holding experiments use the
    exact entry predicate from :func:`generate_sector_signals`, then apply a
    separately stated non-overlap rule while a mechanically timed position is
    open.  It must not borrow an arbitrary temporary exit threshold.
    """
    if not np.isfinite(entry_threshold):
        raise TrendTrackingError("entry_threshold_invalid", "进场阈值必须是明确数值。")
    required = {"Date", "Sector", "萌芽分", "广度扩散分", "当日排名分位"}
    missing = required - set(scored.columns)
    if missing:
        raise TrendTrackingError("scored_schema_invalid", "打分层输出缺少进场候选字段。", details={"missing": sorted(missing)})
    output: list[pd.DataFrame] = []
    for _, sector in scored.groupby("Sector", sort=False):
        item = sector.sort_values("Date").copy()
        qualified = (item["萌芽分"] >= entry_threshold) & (item["广度扩散分"] > 0) & (item["当日排名分位"] <= 0.30)
        item["进场条件成立"] = qualified
        item["连续三日进场条件"] = qualified & qualified.shift(1, fill_value=False) & qualified.shift(2, fill_value=False)
        item["进场"] = item["连续三日进场条件"]
        output.append(item)
    return pd.concat(output, ignore_index=True).sort_values(["Date", "Sector"]).reset_index(drop=True)


def assert_scoring_is_causal(sector_daily: pd.DataFrame, cutoff: pd.Timestamp | str) -> None:
    """Raise if scores/ranks on ``cutoff`` change when future rows are removed."""
    cutoff_timestamp = pd.Timestamp(cutoff).normalize()
    full = score_all_sectors(sector_daily)
    truncated = score_all_sectors(sector_daily[pd.to_datetime(sector_daily["Date"]).dt.normalize() <= cutoff_timestamp])
    columns = ["Sector", "Date", "萌芽分", "稳赢天数分", "广度扩散分", "量能爬坡分", "当日排名分位"]
    left = full[pd.to_datetime(full["Date"]).dt.normalize() == cutoff_timestamp].loc[:, columns].sort_values("Sector").reset_index(drop=True)
    right = truncated[pd.to_datetime(truncated["Date"]).dt.normalize() == cutoff_timestamp].loc[:, columns].sort_values("Sector").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as error:
        raise TrendTrackingError(
            "future_data_leak_detected", "截断测试失败：未来数据改变了历史打分。", details={"cutoff": str(cutoff_timestamp.date()), "difference": str(error)}
        ) from error


def latest_sector_table(signals: pd.DataFrame) -> pd.DataFrame:
    """Return the user-facing latest daily screen sorted by emergence score."""
    if signals.empty:
        return signals.copy()
    latest = pd.to_datetime(signals["Date"]).max()
    columns = ["Sector", "萌芽分", "稳赢天数分", "广度扩散分", "量能爬坡分", "当日排名分位", "状态", "信号原因", "个股数"]
    available = [column for column in columns if column in signals.columns]
    return (
        signals[pd.to_datetime(signals["Date"]) == latest]
        .loc[:, available]
        .sort_values("萌芽分", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _sector_execution_returns(cleaned_daily: pd.DataFrame) -> pd.DataFrame:
    """Build equal-weight sector returns needed for next-open execution."""
    frame = cleaned_daily.sort_values(["Symbol", "Date"]).copy()
    frame["前收盘"] = frame.groupby("Symbol", sort=False)["Close"].shift(1)
    frame["收盘到开盘收益"] = frame["Open"] / frame["前收盘"] - 1
    frame["开盘到收盘收益"] = frame["Close"] / frame["Open"] - 1
    frame["收盘到收盘收益"] = frame["Close"] / frame["前收盘"] - 1
    return (
        frame.groupby(["Date", "Sector"], as_index=False)
        .agg(
            收盘到开盘收益=("收盘到开盘收益", "mean"),
            开盘到收盘收益=("开盘到收盘收益", "mean"),
            收盘到收盘收益=("收盘到收盘收益", "mean"),
            个股数=("Symbol", "size"),
        )
        .sort_values(["Date", "Sector"])
        .reset_index(drop=True)
    )


def _next_sector_date(execution_returns: pd.DataFrame, sector: str, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    dates = execution_returns.loc[
        (execution_returns["Sector"] == sector) & (execution_returns["Date"] > signal_date), "Date"
    ]
    return pd.Timestamp(dates.iloc[0]) if not dates.empty else None


def _trade_return(
    cleaned_daily: pd.DataFrame,
    *,
    sector: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    transaction_cost: float,
) -> tuple[float, int] | None:
    start = cleaned_daily[(cleaned_daily["Sector"] == sector) & (cleaned_daily["Date"] == entry_date)][["Symbol", "Open"]]
    end = cleaned_daily[(cleaned_daily["Sector"] == sector) & (cleaned_daily["Date"] == exit_date)][["Symbol", "Open"]]
    components = start.merge(end, on="Symbol", suffixes=("_entry", "_exit"))
    components = components[(components["Open_entry"] > 0) & (components["Open_exit"] > 0)]
    if components.empty:
        return None
    gross_multiplier = float((components["Open_exit"] / components["Open_entry"]).mean())
    # A buy cost reduces the purchased exposure and a sell cost reduces the
    # liquidation proceeds.  Both are part of the deterministic PRD cost rule.
    net_return = gross_multiplier * (1 - transaction_cost) / (1 + transaction_cost) - 1
    return net_return, len(components)


def run_fixed_holding_backtest(
    cleaned_daily: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    *,
    holding_days: int,
    transaction_cost: float,
) -> FixedHoldingBacktestResult:
    """Test entry candidates with a mechanical holding period, never an exit signal.

    A trade enters on the signal's next available sector open and exits at the
    open exactly ``holding_days`` sector trading sessions later.  Candidates
    whose executable entry would overlap an open diagnostic position are
    deliberately ignored; this keeps the experiment at one equal-weight sector
    position rather than silently layering exposure.  Candidates without a
    complete fixed holding period before the calibration cutoff are reported as
    skipped rather than force-closed.
    """
    if not isinstance(holding_days, int) or holding_days <= 0:
        raise TrendTrackingError("fixed_holding_days_invalid", "固定持有期必须是正整数个交易日。")
    if not 0 <= transaction_cost < 1:
        raise TrendTrackingError("transaction_cost_invalid", "单边摩擦成本必须在 0（含）到 1（不含）之间。")
    required = {"Date", "Sector", "进场"}
    missing = required - set(entry_candidates.columns)
    if missing:
        raise TrendTrackingError("entry_candidates_schema_invalid", "固定持有期诊断缺少进场候选字段。", details={"missing": sorted(missing)})
    execution = _sector_execution_returns(cleaned_daily)
    if execution.empty:
        raise TrendTrackingError("execution_returns_empty", "没有可用于次日开盘模拟的板块日线。")
    candidates = entry_candidates.loc[entry_candidates["进场"].fillna(False)].copy()
    candidates["Date"] = pd.to_datetime(candidates["Date"]).dt.normalize()
    candidates["Sector"] = candidates["Sector"].astype(str)
    trades: list[dict[str, object]] = []
    skipped_overlap = 0
    skipped_incomplete = 0
    for sector, sector_candidates in candidates.groupby("Sector", sort=False):
        sector_dates = pd.Index(
            sorted(pd.to_datetime(execution.loc[execution["Sector"] == sector, "Date"]).unique())
        )
        if sector_dates.empty:
            skipped_incomplete += len(sector_candidates)
            continue
        latest_exit: pd.Timestamp | None = None
        for _, candidate in sector_candidates.sort_values("Date").iterrows():
            signal_date = pd.Timestamp(candidate["Date"])
            entry_after_signal = sector_dates[sector_dates > signal_date]
            if entry_after_signal.empty:
                skipped_incomplete += 1
                continue
            entry_date = pd.Timestamp(entry_after_signal[0])
            if latest_exit is not None and entry_date <= latest_exit:
                skipped_overlap += 1
                continue
            entry_index = int(sector_dates.get_loc(entry_date))
            exit_index = entry_index + holding_days
            if exit_index >= len(sector_dates):
                skipped_incomplete += 1
                continue
            exit_date = pd.Timestamp(sector_dates[exit_index])
            result = _trade_return(
                cleaned_daily,
                sector=sector,
                entry_date=entry_date,
                exit_date=exit_date,
                transaction_cost=transaction_cost,
            )
            if result is None:
                skipped_incomplete += 1
                continue
            net_return, component_count = result
            trades.append({
                "Sector": sector,
                "entry_signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_trading_days": holding_days,
                "component_count": component_count,
                "net_return": net_return,
                "net_return_pct": net_return * 100,
                "exit_policy": "fixed_holding",
            })
            latest_exit = exit_date
    return FixedHoldingBacktestResult(
        trades=pd.DataFrame(trades),
        candidate_count=int(len(candidates)),
        skipped_overlapping_candidates=skipped_overlap,
        skipped_incomplete_candidates=skipped_incomplete,
    )


def _curve_kpis(curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | None]:
    if curve.empty:
        return {"cumulative_return_pct": None, "annualized_return_pct": None, "max_drawdown_pct": None, "signal_count": 0, "win_rate_pct": None, "average_holding_days": None}
    equity = curve["equity"].astype(float)
    cumulative = (float(equity.iloc[-1]) - 1) * 100
    elapsed_days = max((pd.Timestamp(curve["Date"].iloc[-1]) - pd.Timestamp(curve["Date"].iloc[0])).days, 1)
    annualized = (float(equity.iloc[-1]) ** (365.25 / elapsed_days) - 1) * 100 if equity.iloc[-1] > 0 else None
    drawdown = equity / equity.cummax() - 1
    if trades.empty:
        win_rate, holding = None, None
    else:
        win_rate = float((trades["net_return"] > 0).mean() * 100)
        holding = float(trades["holding_trading_days"].mean())
    return {
        "cumulative_return_pct": cumulative,
        "annualized_return_pct": annualized,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "signal_count": int(len(trades)),
        "win_rate_pct": win_rate,
        "average_holding_days": holding,
    }


def _momentum_benchmark(execution_returns: pd.DataFrame, *, transaction_cost: float) -> pd.DataFrame:
    """Monthly top-20% relative-strength benchmark under the same cost rate."""
    dates = pd.Index(sorted(pd.to_datetime(execution_returns["Date"]).unique()))
    pivot = execution_returns.pivot(index="Date", columns="Sector", values="收盘到收盘收益").sort_index()
    open_close = execution_returns.pivot(index="Date", columns="Sector", values="开盘到收盘收益").sort_index()
    selected: set[str] = set()
    equity = 1.0
    rows: list[dict[str, object]] = []
    prior_month: tuple[int, int] | None = None
    for date in dates:
        month = (date.year, date.month)
        rebalance = month != prior_month
        if rebalance:
            history = pivot.loc[pivot.index < date].tail(20)
            if len(history) == 20:
                cumulative = (1 + history).prod(axis=0, min_count=20) - 1
                market_return = float((1 + history.mean(axis=1)).prod() - 1)
                strength = (cumulative - market_return).dropna().sort_values(ascending=False)
                count = max(1, int(np.ceil(len(strength) * 0.20))) if len(strength) else 0
                target = set(strength.head(count).index)
            else:
                target = set()
            # Turnover is the fraction of the equal-weight benchmark changed
            # at this rebalance.  It makes the benchmark comparable under the
            # same single-side cost assumption without claiming an index fill.
            union = selected | target
            turnover = len(selected ^ target) / len(union) if union else 0.0
            equity *= 1 - transaction_cost * turnover
            selected = target
            prior_month = month
        returns = open_close.loc[date, list(selected)] if rebalance and selected else (
            pivot.loc[date, list(selected)] if selected else pd.Series(dtype=float)
        )
        return_ = float(returns.dropna().mean()) if len(returns.dropna()) else 0.0
        equity *= 1 + return_
        rows.append({"Date": date, "equity": equity, "selected_sector_count": len(selected)})
    # A benchmark with a selected portfolio must include the same final
    # one-side liquidation cost as every completed strategy trade.
    if rows and selected:
        rows[-1]["equity"] = float(rows[-1]["equity"]) * (1 - transaction_cost)
    return pd.DataFrame(rows)


def _market_benchmark(cleaned_daily: pd.DataFrame, *, transaction_cost: float) -> pd.DataFrame:
    """All-cleaned-stock equal-weight market curve with an explicit round-trip cost."""
    frame = cleaned_daily.sort_values(["Symbol", "Date"]).copy()
    frame["前收盘"] = frame.groupby("Symbol", sort=False)["Close"].shift(1)
    frame["收盘到收盘收益"] = frame["Close"] / frame["前收盘"] - 1
    # This is deliberately an equal-weight mean over individual stocks, not an
    # equal-weight mean over sectors.  It matches the PRD's market definition.
    market_returns = frame.groupby("Date", as_index=False)["收盘到收盘收益"].mean().sort_values("Date")
    # The entry is an explicit opening transaction at the start of the segment;
    # a matching liquidation cost is deducted from the final observation below.
    equity = 1.0 * (1 - transaction_cost)
    rows: list[dict[str, object]] = []
    for _, row in market_returns.iterrows():
        raw_value = row["收盘到收盘收益"]
        value = float(raw_value) if pd.notna(raw_value) else 0.0
        equity *= 1 + (value if np.isfinite(value) else 0.0)
        rows.append({"Date": row["Date"], "equity": equity})
    if rows:
        rows[-1]["equity"] = float(rows[-1]["equity"]) * (1 - transaction_cost)
    return pd.DataFrame(rows)


def run_sector_backtest(
    cleaned_daily: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    run: TrendTrackingRun,
) -> SectorBacktestResult:
    """Simulate next-open sector execution and two PRD benchmarks deterministically.

    The portfolio policy is explicit in ``TrendTrackingRun``: holdings are
    equal-weighted at each open after signal orders, with no unmodelled
    leverage.  The trade list itself is calculated from constituent open prices
    at each entry/exit, never from a score approximation.
    """
    if not 0 <= run.transaction_cost < 1:
        raise TrendTrackingError("transaction_cost_invalid", "单边摩擦成本必须在 0（含）到 1（不含）之间。")
    execution = _sector_execution_returns(cleaned_daily)
    if execution.empty:
        raise TrendTrackingError("execution_returns_empty", "没有可用于次日开盘模拟的板块日线。")
    signal_frame = signals.copy()
    signal_frame["Date"] = pd.to_datetime(signal_frame["Date"]).dt.normalize()
    events: dict[pd.Timestamp, list[tuple[str, str, pd.Timestamp]]] = {}
    for _, row in signal_frame.loc[signal_frame["进场"] | signal_frame["离场"]].iterrows():
        signal_date, sector = pd.Timestamp(row["Date"]), str(row["Sector"])
        execution_date = _next_sector_date(execution, sector, signal_date)
        if execution_date is None:
            continue  # end-of-sample signal has no next-day executable price
        kind = "entry" if bool(row["进场"]) else "exit"
        events.setdefault(execution_date, []).append((kind, sector, signal_date))
    dates = pd.Index(sorted(pd.to_datetime(execution["Date"]).unique()))
    by_date = execution.set_index(["Date", "Sector"])
    held: set[str] = set()
    entry_details: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    trades: list[dict[str, object]] = []
    open_positions: list[dict[str, object]] = []
    equity = 1.0
    curve_rows: list[dict[str, object]] = []
    for date in dates:
        date_events = events.get(pd.Timestamp(date), [])
        exits = {sector: signal_date for kind, sector, signal_date in date_events if kind == "exit" and sector in held}
        entries = {sector: signal_date for kind, sector, signal_date in date_events if kind == "entry" and sector not in held}
        old_held = set(held)
        # Existing holdings first move from prior close to today's open.  A
        # requested exit realizes that gap and the one-side exit cost at open.
        if old_held:
            gap_returns: list[float] = []
            for sector in old_held:
                try:
                    gap = float(by_date.loc[(date, sector), "收盘到开盘收益"])
                except KeyError:
                    gap = np.nan
                if not np.isfinite(gap):
                    gap = 0.0
                if sector in exits:
                    gap = (1 + gap) * (1 - run.transaction_cost) - 1
                gap_returns.append(gap)
            equity *= 1 + float(np.mean(gap_returns))
        for sector, signal_date in exits.items():
            entry_date, entry_signal = entry_details.pop(sector)
            result = _trade_return(cleaned_daily, sector=sector, entry_date=entry_date, exit_date=pd.Timestamp(date), transaction_cost=run.transaction_cost)
            if result is not None:
                net_return, component_count = result
                sector_dates = execution.loc[(execution["Sector"] == sector) & (execution["Date"] >= entry_date) & (execution["Date"] <= date), "Date"]
                trades.append({
                    "Sector": sector, "entry_signal_date": entry_signal, "exit_signal_date": signal_date,
                    "entry_date": entry_date, "exit_date": pd.Timestamp(date), "holding_trading_days": max(len(sector_dates) - 1, 0),
                    "component_count": component_count, "net_return": net_return, "net_return_pct": net_return * 100,
                })
        held.difference_update(exits)
        held.update(entries)
        for sector, signal_date in entries.items():
            entry_details[sector] = (pd.Timestamp(date), signal_date)
        if held and entries:
            equity *= 1 - run.transaction_cost * len(entries) / len(held)
        # At the open, available capital is explicitly equal-weighted across
        # the active sector portfolios.  New holdings earn open-to-close;
        # carried holdings already had their gap return and now earn O→C.
        if held:
            intraday_returns: list[float] = []
            for sector in held:
                try:
                    value = float(by_date.loc[(date, sector), "开盘到收盘收益"])
                except KeyError:
                    value = np.nan
                intraday_returns.append(value if np.isfinite(value) else 0.0)
            equity *= 1 + float(np.mean(intraday_returns))
        curve_rows.append({"Date": pd.Timestamp(date), "equity": equity, "active_sector_count": len(held)})
    for sector, (entry_date, signal_date) in sorted(entry_details.items()):
        open_positions.append({"Sector": sector, "entry_signal_date": signal_date, "entry_date": entry_date, "status": "sample_end_open_excluded"})
    curve = pd.DataFrame(curve_rows)
    trade_frame = pd.DataFrame(trades)
    market_curve = _market_benchmark(cleaned_daily, transaction_cost=run.transaction_cost)
    momentum_curve = _momentum_benchmark(execution, transaction_cost=run.transaction_cost)
    return SectorBacktestResult(
        trades=trade_frame,
        equity_curve=curve,
        market_benchmark=market_curve,
        momentum_benchmark=momentum_curve,
        kpis=_curve_kpis(curve, trade_frame),
        open_positions=pd.DataFrame(open_positions),
    )


def _calibration_segment(
    cleaned_daily: pd.DataFrame,
    sector_daily: pd.DataFrame,
    *,
    calibration_ratio: float,
) -> CalibrationSegment:
    """Split chronological dates without calculating or inspecting validation values."""
    if not 0 < calibration_ratio < 1:
        raise TrendTrackingError("calibration_ratio_invalid", "标定比例必须严格介于 0 和 1 之间。")
    dates = pd.Index(sorted(pd.to_datetime(sector_daily["Date"]).unique()))
    split_at = int(len(dates) * calibration_ratio)
    if len(dates) < 2 or split_at < 1 or split_at >= len(dates):
        raise TrendTrackingError("calibration_history_too_short", "历史交易日不足，无法按 70/30 切分校准与检验段。")
    calibration_dates = dates[:split_at]
    return CalibrationSegment(
        start=pd.Timestamp(calibration_dates[0]),
        end=pd.Timestamp(calibration_dates[-1]),
        validation_start=pd.Timestamp(dates[split_at]),
        cleaned_daily=cleaned_daily[pd.to_datetime(cleaned_daily["Date"]).isin(calibration_dates)].copy(),
        sector_daily=sector_daily[pd.to_datetime(sector_daily["Date"]).isin(calibration_dates)].copy(),
    )


def fixed_holding_diagnostics(
    cleaned_daily: pd.DataFrame,
    sector_daily: pd.DataFrame,
    *,
    entry_thresholds: Iterable[float],
    holding_days: Iterable[int] = (20, 40, 60),
    transaction_cost: float = 0.001,
    calibration_ratio: float = 0.7,
) -> pd.DataFrame:
    """Compare 20/40/60-day forced exits on the front calibration segment only."""
    entry_values = sorted({float(value) for value in entry_thresholds})
    holding_values = sorted({int(value) for value in holding_days})
    if not entry_values or not holding_values:
        raise TrendTrackingError("fixed_holding_grid_required", "固定持有期诊断需要至少一个进场阈值和一个持有期。")
    if any(value <= 0 for value in holding_values):
        raise TrendTrackingError("fixed_holding_days_invalid", "固定持有期必须是正整数个交易日。")
    segment = _calibration_segment(cleaned_daily, sector_daily, calibration_ratio=calibration_ratio)
    assert_scoring_is_causal(segment.sector_daily, segment.end)
    scored = score_all_sectors(segment.sector_daily)
    rows: list[dict[str, object]] = []
    for entry_threshold in entry_values:
        candidates = generate_entry_candidates(scored, entry_threshold=entry_threshold)
        for fixed_days in holding_values:
            result = run_fixed_holding_backtest(
                segment.cleaned_daily,
                candidates,
                holding_days=fixed_days,
                transaction_cost=transaction_cost,
            )
            trades = result.trades
            if trades.empty:
                win_rate: float | None = None
                mean_return: float | None = None
                median_return: float | None = None
                average_holding: float | None = None
            else:
                win_rate = float((trades["net_return"] > 0).mean() * 100)
                mean_return = float(trades["net_return"].mean() * 100)
                median_return = float(trades["net_return"].median() * 100)
                average_holding = float(trades["holding_trading_days"].mean())
            rows.append({
                "进场阈值": entry_threshold,
                "固定持有期（交易日）": fixed_days,
                "校准段起始日": segment.start,
                "校准段截止日": segment.end,
                "检验段起始日（未使用）": segment.validation_start,
                "已平仓信号次数": len(trades),
                "胜率": win_rate,
                "平均单笔净收益": mean_return,
                "中位单笔净收益": median_return,
                "平均实际持有天数": average_holding,
                "三日进场候选数": result.candidate_count,
                "持有期内忽略的重叠候选数": result.skipped_overlapping_candidates,
                "样本末尾无法完成持有期的候选数": result.skipped_incomplete_candidates,
                "离场规则": "固定持有期（不使用评分离场）",
            })
    return pd.DataFrame(rows).sort_values(["进场阈值", "固定持有期（交易日）"]).reset_index(drop=True)


def calibrate_threshold_grid(
    cleaned_daily: pd.DataFrame,
    sector_daily: pd.DataFrame,
    *,
    entry_thresholds: Iterable[float],
    exit_gaps: Iterable[float],
    adjustment_policy: AdjustmentPolicy,
    transaction_cost: float = 0.001,
    calibration_ratio: float = 0.7,
) -> pd.DataFrame:
    """Run every candidate only on the chronological calibration segment.

    This is step five of the PRD workflow.  It intentionally does *not* inspect
    the final 30% validation segment, choose a winning row, or write a locked
    threshold.  The researcher selects a robust plateau from this table first;
    only that separately locked pair may later enter validation.
    """
    entry_values = sorted({float(value) for value in entry_thresholds})
    gap_values = sorted({float(value) for value in exit_gaps})
    if not entry_values or not gap_values:
        raise TrendTrackingError("calibration_grid_required", "候选进场阈值与离场阈值差必须至少各有一个数值。")
    segment = _calibration_segment(cleaned_daily, sector_daily, calibration_ratio=calibration_ratio)
    assert_scoring_is_causal(segment.sector_daily, segment.end)
    scored = score_all_sectors(segment.sector_daily)
    baseline_run = TrendTrackingRun(
        entry_threshold=entry_values[0],
        exit_threshold=max(entry_values[0] - gap_values[0], 0),
        adjustment_policy=adjustment_policy,
        transaction_cost=transaction_cost,
    )
    empty_signals = pd.DataFrame(columns=["Date", "Sector", "进场", "离场"])
    baseline_result = run_sector_backtest(segment.cleaned_daily, empty_signals, run=baseline_run)
    market_kpis = _curve_kpis(baseline_result.market_benchmark, pd.DataFrame())
    momentum_kpis = _curve_kpis(baseline_result.momentum_benchmark, pd.DataFrame())
    rows: list[dict[str, object]] = []
    for entry_threshold in entry_values:
        for exit_gap in gap_values:
            exit_threshold = entry_threshold - exit_gap
            if exit_threshold < 0:
                continue
            run = TrendTrackingRun(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                adjustment_policy=adjustment_policy,
                transaction_cost=transaction_cost,
            )
            signals = generate_sector_signals(scored, entry_threshold=entry_threshold, exit_threshold=exit_threshold)
            result = run_sector_backtest(segment.cleaned_daily, signals, run=run)
            calibration = result.kpis
            rows.append({
                "进场阈值": entry_threshold, "离场阈值": exit_threshold, "阈值差": exit_gap,
                "校准段起始日": segment.start, "校准段截止日": segment.end, "检验段起始日（未使用）": segment.validation_start,
                "校准段累计收益": calibration["cumulative_return_pct"], "校准段年化收益": calibration["annualized_return_pct"],
                "校准段最大回撤": calibration["max_drawdown_pct"], "校准段信号次数": calibration["signal_count"],
                "校准段胜率": calibration["win_rate_pct"], "校准段平均持有天数": calibration["average_holding_days"],
                "校准段等权大盘累计收益": market_kpis["cumulative_return_pct"],
                "校准段等权大盘年化收益": market_kpis["annualized_return_pct"],
                "校准段等权大盘最大回撤": market_kpis["max_drawdown_pct"],
                "校准段朴素动量累计收益": momentum_kpis["cumulative_return_pct"],
                "校准段朴素动量年化收益": momentum_kpis["annualized_return_pct"],
                "校准段朴素动量最大回撤": momentum_kpis["max_drawdown_pct"],
                "信号样本至少二十次": calibration["signal_count"] >= 20,
            })
    if not rows:
        raise TrendTrackingError("calibration_grid_invalid", "候选网格没有有效的非负离场阈值组合。")
    return pd.DataFrame(rows).sort_values(["进场阈值", "离场阈值"]).reset_index(drop=True)
