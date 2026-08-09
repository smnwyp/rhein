"""板块萌芽统计研究的数据、特征与前向结果层。

这里的特征只使用决策日及之前的数据；前向超额结果仅作为统计检验的
被解释变量，绝不参与当日分数构造。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from rhein.data.ohlc import load_ohlc
from rhein.emergence_study import EmergenceStudyError


@dataclass(frozen=True)
class SectorWideData:
    """L3 的板块宽表：行是交易日，列是研究板块。"""

    daily_change: pd.DataFrame
    breadth: pd.DataFrame
    turnover: pd.DataFrame
    market_change: pd.Series
    excess_change: pd.DataFrame
    constituent_count: pd.DataFrame


@dataclass(frozen=True)
class EmergenceFeatures:
    """L4 特征宽表；所有字段均只依赖当前日及此前输入。"""

    steady: pd.DataFrame
    breadth: pd.DataFrame
    volume_ramp: pd.DataFrame
    composite: pd.DataFrame


_MAP_SYMBOL_COLUMNS = ("代码", "symbol", "Symbol", "ticker", "Ticker")
_MAP_SECTOR_COLUMNS = ("行业板块", "行业组", "industry_group", "sector", "Sector", "industry", "Industry")


def _column(frame: pd.DataFrame, options: tuple[str, ...], label: str) -> str:
    name = next((option for option in options if option in frame.columns), None)
    if name is None:
        raise EmergenceStudyError(
            "study_map_column_missing", f"研究板块对照表缺少{label}列。",
            details={"accepted_columns": list(options), "actual_columns": list(frame.columns)},
        )
    return name


def normalize_study_map(source: pd.DataFrame) -> pd.DataFrame:
    """校验研究用 ticker→细粒度板块静态快照。"""
    symbol_column = _column(source, _MAP_SYMBOL_COLUMNS, "代码")
    sector_column = _column(source, _MAP_SECTOR_COLUMNS, "研究板块")
    mapping = source.loc[:, [symbol_column, sector_column]].rename(columns={symbol_column: "Symbol", sector_column: "Sector"}).copy()
    mapping["Symbol"] = mapping["Symbol"].astype(str).str.strip().str.upper()
    mapping["Sector"] = mapping["Sector"].astype(str).str.strip()
    mapping = mapping[(mapping["Symbol"] != "") & (mapping["Sector"] != "")]
    if mapping.empty:
        raise EmergenceStudyError("study_map_empty", "研究板块对照表没有有效映射。")
    conflicts = mapping.groupby("Symbol")["Sector"].nunique()
    if (conflicts > 1).any():
        raise EmergenceStudyError(
            "study_map_conflict", "同一代码映射到多个研究板块。",
            details={"sample_symbols": conflicts[conflicts > 1].index[:10].tolist()},
        )
    return mapping.drop_duplicates("Symbol").sort_values("Symbol").reset_index(drop=True)


def load_study_map(path: Path) -> pd.DataFrame:
    """读取研究板块映射；失败信息带文件路径便于诊断。"""
    try:
        return normalize_study_map(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise EmergenceStudyError("study_map_read_failed", "无法读取研究板块对照表。", details={"path": str(path), "error": str(error)}) from error


def load_study_daily(paths: Iterable[Path]) -> pd.DataFrame:
    """加载显式日线文件，不静默跳过任何坏文件。"""
    parts: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            item = load_ohlc(path).loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            item.insert(1, "Symbol", path.stem.upper())
            parts.append(item)
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
    if failures:
        raise EmergenceStudyError("study_ohlcv_load_failed", "部分日线文件不可读取，拒绝在不完整数据上研究。", details={"failures": failures})
    if not parts:
        raise EmergenceStudyError("study_ohlcv_empty", "没有可用于统计研究的日线文件。")
    return pd.concat(parts, ignore_index=True)


def clean_study_daily(daily: pd.DataFrame, mapping: pd.DataFrame, *, adjustment_confirmed: bool) -> pd.DataFrame:
    """执行 L1/L2 最小契约；未确认复权时拒绝进入统计研究。"""
    if not adjustment_confirmed:
        raise EmergenceStudyError("adjustment_not_confirmed", "统计研究要求确认价格已一致复权；未确认时不得运行。")
    required_daily = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    missing_daily = required_daily - set(daily.columns)
    missing_map = {"Symbol", "Sector"} - set(mapping.columns)
    if missing_daily or missing_map:
        raise EmergenceStudyError(
            "study_input_schema_invalid", "日线或研究板块映射字段不完整。",
            details={"missing_daily": sorted(missing_daily), "missing_mapping": sorted(missing_map)},
        )
    frame = daily.loc[:, ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Symbol"] = frame["Symbol"].astype(str).str.upper().str.strip()
    for name in ("Open", "High", "Low", "Close", "Volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame.isna().any().any():
        invalid = frame[frame.isna().any(axis=1)]
        raise EmergenceStudyError("study_ohlcv_invalid", "日线含无法解析的日期或 OHLCV 数值。", details={"invalid_rows": len(invalid)})
    if frame.duplicated(["Date", "Symbol"]).any():
        raise EmergenceStudyError("study_ohlcv_duplicate", "同一代码同一交易日出现重复日线。")
    frame = frame.merge(mapping.loc[:, ["Symbol", "Sector"]], on="Symbol", how="inner", validate="many_to_one")
    frame = frame[(frame["Volume"] > 0) & (frame["Close"] >= 1.0)].copy()
    if frame.empty:
        raise EmergenceStudyError("study_clean_empty", "清洗后没有可用的日线记录。")
    frame = frame.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    frame["daily_change"] = frame.groupby("Symbol", sort=False)["Close"].pct_change()
    return frame.sort_values(["Date", "Sector", "Symbol"]).reset_index(drop=True)


def aggregate_sector_wide(cleaned_daily: pd.DataFrame) -> SectorWideData:
    """聚合为宽表；市场基准是当日全研究板块等权平均变化。"""
    required = {"Date", "Symbol", "Sector", "Close", "Volume", "daily_change"}
    missing = required - set(cleaned_daily.columns)
    if missing:
        raise EmergenceStudyError("study_clean_schema_invalid", "清洗层输出缺少板块聚合字段。", details={"missing": sorted(missing)})
    frame = cleaned_daily.dropna(subset=["daily_change"]).copy()
    if frame.empty:
        raise EmergenceStudyError("study_daily_change_empty", "没有足够相邻日线计算日变化。")
    frame["turnover"] = frame["Close"] * frame["Volume"]
    grouped = frame.groupby(["Date", "Sector"], as_index=False).agg(
        daily_change=("daily_change", "mean"),
        breadth=("daily_change", lambda values: float((values > 0).mean())),
        turnover=("turnover", "sum"),
        constituent_count=("Symbol", "size"),
    )
    def wide(value: str) -> pd.DataFrame:
        return grouped.pivot(index="Date", columns="Sector", values=value).sort_index().sort_index(axis=1)
    daily_change = wide("daily_change")
    market_change = daily_change.mean(axis=1, skipna=True).rename("market_change")
    return SectorWideData(
        daily_change=daily_change,
        breadth=wide("breadth"),
        turnover=wide("turnover"),
        market_change=market_change,
        excess_change=daily_change.sub(market_change, axis=0),
        constituent_count=wide("constituent_count"),
    )


def _volume_score(ratio: pd.DataFrame) -> pd.DataFrame:
    values = ratio.to_numpy(dtype=float)
    scored = np.full_like(values, np.nan, dtype=float)
    scored[values <= 1] = 0.0
    middle = (values > 1) & (values <= 1.5)
    scored[middle] = (values[middle] - 1) / 0.5 * 100
    scored[(values > 1.5) & (values <= 2)] = 100.0
    scored[values > 2] = 20.0
    return pd.DataFrame(scored, index=ratio.index, columns=ratio.columns)


def calculate_emergence_features(
    wide: SectorWideData,
    *,
    steady_window: int,
    cap_window: int,
    cap_quantile: float,
    steady_target_days: float,
    breadth_window: int,
    breadth_scale: float,
    weights: Mapping[str, float],
) -> EmergenceFeatures:
    """按预注册公式计算三个分量与合成分，严格只使用 <=t 数据。"""
    if steady_window <= 0 or cap_window <= 0 or breadth_window <= 0 or steady_target_days <= 0 or breadth_scale <= 0:
        raise EmergenceStudyError("feature_parameter_invalid", "萌芽分窗口与缩放参数必须为正数。")
    if not 0 < cap_quantile < 1:
        raise EmergenceStudyError("feature_cap_quantile_invalid", "暴涨分位必须严格介于 0 和 1。")
    if set(weights) != {"steady", "breadth", "volume_ramp"} or not np.isclose(sum(weights.values()), 1.0):
        raise EmergenceStudyError("feature_weights_invalid", "三项权重必须完整且和为 1。")
    excess = wide.excess_change
    # cap(t) 仅用 t 及以前的超额变化；随后对 t-19..t 的每一天用该
    # 决策日阈值计分，完全不读取 t 之后的数据。
    cap = excess.rolling(cap_window, min_periods=cap_window).quantile(cap_quantile)
    values = excess.to_numpy(dtype=float)
    cap_values = cap.to_numpy(dtype=float)
    steady_values = np.full_like(values, np.nan, dtype=float)
    for position in range(steady_window - 1, len(excess)):
        window = values[position - steady_window + 1 : position + 1]
        threshold = cap_values[position]
        valid = np.isfinite(window).all(axis=0) & np.isfinite(threshold)
        contribution = np.where(window > threshold, -0.5, np.where(window > 0, 1.0, 0.0))
        totals = contribution.sum(axis=0)
        steady_values[position, valid] = np.clip(totals[valid], 0, None) / steady_target_days * 100
    steady = pd.DataFrame(np.clip(steady_values, 0, 100), index=excess.index, columns=excess.columns)
    recent_breadth = wide.breadth.rolling(breadth_window, min_periods=breadth_window).mean()
    prior_breadth = recent_breadth.shift(breadth_window)
    breadth = ((recent_breadth - prior_breadth).clip(lower=0) / breadth_scale).clip(upper=1) * 100
    recent_turnover = wide.turnover.rolling(breadth_window, min_periods=breadth_window).mean()
    prior_turnover = recent_turnover.shift(breadth_window)
    volume_ramp = _volume_score(recent_turnover / prior_turnover.replace(0, np.nan))
    composite = weights["steady"] * steady + weights["breadth"] * breadth + weights["volume_ramp"] * volume_ramp
    return EmergenceFeatures(steady=steady, breadth=breadth, volume_ramp=volume_ramp, composite=composite)


def forward_excess_outcome(daily_change: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """构造 t 到 t+N 的前向超额结果；最后 N 日保持缺失而非填充。"""
    if horizon <= 0:
        raise EmergenceStudyError("forward_horizon_invalid", "前向 horizon 必须为正整数。")
    # 在 t 处，shift(-N) 取到的窗口是 t+1..t+N；这些 >t 的价格只作为
    # 统计检验结果，绝不反向输入当日的特征计算。
    forward_sector = (1 + daily_change).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    forward_market = forward_sector.mean(axis=1, skipna=True)
    return forward_sector.sub(forward_market, axis=0)


def assert_features_are_causal(
    wide: SectorWideData,
    *,
    cutoff: pd.Timestamp | str,
    feature_kwargs: Mapping[str, object],
) -> None:
    """截断后 t 日特征须与全样本 t 日逐值相同，防止未来函数。"""
    cutoff_date = pd.Timestamp(cutoff).normalize()
    full = calculate_emergence_features(wide, **feature_kwargs)
    mask = wide.daily_change.index <= cutoff_date
    truncated = SectorWideData(
        daily_change=wide.daily_change.loc[mask], breadth=wide.breadth.loc[mask], turnover=wide.turnover.loc[mask],
        market_change=wide.market_change.loc[mask], excess_change=wide.excess_change.loc[mask], constituent_count=wide.constituent_count.loc[mask],
    )
    limited = calculate_emergence_features(truncated, **feature_kwargs)
    for name in ("steady", "breadth", "volume_ramp", "composite"):
        left = getattr(full, name).loc[[cutoff_date]].sort_index(axis=1)
        right = getattr(limited, name).loc[[cutoff_date]].sort_index(axis=1)
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError as error:
            raise EmergenceStudyError("future_data_leak_detected", "截断测试失败：未来数据改变了历史萌芽分特征。", details={"feature": name, "cutoff": str(cutoff_date.date()), "difference": str(error)}) from error
