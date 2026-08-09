"""板块萌芽现象的纯统计检验内核。

本模块只处理横截面分数与前向超额结果。它没有交易、仓位、成本或
净值概念；真实数据研究必须先通过 :func:`run_synthetic_gate`。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr


class EmergenceStudyError(ValueError):
    """统计研究输入或前置自检不满足契约时的可诊断错误。"""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class StudyConfig:
    """预注册的统计研究常数；这些值只用于复现，不能供网格搜索。"""

    horizons: tuple[int, ...] = (20, 40, 60)
    quantiles: int = 5
    minimum_sectors: int = 5
    synthetic_days: int = 480
    synthetic_sectors: int = 30
    synthetic_repetitions: int = 200
    random_seed: int = 20260809


@dataclass(frozen=True)
class HACStatistic:
    """一条统计序列均值的 Newey-West / HAC 检验结果。"""

    mean: float
    standard_error: float
    t_statistic: float
    p_value: float
    observations: int
    lag: int


@dataclass(frozen=True)
class ICMetrics:
    """每日横截面 IC 序列的汇总；不含任何交易表现字段。"""

    daily_ic: pd.Series
    valid_sector_counts: pd.Series
    skipped_days: int
    mean_ic: float | None
    standard_deviation: float | None
    information_ratio: float | None
    positive_day_fraction: float | None
    hac: HACStatistic | None


@dataclass(frozen=True)
class QuantileMetrics:
    """分位数横截面结果及 Top-Bottom 的统计检验。"""

    daily_bucket_means: pd.DataFrame
    bucket_summary: pd.DataFrame
    daily_top_bottom_excess: pd.Series
    hac: HACStatistic | None
    monotonic_by_bucket_count: Mapping[int, bool]
    downgraded_days: int
    skipped_days: int


@dataclass(frozen=True)
class HorizonStudyResult:
    """一个 horizon 的完整统计结果，字段被限制为研究统计量。"""

    horizon: int
    composite_ic: ICMetrics
    component_ic: Mapping[str, ICMetrics]
    quantiles: QuantileMetrics
    yearly: pd.DataFrame
    non_overlapping_ic: HACStatistic | None


@dataclass(frozen=True)
class SyntheticCheck:
    """一个合成对照关卡的可审计结论。"""

    name: str
    passed: bool
    details: Mapping[str, float | int | bool]


@dataclass(frozen=True)
class SyntheticGateReport:
    """真实数据研究前必须全数通过的四道合成对照。"""

    checks: tuple[SyntheticCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _validate_config(config: StudyConfig) -> None:
    if not config.horizons or any(not isinstance(value, int) or value <= 0 for value in config.horizons):
        raise EmergenceStudyError("study_horizons_invalid", "研究 horizon 必须是正整数列表。")
    if config.quantiles < 2:
        raise EmergenceStudyError("study_quantiles_invalid", "分位档数至少为 2。")
    if config.minimum_sectors < 3:
        raise EmergenceStudyError("study_minimum_sectors_invalid", "横截面最小板块数至少为 3。")
    if config.synthetic_days < 40 or config.synthetic_sectors < config.minimum_sectors:
        raise EmergenceStudyError("synthetic_control_shape_invalid", "合成对照的日期或板块数不足。")


def _aligned_frames(score: pd.DataFrame, outcome: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(score.index, pd.DatetimeIndex) or not isinstance(outcome.index, pd.DatetimeIndex):
        raise EmergenceStudyError("study_date_index_required", "分数与前向超额结果必须使用 DatetimeIndex。")
    common_dates = score.index.intersection(outcome.index).sort_values()
    common_sectors = score.columns.intersection(outcome.columns).sort_values()
    if common_dates.empty or common_sectors.empty:
        raise EmergenceStudyError("study_no_common_cross_section", "分数与前向超额结果没有共同的日期或板块。")
    return score.loc[common_dates, common_sectors].astype(float), outcome.loc[common_dates, common_sectors].astype(float)


def newey_west_mean(values: Sequence[float] | pd.Series | np.ndarray, *, lag: int) -> HACStatistic:
    """用 Bartlett 核计算均值的 Newey-West 标准误与双侧 p 值。"""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    observations = len(array)
    if observations < 2:
        raise EmergenceStudyError("hac_observations_insufficient", "HAC 检验至少需要两个有限观测。", details={"observations": observations})
    if lag < 0:
        raise EmergenceStudyError("hac_lag_invalid", "HAC 滞后阶不能为负数。")
    effective_lag = min(int(lag), observations - 1)
    mean = float(array.mean())
    residuals = array - mean
    gamma_zero = float(np.mean(residuals * residuals))
    long_run_variance = gamma_zero
    for step in range(1, effective_lag + 1):
        autocovariance = float(np.mean(residuals[step:] * residuals[:-step]))
        bartlett_weight = 1 - step / (effective_lag + 1)
        long_run_variance += 2 * bartlett_weight * autocovariance
    # 有限样本数值舍入可带来极小负值；理论上该量非负。
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / observations))
    if standard_error == 0:
        t_statistic = 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
        p_value = 1.0 if mean == 0 else 0.0
    else:
        t_statistic = mean / standard_error
        p_value = float(2 * norm.sf(abs(t_statistic)))
    return HACStatistic(
        mean=mean,
        standard_error=standard_error,
        t_statistic=float(t_statistic),
        p_value=p_value,
        observations=observations,
        lag=effective_lag,
    )


def cross_sectional_ic(score: pd.DataFrame, outcome: pd.DataFrame, *, minimum_sectors: int) -> ICMetrics:
    """逐日计算 Spearman 横截面 IC；每一日只横向比较板块。"""
    if minimum_sectors < 3:
        raise EmergenceStudyError("study_minimum_sectors_invalid", "横截面最小板块数至少为 3。")
    score_frame, outcome_frame = _aligned_frames(score, outcome)
    daily: dict[pd.Timestamp, float] = {}
    counts: dict[pd.Timestamp, int] = {}
    skipped_days = 0
    score_values = score_frame.to_numpy(dtype=float)
    outcome_values = outcome_frame.to_numpy(dtype=float)
    # 合成对照与完整宽表通常没有单格缺失。此路径一次性对所有日期
    # 做秩化，避免负对照 200 次重复时退化为近十万次 Python 调用。
    if np.isfinite(score_values).all() and np.isfinite(outcome_values).all():
        row_count, sector_count = score_values.shape
        counts = {pd.Timestamp(date): sector_count for date in score_frame.index}
        if sector_count < minimum_sectors:
            skipped_days = row_count
        else:
            score_rank = rankdata(score_values, axis=1)
            outcome_rank = rankdata(outcome_values, axis=1)
            score_centered = score_rank - score_rank.mean(axis=1, keepdims=True)
            outcome_centered = outcome_rank - outcome_rank.mean(axis=1, keepdims=True)
            denominator = np.sqrt((score_centered * score_centered).sum(axis=1) * (outcome_centered * outcome_centered).sum(axis=1))
            values = np.divide(
                (score_centered * outcome_centered).sum(axis=1), denominator,
                out=np.full(row_count, np.nan), where=denominator > 0,
            )
            for date, value in zip(score_frame.index, values):
                if isfinite(float(value)):
                    daily[pd.Timestamp(date)] = float(value)
                else:
                    skipped_days += 1
        daily_ic = pd.Series(daily, dtype=float, name="ic").sort_index()
        valid_counts = pd.Series(counts, dtype=int, name="valid_sector_count").sort_index()
        if daily_ic.empty:
            return ICMetrics(daily_ic, valid_counts, skipped_days, None, None, None, None, None)
        standard_deviation = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else 0.0
        mean_ic = float(daily_ic.mean())
        return ICMetrics(
            daily_ic=daily_ic,
            valid_sector_counts=valid_counts,
            skipped_days=skipped_days,
            mean_ic=mean_ic,
            standard_deviation=standard_deviation,
            information_ratio=mean_ic / standard_deviation if standard_deviation > 0 else None,
            positive_day_fraction=float((daily_ic > 0).mean()),
            hac=None,
        )
    for date in score_frame.index:
        pair = pd.DataFrame({"score": score_frame.loc[date], "outcome": outcome_frame.loc[date]}).dropna()
        counts[pd.Timestamp(date)] = len(pair)
        if len(pair) < minimum_sectors or pair["score"].nunique() < 2 or pair["outcome"].nunique() < 2:
            skipped_days += 1
            continue
        value = float(spearmanr(pair["score"], pair["outcome"]).statistic)
        if isfinite(value):
            daily[pd.Timestamp(date)] = value
        else:
            skipped_days += 1
    daily_ic = pd.Series(daily, dtype=float, name="ic").sort_index()
    valid_counts = pd.Series(counts, dtype=int, name="valid_sector_count").sort_index()
    if daily_ic.empty:
        return ICMetrics(daily_ic, valid_counts, skipped_days, None, None, None, None, None)
    standard_deviation = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else 0.0
    mean_ic = float(daily_ic.mean())
    information_ratio = mean_ic / standard_deviation if standard_deviation > 0 else None
    hac = newey_west_mean(daily_ic, lag=0) if len(daily_ic) < 3 else None
    return ICMetrics(
        daily_ic=daily_ic,
        valid_sector_counts=valid_counts,
        skipped_days=skipped_days,
        mean_ic=mean_ic,
        standard_deviation=standard_deviation,
        information_ratio=information_ratio,
        positive_day_fraction=float((daily_ic > 0).mean()),
        hac=hac,
    )


def _with_hac(metrics: ICMetrics, *, lag: int) -> ICMetrics:
    if metrics.daily_ic.empty:
        return metrics
    return ICMetrics(
        daily_ic=metrics.daily_ic,
        valid_sector_counts=metrics.valid_sector_counts,
        skipped_days=metrics.skipped_days,
        mean_ic=metrics.mean_ic,
        standard_deviation=metrics.standard_deviation,
        information_ratio=metrics.information_ratio,
        positive_day_fraction=metrics.positive_day_fraction,
        hac=newey_west_mean(metrics.daily_ic, lag=lag),
    )


def quantile_analysis(score: pd.DataFrame, outcome: pd.DataFrame, *, requested_quantiles: int, lag: int) -> QuantileMetrics:
    """按日分位比较前向超额结果，并记录板块不足时的降级。"""
    if requested_quantiles < 2:
        raise EmergenceStudyError("study_quantiles_invalid", "分位档数至少为 2。")
    score_frame, outcome_frame = _aligned_frames(score, outcome)
    rows: list[dict[str, object]] = []
    spreads: dict[pd.Timestamp, float] = {}
    downgraded_days = 0
    skipped_days = 0
    for date in score_frame.index:
        pair = pd.DataFrame({"score": score_frame.loc[date], "outcome": outcome_frame.loc[date]}).dropna()
        count = len(pair)
        possible = next((value for value in (requested_quantiles, 3, 2) if count >= 2 * value), None)
        if possible is None or pair["score"].nunique() < 2:
            skipped_days += 1
            continue
        if possible != requested_quantiles:
            downgraded_days += 1
        # average ranks keep ties together; an empty bucket is explicitly
        # absent instead of fabricating a value.
        rank_fraction = pair["score"].rank(method="average", pct=True)
        bucket = np.minimum(np.ceil(rank_fraction * possible).astype(int), possible)
        grouped = pair.assign(bucket=bucket).groupby("bucket", sort=True)["outcome"].mean()
        for bucket_number, mean_value in grouped.items():
            rows.append({
                "date": pd.Timestamp(date), "bucket_count": possible, "bucket": int(bucket_number),
                "mean_forward_excess": float(mean_value), "valid_sector_count": count,
            })
        if 1 in grouped.index and possible in grouped.index:
            spreads[pd.Timestamp(date)] = float(grouped.loc[possible] - grouped.loc[1])
    daily = pd.DataFrame(rows, columns=["date", "bucket_count", "bucket", "mean_forward_excess", "valid_sector_count"])
    if daily.empty:
        summary = pd.DataFrame(columns=["bucket_count", "bucket", "mean_forward_excess", "observation_days"])
    else:
        summary = (
            daily.groupby(["bucket_count", "bucket"], as_index=False)
            .agg(mean_forward_excess=("mean_forward_excess", "mean"), observation_days=("date", "nunique"))
            .sort_values(["bucket_count", "bucket"])
            .reset_index(drop=True)
        )
    monotonic: dict[int, bool] = {}
    for bucket_count, section in summary.groupby("bucket_count", sort=True):
        values = section.sort_values("bucket")["mean_forward_excess"].to_numpy()
        monotonic[int(bucket_count)] = bool(len(values) == int(bucket_count) and np.all(np.diff(values) > 0))
    daily_spread = pd.Series(spreads, dtype=float, name="top_bottom_excess").sort_index()
    return QuantileMetrics(
        daily_bucket_means=daily,
        bucket_summary=summary,
        daily_top_bottom_excess=daily_spread,
        hac=newey_west_mean(daily_spread, lag=lag) if len(daily_spread) >= 2 else None,
        monotonic_by_bucket_count=monotonic,
        downgraded_days=downgraded_days,
        skipped_days=skipped_days,
    )


def yearly_ic_metrics(ic: pd.Series, top_bottom_excess: pd.Series) -> pd.DataFrame:
    """按自然年汇总 IC 与 Top-Bottom，供稳定性判断使用。"""
    if ic.empty:
        return pd.DataFrame(columns=["year", "mean_ic", "mean_top_bottom_excess", "ic_observations", "spread_observations"])
    ic_frame = ic.rename("ic").to_frame()
    ic_frame["year"] = ic_frame.index.year
    result = ic_frame.groupby("year", as_index=False).agg(mean_ic=("ic", "mean"), ic_observations=("ic", "size"))
    if not top_bottom_excess.empty:
        spread_frame = top_bottom_excess.rename("spread").to_frame()
        spread_frame["year"] = spread_frame.index.year
        spreads = spread_frame.groupby("year", as_index=False).agg(
            mean_top_bottom_excess=("spread", "mean"), spread_observations=("spread", "size")
        )
        result = result.merge(spreads, on="year", how="left", validate="one_to_one")
    else:
        result["mean_top_bottom_excess"] = np.nan
        result["spread_observations"] = 0
    return result.sort_values("year").reset_index(drop=True)


def non_overlapping_ic_check(ic: pd.Series, *, horizon: int) -> HACStatistic | None:
    """每 horizon 个有效日期抽一次 IC，作为与 HAC 对照的独立样本近似。"""
    if horizon <= 0:
        raise EmergenceStudyError("study_horizon_invalid", "horizon 必须是正整数。")
    sampled = ic.iloc[::horizon]
    return newey_west_mean(sampled, lag=0) if len(sampled) >= 2 else None


def run_horizon_study(
    score: pd.DataFrame,
    forward_excess: pd.DataFrame,
    *,
    components: Mapping[str, pd.DataFrame],
    horizon: int,
    config: StudyConfig,
) -> HorizonStudyResult:
    """在一个预注册 horizon 下计算所有横截面统计量。"""
    _validate_config(config)
    if horizon not in config.horizons:
        raise EmergenceStudyError("study_horizon_not_preregistered", "请求的 horizon 不在预注册配置中。", details={"horizon": horizon})
    composite_ic = _with_hac(cross_sectional_ic(score, forward_excess, minimum_sectors=config.minimum_sectors), lag=horizon)
    component_ic = {
        name: _with_hac(cross_sectional_ic(component_score, forward_excess, minimum_sectors=config.minimum_sectors), lag=horizon)
        for name, component_score in components.items()
    }
    quantiles = quantile_analysis(score, forward_excess, requested_quantiles=config.quantiles, lag=horizon)
    return HorizonStudyResult(
        horizon=horizon,
        composite_ic=composite_ic,
        component_ic=component_ic,
        quantiles=quantiles,
        yearly=yearly_ic_metrics(composite_ic.daily_ic, quantiles.daily_top_bottom_excess),
        non_overlapping_ic=non_overlapping_ic_check(composite_ic.daily_ic, horizon=horizon),
    )


def _synthetic_frames(*, days: int, sectors: int, beta: float, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2018-01-01", periods=days)
    names = [f"S{number:02d}" for number in range(sectors)]
    score_values = rng.normal(size=(days, sectors))
    standardized = (score_values - score_values.mean(axis=1, keepdims=True)) / score_values.std(axis=1, keepdims=True)
    outcome_values = beta * standardized + rng.normal(size=(days, sectors))
    return pd.DataFrame(score_values, index=dates, columns=names), pd.DataFrame(outcome_values, index=dates, columns=names)


def _positive_control(config: StudyConfig) -> SyntheticCheck:
    score, outcome = _synthetic_frames(
        days=config.synthetic_days, sectors=config.synthetic_sectors, beta=0.11,
        rng=np.random.default_rng(config.random_seed),
    )
    ic = _with_hac(cross_sectional_ic(score, outcome, minimum_sectors=config.minimum_sectors), lag=20)
    quantiles = quantile_analysis(score, outcome, requested_quantiles=config.quantiles, lag=20)
    mean_ic = float(ic.mean_ic or 0.0)
    top_bottom = float(quantiles.hac.mean) if quantiles.hac is not None else 0.0
    passed = 0.08 <= mean_ic <= 0.12 and bool(quantiles.monotonic_by_bucket_count.get(config.quantiles)) and top_bottom > 0 and bool(quantiles.hac and abs(quantiles.hac.t_statistic) > 2)
    return SyntheticCheck("正对照", passed, {"mean_ic": mean_ic, "top_bottom": top_bottom, "hac_t": float(quantiles.hac.t_statistic if quantiles.hac else 0.0)})


def _negative_control(config: StudyConfig) -> SyntheticCheck:
    significant = 0
    mean_ics: list[float] = []
    rng = np.random.default_rng(config.random_seed + 1)
    for _ in range(config.synthetic_repetitions):
        score, outcome = _synthetic_frames(days=config.synthetic_days, sectors=config.synthetic_sectors, beta=0.0, rng=rng)
        ic = _with_hac(cross_sectional_ic(score, outcome, minimum_sectors=config.minimum_sectors), lag=20)
        if ic.hac is None or ic.mean_ic is None:
            raise EmergenceStudyError("synthetic_control_failed", "负对照没有生成足够的 IC 观测。")
        mean_ics.append(ic.mean_ic)
        if ic.hac.p_value < 0.05:
            significant += 1
    false_positive_rate = significant / config.synthetic_repetitions
    mean_ic = float(np.mean(mean_ics))
    # 对 200 次伯努利试验而言，10% 已显著偏离名义 5%，是保守的失败界线。
    passed = abs(mean_ic) <= 0.015 and false_positive_rate <= 0.10
    return SyntheticCheck("负对照", passed, {"mean_ic": mean_ic, "false_positive_rate": false_positive_rate, "repetitions": config.synthetic_repetitions})


def _permutation_control(config: StudyConfig) -> SyntheticCheck:
    score, outcome = _synthetic_frames(
        days=config.synthetic_days, sectors=config.synthetic_sectors, beta=0.25,
        rng=np.random.default_rng(config.random_seed + 2),
    )
    rng = np.random.default_rng(config.random_seed + 3)
    shuffled = score.copy()
    for date in shuffled.index:
        shuffled.loc[date] = rng.permutation(shuffled.loc[date].to_numpy())
    ic = _with_hac(cross_sectional_ic(shuffled, outcome, minimum_sectors=config.minimum_sectors), lag=20)
    mean_ic = float(ic.mean_ic or 0.0)
    p_value = float(ic.hac.p_value if ic.hac else 1.0)
    return SyntheticCheck("置换检验", abs(mean_ic) <= 0.03 and p_value >= 0.05, {"mean_ic": mean_ic, "p_value": p_value})


def _hand_calculation_control() -> SyntheticCheck:
    dates = pd.bdate_range("2025-01-02", periods=3)
    score = pd.DataFrame([[1, 2, 3], [3, 2, 1], [1, 2, 3]], index=dates, columns=["A", "B", "C"])
    outcome = pd.DataFrame([[4, 5, 6], [1, 2, 3], [3, 2, 1]], index=dates, columns=["A", "B", "C"])
    ic = cross_sectional_ic(score, outcome, minimum_sectors=3)
    hac = newey_west_mean([1.0, 2.0, 3.0], lag=1)
    expected_ic = [1.0, -1.0, -1.0]
    expected_t = 3 * np.sqrt(2)
    passed = np.allclose(ic.daily_ic.to_numpy(), expected_ic, atol=1e-12) and np.isclose(hac.t_statistic, expected_t, atol=1e-12)
    return SyntheticCheck("手算单测", bool(passed), {"first_day_ic": float(ic.daily_ic.iloc[0]), "hac_t": float(hac.t_statistic)})


def run_synthetic_gate(config: StudyConfig = StudyConfig()) -> SyntheticGateReport:
    """运行四道合成对照；任一失败即抛错并拒绝真实数据研究。"""
    _validate_config(config)
    report = SyntheticGateReport(checks=(
        _positive_control(config),
        _negative_control(config),
        _permutation_control(config),
        _hand_calculation_control(),
    ))
    if not report.passed:
        failures = [check.name for check in report.checks if not check.passed]
        raise EmergenceStudyError("synthetic_gate_failed", "合成对照未全部通过，禁止运行真实数据研究。", details={"failed_checks": failures})
    return report
