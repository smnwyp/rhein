"""v1.1 steady 分量单独检验：复用 v1.0 管线，不改变任何特征参数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Mapping

import pandas as pd

from emergence_runner import RealStudyResult, ResearchConfig, load_research_config, run_real_study
from rhein.emergence_data import forward_excess_outcome
from rhein.emergence_study import EmergenceStudyError, HorizonStudyResult, run_horizon_study
from rhein.paths import PROJECT_ROOT


@dataclass(frozen=True)
class SteadySupplementConfig:
    """steady 增补研究的预注册判据；没有可调特征或权重。"""

    study_name: str
    source_study_config: Path
    score_component: str
    static_universe_bias_notice: str
    required_monotonic_horizons: tuple[int, ...]
    required_all_horizons: tuple[int, ...]
    minimum_positive_spread_years: int
    early_years: tuple[int, ...]
    q5_must_be_positive: bool
    top_bottom_hac_abs_t_minimum: float


@dataclass(frozen=True)
class SteadyHorizonVerdict:
    """一个 horizon 的 steady 分位结构与稳定性判据。"""

    horizon: int
    quantiles_strictly_upward: bool
    q5_positive: bool
    top_bottom_positive_and_significant: bool
    positive_spread_years: int
    yearly_spread_stable: bool
    early_years_all_negative: bool
    non_overlapping_consistent: bool


@dataclass(frozen=True)
class SteadySupplementVerdict:
    """本次增补研究的三出口裁决。"""

    quantile_requirement_passed: bool
    top_bottom_requirement_passed: bool
    yearly_requirement_passed: bool
    non_overlapping_requirement_passed: bool
    outcome: str
    accepted_for_oos_design: bool


@dataclass(frozen=True)
class SteadySupplementResult:
    """v1.0 基础结果与 steady 单独送考结果的组合。"""

    config: SteadySupplementConfig
    base: RealStudyResult
    steady_horizon_results: Mapping[int, HorizonStudyResult]
    horizon_verdicts: Mapping[int, SteadyHorizonVerdict]
    verdict: SteadySupplementVerdict


def load_steady_supplement_config(path: Path | str = PROJECT_ROOT / "config" / "steady_supplement.toml") -> SteadySupplementConfig:
    """读取补充研究 TOML；禁止通过此配置改动 steady 特征常数。"""
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        criteria = raw["preregistered_criteria"]
        base_path = Path(str(raw["source_study_config"]))
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise EmergenceStudyError("steady_supplement_config_invalid", "steady 增补研究配置无效。", details={"path": str(source), "error": str(error)}) from error
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    component = str(raw["score_component"])
    if component != "steady":
        raise EmergenceStudyError("steady_supplement_component_invalid", "v1.1 只允许既有 steady 分量作为 score。")
    return SteadySupplementConfig(
        study_name=str(raw["study_name"]),
        source_study_config=base_path,
        score_component=component,
        static_universe_bias_notice=str(raw["static_universe_bias_notice"]),
        required_monotonic_horizons=tuple(int(value) for value in criteria["required_monotonic_horizons"]),
        required_all_horizons=tuple(int(value) for value in criteria["required_all_horizons"]),
        minimum_positive_spread_years=int(criteria["minimum_positive_spread_years"]),
        early_years=tuple(int(value) for value in criteria["early_years"]),
        q5_must_be_positive=bool(criteria["q5_must_be_positive"]),
        top_bottom_hac_abs_t_minimum=float(criteria["top_bottom_hac_abs_t_minimum"]),
    )


def _summary_for_requested_quantiles(result: HorizonStudyResult, quantiles: int) -> pd.DataFrame:
    return result.quantiles.bucket_summary.query("bucket_count == @quantiles").sort_values("bucket").reset_index(drop=True)


def _horizon_verdict(
    result: HorizonStudyResult,
    *,
    research_config: ResearchConfig,
    supplement_config: SteadySupplementConfig,
) -> SteadyHorizonVerdict:
    summary = _summary_for_requested_quantiles(result, research_config.study_config.quantiles)
    values = summary["mean_forward_excess"].to_numpy()
    strictly_upward = bool(
        len(values) == research_config.study_config.quantiles and (pd.Series(values).diff().iloc[1:] > 0).all()
    )
    q5_positive = bool(len(values) == research_config.study_config.quantiles and values[-1] > 0) if supplement_config.q5_must_be_positive else True
    spread_hac = result.quantiles.hac
    top_bottom_ok = bool(
        spread_hac is not None and spread_hac.mean > 0 and abs(spread_hac.t_statistic) > supplement_config.top_bottom_hac_abs_t_minimum
    )
    yearly = result.yearly
    positive_spread_years = int((yearly["mean_top_bottom_excess"] > 0).sum()) if not yearly.empty else 0
    early = yearly[yearly["year"].isin(supplement_config.early_years)]
    early_all_negative = bool(len(early) == len(supplement_config.early_years) and (early["mean_top_bottom_excess"] < 0).all())
    yearly_ok = positive_spread_years >= supplement_config.minimum_positive_spread_years and not early_all_negative
    nonoverlap = result.non_overlapping_ic
    ic_hac = result.composite_ic.hac
    # 非重叠核对沿用 v1.0 的 IC 方法：与同一 steady IC 的 HAC 结论
    # 比较，不把分位价差与 IC 两种统计量混为一谈。
    nonoverlap_ok = bool(
        ic_hac is not None and nonoverlap is not None
        and (ic_hac.mean > 0 and abs(ic_hac.t_statistic) > 2)
        == (nonoverlap.mean > 0 and abs(nonoverlap.t_statistic) > 2)
    )
    return SteadyHorizonVerdict(
        horizon=result.horizon,
        quantiles_strictly_upward=strictly_upward,
        q5_positive=q5_positive,
        top_bottom_positive_and_significant=top_bottom_ok,
        positive_spread_years=positive_spread_years,
        yearly_spread_stable=yearly_ok,
        early_years_all_negative=early_all_negative,
        non_overlapping_consistent=nonoverlap_ok,
    )


def _verdict(
    horizon_verdicts: Mapping[int, SteadyHorizonVerdict],
    *,
    supplement_config: SteadySupplementConfig,
) -> SteadySupplementVerdict:
    required_quantile = [horizon_verdicts[horizon] for horizon in supplement_config.required_monotonic_horizons]
    required_all = [horizon_verdicts[horizon] for horizon in supplement_config.required_all_horizons]
    quantile_ok = all(item.quantiles_strictly_upward and item.q5_positive for item in required_quantile)
    top_bottom_ok = all(item.top_bottom_positive_and_significant for item in required_all)
    yearly_ok = all(item.yearly_spread_stable for item in required_all)
    nonoverlap_ok = all(item.non_overlapping_consistent for item in required_all)
    if quantile_ok and top_bottom_ok and yearly_ok and nonoverlap_ok:
        outcome = "通过：steady 注册为独立假设 H-steady，可进入 OOS 检验设计"
        accepted = True
    elif quantile_ok and top_bottom_ok and nonoverlap_ok and any(item.early_years_all_negative for item in required_all):
        outcome = "regime 依赖：早期年份价差结构不成立；不外推、不使用，仅留档观察"
        accepted = False
    else:
        outcome = "否决：steady 未满足结构干净判据；板块萌芽方向关闭"
        accepted = False
    return SteadySupplementVerdict(
        quantile_requirement_passed=quantile_ok,
        top_bottom_requirement_passed=top_bottom_ok,
        yearly_requirement_passed=yearly_ok,
        non_overlapping_requirement_passed=nonoverlap_ok,
        outcome=outcome,
        accepted_for_oos_design=accepted,
    )


def run_steady_supplement(
    *,
    data_path: Path | str,
    adjustment_confirmed: bool,
    config: SteadySupplementConfig | None = None,
) -> SteadySupplementResult:
    """用 v1.0 的相同数据和参数，仅将 score 换为 steady。"""
    supplement_config = config or load_steady_supplement_config()
    research_config = load_research_config(supplement_config.source_study_config)
    base = run_real_study(data_path=data_path, adjustment_confirmed=adjustment_confirmed, config=research_config)
    unavailable = set(supplement_config.required_all_horizons) - set(research_config.study_config.horizons)
    if unavailable:
        raise EmergenceStudyError("steady_supplement_horizon_invalid", "补充研究要求的 horizon 不在 v1.0 预注册集合中。", details={"unavailable": sorted(unavailable)})
    results: dict[int, HorizonStudyResult] = {}
    verdicts: dict[int, SteadyHorizonVerdict] = {}
    for horizon in research_config.study_config.horizons:
        outcome = forward_excess_outcome(base.wide.daily_change, horizon=horizon)
        result = run_horizon_study(
            base.features.steady,
            outcome,
            components={},
            horizon=horizon,
            config=research_config.study_config,
        )
        results[horizon] = result
        verdicts[horizon] = _horizon_verdict(
            result, research_config=research_config, supplement_config=supplement_config
        )
    return SteadySupplementResult(
        config=supplement_config,
        base=base,
        steady_horizon_results=results,
        horizon_verdicts=verdicts,
        verdict=_verdict(verdicts, supplement_config=supplement_config),
    )
