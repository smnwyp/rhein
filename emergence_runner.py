"""板块萌芽统计研究的确定性主流程。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Mapping

import pandas as pd

from rhein.data.discovery import input_files
from rhein.emergence_data import (
    EmergenceFeatures,
    SectorWideData,
    aggregate_sector_wide,
    assert_features_are_causal,
    calculate_emergence_features,
    clean_study_daily,
    forward_excess_outcome,
    load_study_daily,
    load_study_map,
)
from rhein.emergence_study import (
    EmergenceStudyError,
    HorizonStudyResult,
    StudyConfig,
    SyntheticGateReport,
    run_horizon_study,
    run_synthetic_gate,
)
from rhein.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ResearchConfig:
    """从 TOML 加载的预注册研究配置。"""

    study_name: str
    sector_definition: str
    mapping_path: Path
    adjustment_required: bool
    static_universe_bias_notice: str
    feature_kwargs: Mapping[str, object]
    study_config: StudyConfig
    minimum_same_sign_years: int
    verdict_scope: str


@dataclass(frozen=True)
class HorizonVerdict:
    """一个 horizon 对预注册四条判据的逐项结论。"""

    horizon: int
    ic_positive_and_significant: bool
    yearly_sign_stable: bool
    quantiles_monotonic: bool
    non_overlapping_consistent: bool
    accepted: bool
    stable_same_sign_years: int


@dataclass(frozen=True)
class RealStudyResult:
    """真实研究的分层输出；L5 结果保持为纯统计对象。"""

    config: ResearchConfig
    synthetic_gate: SyntheticGateReport
    wide: SectorWideData
    features: EmergenceFeatures
    horizon_results: Mapping[int, HorizonStudyResult]
    verdicts: Mapping[int, HorizonVerdict]
    dropped_forward_tail: Mapping[int, int]
    source_file_count: int
    mapped_symbol_count: int


def load_research_config(path: Path | str = PROJECT_ROOT / "config" / "emergence_study.toml") -> ResearchConfig:
    """读取并校验研究配置；不接受阈值网格或交易参数。"""
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        features = raw["features"]
        statistics = raw["statistics"]
        gate = raw["synthetic_gate"]
        weights = features["weights"]
        study_config = StudyConfig(
            horizons=tuple(int(value) for value in statistics["horizons"]),
            quantiles=int(statistics["quantiles"]),
            minimum_sectors=int(statistics["minimum_sectors"]),
            synthetic_days=int(gate["synthetic_days"]),
            synthetic_sectors=int(gate["synthetic_sectors"]),
            synthetic_repetitions=int(gate["synthetic_repetitions"]),
            random_seed=int(gate["random_seed"]),
        )
        mapping_path = Path(str(raw["mapping_path"]))
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise EmergenceStudyError("research_config_invalid", "统计研究配置无效。", details={"path": str(source), "error": str(error)}) from error
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path
    feature_kwargs: dict[str, object] = {
        "steady_window": int(features["steady_window"]),
        "cap_window": int(features["cap_window"]),
        "cap_quantile": float(features["cap_quantile"]),
        "steady_target_days": float(features["steady_target_days"]),
        "breadth_window": int(features["breadth_window"]),
        "breadth_scale": float(features["breadth_scale"]),
        "weights": {name: float(value) for name, value in weights.items()},
    }
    verdict_scope = str(statistics["verdict_scope"])
    if verdict_scope != "each_horizon_separately":
        raise EmergenceStudyError("research_verdict_scope_invalid", "本阶段仅支持按 horizon 分别执行预注册判据。")
    return ResearchConfig(
        study_name=str(raw["study_name"]),
        sector_definition=str(raw["sector_definition"]),
        mapping_path=mapping_path,
        adjustment_required=bool(raw["adjustment_required"]),
        static_universe_bias_notice=str(raw["static_universe_bias_notice"]),
        feature_kwargs=feature_kwargs,
        study_config=study_config,
        minimum_same_sign_years=int(statistics["minimum_same_sign_years"]),
        verdict_scope=verdict_scope,
    )


def _verdict(result: HorizonStudyResult, *, minimum_same_sign_years: int, requested_quantiles: int) -> HorizonVerdict:
    ic = result.composite_ic
    hac = ic.hac
    ic_ok = bool(ic.mean_ic is not None and hac is not None and ic.mean_ic > 0 and abs(hac.t_statistic) > 2)
    if ic.mean_ic is None or ic.mean_ic == 0 or result.yearly.empty:
        same_sign_years = 0
    else:
        same_sign_years = int((result.yearly["mean_ic"] * ic.mean_ic > 0).sum())
    yearly_ok = same_sign_years >= minimum_same_sign_years
    quantile_hac = result.quantiles.hac
    quantile_ok = bool(
        result.quantiles.monotonic_by_bucket_count.get(requested_quantiles, False)
        and quantile_hac is not None
        and quantile_hac.mean > 0
    )
    non_overlapping = result.non_overlapping_ic
    non_overlapping_ok = bool(
        hac is not None and non_overlapping is not None
        and (hac.mean > 0 and abs(hac.t_statistic) > 2) == (non_overlapping.mean > 0 and abs(non_overlapping.t_statistic) > 2)
    )
    return HorizonVerdict(
        horizon=result.horizon,
        ic_positive_and_significant=ic_ok,
        yearly_sign_stable=yearly_ok,
        quantiles_monotonic=quantile_ok,
        non_overlapping_consistent=non_overlapping_ok,
        accepted=ic_ok and yearly_ok and quantile_ok and non_overlapping_ok,
        stable_same_sign_years=same_sign_years,
    )


def run_real_study(
    *,
    data_path: Path | str,
    adjustment_confirmed: bool,
    config: ResearchConfig | None = None,
) -> RealStudyResult:
    """运行真实统计研究；第一步强制四道合成闸门，不提供绕过路径。"""
    research_config = config or load_research_config()
    if research_config.adjustment_required and not adjustment_confirmed:
        raise EmergenceStudyError("adjustment_not_confirmed", "配置要求确认复权后才能运行真实统计研究。")
    # INV-5：此调用刻意位于所有真实数据读取之前，不能被参数跳过。
    gate = run_synthetic_gate(research_config.study_config)
    paths = input_files(Path(data_path))
    mapping = load_study_map(research_config.mapping_path)
    cleaned = clean_study_daily(load_study_daily(paths), mapping, adjustment_confirmed=adjustment_confirmed)
    wide = aggregate_sector_wide(cleaned)
    features = calculate_emergence_features(wide, **research_config.feature_kwargs)
    valid_feature_dates = features.composite.dropna(how="all").index
    if valid_feature_dates.empty:
        raise EmergenceStudyError("study_feature_history_insufficient", "历史长度不足，无法形成萌芽分。")
    assert_features_are_causal(wide, cutoff=valid_feature_dates[-1], feature_kwargs=research_config.feature_kwargs)
    components = {"steady": features.steady, "breadth": features.breadth, "volume_ramp": features.volume_ramp}
    horizon_results: dict[int, HorizonStudyResult] = {}
    verdicts: dict[int, HorizonVerdict] = {}
    dropped_forward_tail: dict[int, int] = {}
    for horizon in research_config.study_config.horizons:
        outcome = forward_excess_outcome(wide.daily_change, horizon=horizon)
        dropped_forward_tail[horizon] = int(outcome.tail(horizon).isna().all(axis=1).sum())
        result = run_horizon_study(
            features.composite, outcome, components=components, horizon=horizon, config=research_config.study_config
        )
        horizon_results[horizon] = result
        verdicts[horizon] = _verdict(
            result,
            minimum_same_sign_years=research_config.minimum_same_sign_years,
            requested_quantiles=research_config.study_config.quantiles,
        )
    return RealStudyResult(
        config=research_config,
        synthetic_gate=gate,
        wide=wide,
        features=features,
        horizon_results=horizon_results,
        verdicts=verdicts,
        dropped_forward_tail=dropped_forward_tail,
        source_file_count=len(paths),
        mapped_symbol_count=int(cleaned["Symbol"].nunique()),
    )
