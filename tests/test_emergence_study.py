from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhein.emergence_study import (
    EmergenceStudyError,
    StudyConfig,
    cross_sectional_ic,
    newey_west_mean,
    quantile_analysis,
    run_horizon_study,
    run_synthetic_gate,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=12)
    columns = ["行业A", "行业B", "行业C", "行业D", "行业E", "行业F"]
    score = pd.DataFrame([np.arange(1, 7) + day % 2 for day in range(len(dates))], index=dates, columns=columns)
    outcome = pd.DataFrame([np.arange(1, 7) * (1 if day % 3 else -1) for day in range(len(dates))], index=dates, columns=columns)
    return score.astype(float), outcome.astype(float)


def test_newey_west_matches_hand_calculation() -> None:
    statistic = newey_west_mean([1.0, 2.0, 3.0], lag=1)

    assert statistic.mean == 2.0
    assert statistic.standard_error == pytest.approx(np.sqrt(2) / 3)
    assert statistic.t_statistic == pytest.approx(3 * np.sqrt(2))
    assert statistic.lag == 1


def test_cross_sectional_ic_is_not_time_series_correlation() -> None:
    score, outcome = _frames()
    metrics = cross_sectional_ic(score, outcome, minimum_sectors=5)

    assert len(metrics.daily_ic) == len(score)
    assert metrics.daily_ic.iloc[0] == pytest.approx(-1.0)
    assert metrics.daily_ic.iloc[1] == pytest.approx(1.0)
    assert (metrics.valid_sector_counts == 6).all()


def test_quantile_analysis_degrades_when_cross_section_is_too_thin() -> None:
    score, outcome = _frames()
    thin_score = score.iloc[:, :7 - 1]
    thin_outcome = outcome.iloc[:, :7 - 1]
    metrics = quantile_analysis(thin_score, thin_outcome, requested_quantiles=5, lag=2)

    assert metrics.downgraded_days == len(score)
    assert set(metrics.bucket_summary["bucket_count"]) == {3}
    assert len(metrics.daily_top_bottom_excess) == len(score)


def test_horizon_study_keeps_only_statistical_result_fields() -> None:
    score, outcome = _frames()
    result = run_horizon_study(
        score,
        outcome,
        components={"steady": score, "breadth": score, "volramp": score},
        horizon=20,
        config=StudyConfig(horizons=(20,), quantiles=3, minimum_sectors=5),
    )

    assert result.horizon == 20
    assert set(result.component_ic) == {"steady", "breadth", "volramp"}
    forbidden = {"equity", "annualized", "drawdown", "transaction_cost", "trade", "position"}
    assert not forbidden & set(result.__dataclass_fields__)
    assert not forbidden & set(result.composite_ic.__dataclass_fields__)


def test_synthetic_gate_passes_all_four_controls() -> None:
    report = run_synthetic_gate()

    assert report.passed
    assert [check.name for check in report.checks] == ["正对照", "负对照", "置换检验", "手算单测"]


def test_horizon_outside_preregistered_config_is_rejected() -> None:
    score, outcome = _frames()

    with pytest.raises(EmergenceStudyError) as caught:
        run_horizon_study(score, outcome, components={}, horizon=40, config=StudyConfig(horizons=(20,), quantiles=3, minimum_sectors=5))

    assert caught.value.code == "study_horizon_not_preregistered"
