from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhein.emergence_data import (
    SectorWideData,
    _volume_score,
    aggregate_sector_wide,
    assert_features_are_causal,
    calculate_emergence_features,
    clean_study_daily,
    forward_excess_outcome,
    normalize_study_map,
)
from rhein.emergence_study import EmergenceStudyError


FEATURE_KWARGS = {
    "steady_window": 3,
    "cap_window": 4,
    "cap_quantile": 0.98,
    "steady_target_days": 2,
    "breadth_window": 2,
    "breadth_scale": 0.10,
    "weights": {"steady": 0.4, "breadth": 0.4, "volume_ramp": 0.2},
}


def _daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=12)
    rows: list[dict[str, object]] = []
    mapping: list[dict[str, str]] = []
    for sector_number, sector in enumerate(("行业甲", "行业乙", "行业丙")):
        for number in range(3):
            symbol = f"{sector_number}{number}"
            mapping.append({"代码": symbol, "行业板块": sector})
            close = 10 + number
            for day, date in enumerate(dates):
                close *= 1 + 0.005 * (sector_number - 1) + (0.002 if day % 4 else -0.001)
                rows.append({"Date": date, "Symbol": symbol, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000 + day * 10})
    return pd.DataFrame(rows), pd.DataFrame(mapping)


def test_study_cleaning_requires_confirmed_adjustment_and_produces_wide_sector_data() -> None:
    daily, raw_map = _daily()
    mapping = normalize_study_map(raw_map)
    with pytest.raises(EmergenceStudyError) as caught:
        clean_study_daily(daily, mapping, adjustment_confirmed=False)
    assert caught.value.code == "adjustment_not_confirmed"

    wide = aggregate_sector_wide(clean_study_daily(daily, mapping, adjustment_confirmed=True))
    assert set(wide.daily_change.columns) == {"行业甲", "行业乙", "行业丙"}
    assert wide.market_change.equals(wide.daily_change.mean(axis=1, skipna=True).rename("market_change"))
    assert wide.excess_change.loc[wide.daily_change.index[1]].mean() == pytest.approx(0.0)


def test_feature_calculation_is_causal_under_truncation() -> None:
    daily, raw_map = _daily()
    wide = aggregate_sector_wide(clean_study_daily(daily, normalize_study_map(raw_map), adjustment_confirmed=True))
    cutoff = wide.daily_change.index[-2]
    assert_features_are_causal(wide, cutoff=cutoff, feature_kwargs=FEATURE_KWARGS)


def test_forward_excess_uses_only_future_window_and_leaves_tail_missing() -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    changes = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    outcome = forward_excess_outcome(changes, horizon=2)

    sector_a = (1.10 * 1.10 - 1)
    expected = sector_a - sector_a / 2
    assert outcome.loc[dates[0], "A"] == pytest.approx(expected)
    assert outcome.loc[dates[-2]:, :].isna().all().all()


def test_volume_score_boundaries_flow_into_composite_without_future_data() -> None:
    daily, raw_map = _daily()
    wide = aggregate_sector_wide(clean_study_daily(daily, normalize_study_map(raw_map), adjustment_confirmed=True))
    features = calculate_emergence_features(wide, **FEATURE_KWARGS)

    assert features.composite.index.equals(wide.daily_change.index)
    assert features.composite.columns.equals(wide.daily_change.columns)
    assert np.isfinite(features.composite.to_numpy()[~np.isnan(features.composite.to_numpy())]).all()


def test_hand_calculation_covers_cap_window_and_volume_boundaries() -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    excess = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.02]}, index=dates)
    zeros = pd.DataFrame({"A": [0.5] * 5}, index=dates)
    wide = SectorWideData(
        daily_change=excess,
        breadth=zeros,
        turnover=pd.DataFrame({"A": [100.0] * 5}, index=dates),
        market_change=pd.Series([0.0] * 5, index=dates),
        excess_change=excess,
        constituent_count=pd.DataFrame({"A": [3] * 5}, index=dates),
    )
    features = calculate_emergence_features(wide, **FEATURE_KWARGS)
    # day 5 cap=quantile([.02,.03,.04,.02], .98)=.0394；最近三日贡献
    # 为 +1,-.5,+1，因此稳赢分=1.5/2*100=75。
    assert features.steady.loc[dates[-1], "A"] == pytest.approx(75.0)
    ratio = pd.DataFrame({"A": [1.0, 1.25, 1.5, 2.0, 2.01]})
    assert _volume_score(ratio)["A"].tolist() == pytest.approx([0.0, 50.0, 100.0, 100.0, 20.0])


def test_research_industry_map_uses_nasdaq_industry_without_claiming_gics() -> None:
    from scripts.build_research_industry_map import build_research_industry_map

    mapped, unmapped = build_research_industry_map(pd.DataFrame({
        "代码": ["AAA", "BBB"],
        "细分行业": ["Computer Software", ""],
        "Nasdaq板块": ["Technology", "Finance"],
        "数据源": ["Nasdaq", "Nasdaq"],
        "分类快照UTC": ["2026-08-09T00:00:00+00:00", "2026-08-09T00:00:00+00:00"],
    }))

    assert mapped[["代码", "行业板块", "Nasdaq细分行业"]].to_dict("records") == [{"代码": "AAA", "行业板块": "Computer Software", "Nasdaq细分行业": "Computer Software"}]
    assert unmapped.loc[0, "原因"] == "Nasdaq 当前快照没有细分行业，无法构造细粒度研究板块"
