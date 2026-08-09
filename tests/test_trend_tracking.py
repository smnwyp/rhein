from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rhein.trend_tracking import (
    TrendTrackingError,
    TrendTrackingRun,
    aggregate_sector_daily,
    assert_scoring_is_causal,
    calibrate_threshold_grid,
    clean_stock_daily,
    fixed_holding_diagnostics,
    generate_sector_signals,
    normalize_industry_map,
    run_fixed_holding_backtest,
    score_all_sectors,
    run_sector_backtest,
)


def _stock_daily(days: int = 205) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    mapping: list[dict[str, str]] = []
    for sector_number, sector in enumerate(("软件", "医疗", "能源", "工业")):
        for number in range(20):
            symbol = f"{sector_number}{number:02d}"
            mapping.append({"代码": symbol, "行业板块": sector})
            close = 10.0 + number / 10
            for day, date in enumerate(dates):
                close *= 1.002 + (sector_number - 1) * 0.0003
                rows.append({
                    "Date": date, "Symbol": symbol, "Open": close * .998,
                    "High": close * 1.01, "Low": close * .99, "Close": close,
                    "Volume": 1_000 + day + number,
                })
    return pd.DataFrame(rows), pd.DataFrame(mapping)


def test_clean_and_aggregate_apply_explicit_prd_guards() -> None:
    daily, raw_mapping = _stock_daily()
    mapping = normalize_industry_map(raw_mapping)
    cleaned = clean_stock_daily(daily, mapping, adjustment_policy="adjusted")
    sector_daily = aggregate_sector_daily(cleaned)

    assert cleaned["上市序号"].min() >= 61
    assert (cleaned["Close"] >= 1).all()
    assert (cleaned["Volume"] > 0).all()
    assert (sector_daily["个股数"] == 20).all()
    assert {"板块涨跌幅", "上涨占比", "板块成交额", "大盘涨跌幅", "超额收益"} <= set(sector_daily)


def test_unknown_adjustment_policy_conservatively_excludes_large_price_jump() -> None:
    daily, raw_mapping = _stock_daily(days=65)
    daily.loc[(daily["Symbol"] == "000") & (daily["Date"] == daily["Date"].iloc[-1]), "Close"] *= 2
    mapping = normalize_industry_map(raw_mapping)
    conservative = clean_stock_daily(daily, mapping, adjustment_policy="unknown_conservative_drop_outliers")
    adjusted = clean_stock_daily(daily, mapping, adjustment_policy="adjusted")

    assert len(conservative) == len(adjusted) - 1


def test_scoring_cutoff_is_identical_with_or_without_future_rows() -> None:
    daily, raw_mapping = _stock_daily()
    sector_daily = aggregate_sector_daily(
        clean_stock_daily(daily, normalize_industry_map(raw_mapping), adjustment_policy="adjusted")
    )
    cutoff = sector_daily["Date"].sort_values().iloc[-3]
    assert_scoring_is_causal(sector_daily, cutoff)
    scored = score_all_sectors(sector_daily)

    assert scored.loc[scored["Date"] == cutoff, "萌芽分"].notna().all()
    assert scored.loc[scored["Date"] == cutoff, "当日排名分位"].notna().all()


def test_signal_state_machine_requires_three_days_and_exits_on_mechanism_break() -> None:
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    scored = pd.DataFrame({
        "Date": dates,
        "Sector": "软件",
        "萌芽分": [40, 61, 62, 63, 64, 64, 64, 64, 64, 64],
        "广度扩散分": [1, 10, 10, 10, 10, 0, 0, 0, 0, 0],
        "当日排名分位": [.2] * 10,
        "超额收益": [.01] * 10,
        "暴涨上限": [.03] * 10,
        "上涨占比": [.5] * 10,
    })
    signals = generate_sector_signals(scored, entry_threshold=60, exit_threshold=50)

    assert signals.loc[signals["进场"], "Date"].tolist() == [dates[3]]
    assert signals.loc[signals["离场"], "Date"].tolist() == [dates[9]]
    assert signals.loc[signals["Date"] == dates[4], "状态"].item() == "持有中"
    assert "广度扩散分连续五日为零" in signals.loc[signals["离场"], "信号原因"].item()


def test_pulse_exit_is_stamped_after_next_day_breadth_decline_not_in_advance() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    scored = pd.DataFrame({
        "Date": dates,
        "Sector": "软件",
        "萌芽分": [61, 62, 63, 65, 65],
        "广度扩散分": [10] * 5,
        "当日排名分位": [.2] * 5,
        "超额收益": [.01, .01, .01, .08, .01],
        "暴涨上限": [.03] * 5,
        "上涨占比": [.5, .5, .5, .8, .4],
    })
    signals = generate_sector_signals(scored, entry_threshold=60, exit_threshold=50)

    assert signals.loc[signals["进场"], "Date"].tolist() == [dates[2]]
    assert signals.loc[signals["离场"], "Date"].tolist() == [dates[4]]
    assert signals.loc[signals["Date"] == dates[3], "状态"].item() == "持有中"
    assert "前日暴涨且今日广度下降" in signals.loc[signals["离场"], "信号原因"].item()


def test_industry_mapping_conflicts_are_diagnosed() -> None:
    with pytest.raises(TrendTrackingError) as caught:
        normalize_industry_map(pd.DataFrame({"代码": ["AAA", "AAA"], "行业板块": ["软件", "医疗"]}))

    assert caught.value.code == "industry_map_conflict"


def test_nasdaq_industry_map_builder_keeps_only_explicit_standard_sector_mappings() -> None:
    from scripts.build_industry_map import build_industry_map

    mapped, unmapped = build_industry_map(
        pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"]}),
        pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "sector": ["Technology", "Miscellaneous"],
            "industry": ["Computer Software", "Multi-Sector Companies"],
        }),
        "2026-08-09T00:00:00+00:00",
    )

    assert mapped[["代码", "行业板块"]].to_dict("records") == [{"代码": "AAA", "行业板块": "信息技术"}]
    assert set(unmapped["代码"]) == {"BBB", "CCC"}
    assert "未纳入十一大板块" in unmapped.loc[unmapped["代码"] == "BBB", "原因"].item()


def test_calibration_config_declares_candidates_but_keeps_locked_thresholds_empty() -> None:
    config = json.loads(Path("config/trend_tracking.json").read_text(encoding="utf-8"))

    assert config["calibration"]["entry_thresholds"] == [50, 55, 60, 65, 70, 75]
    assert config["calibration"]["exit_gaps"] == [5, 10, 15, 20]
    assert config["locked_thresholds"]["entry_threshold"] is None
    assert config["locked_thresholds"]["exit_threshold"] is None


def test_sector_backtest_executes_on_next_open_and_deducts_both_side_costs() -> None:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    rows = []
    for symbol, opens in {"AAA": [10, 10, 11, 12, 13, 14], "BBB": [20, 20, 22, 24, 26, 28]}.items():
        for date, open_ in zip(dates, opens):
            rows.append({
                "Date": date, "Symbol": symbol, "Sector": "软件", "Open": float(open_),
                "High": open_ + 1, "Low": open_ - 1, "Close": float(open_), "Volume": 100,
                "涨跌幅": 0.0,
            })
    signals = pd.DataFrame({"Date": [dates[0], dates[3]], "Sector": ["软件", "软件"], "进场": [True, False], "离场": [False, True]})
    result = run_sector_backtest(pd.DataFrame(rows), signals, run=TrendTrackingRun(60, 50, "adjusted", transaction_cost=.001))

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade.entry_date == dates[1]
    assert trade.exit_date == dates[4]
    assert trade.component_count == 2
    assert trade.net_return < ((13 / 10 + 26 / 20) / 2 - 1)
    assert {"equity", "active_sector_count"} <= set(result.equity_curve)
    assert {"equity"} <= set(result.market_benchmark)
    assert {"equity", "selected_sector_count"} <= set(result.momentum_benchmark)


def test_fixed_holding_diagnostic_exits_after_exact_number_of_sector_trading_days() -> None:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    rows = []
    for symbol, opens in {"AAA": [10, 10, 11, 12, 13, 14], "BBB": [20, 20, 22, 24, 26, 28]}.items():
        for date, open_ in zip(dates, opens):
            rows.append({
                "Date": date, "Symbol": symbol, "Sector": "软件", "Open": float(open_),
                "High": open_ + 1, "Low": open_ - 1, "Close": float(open_), "Volume": 100,
                "涨跌幅": 0.0,
            })
    candidates = pd.DataFrame({"Date": [dates[0]], "Sector": ["软件"], "进场": [True]})
    result = run_fixed_holding_backtest(
        pd.DataFrame(rows), candidates, holding_days=2, transaction_cost=.001
    )

    assert result.candidate_count == 1
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade.entry_date == dates[1]
    assert trade.exit_date == dates[3]
    assert trade.holding_trading_days == 2
    assert trade.exit_policy == "fixed_holding"


def test_calibration_grid_enumerates_the_24_train_only_candidates_without_locking_a_winner() -> None:
    daily, raw_mapping = _stock_daily()
    cleaned = clean_stock_daily(daily, normalize_industry_map(raw_mapping), adjustment_policy="adjusted")
    grid = calibrate_threshold_grid(
        cleaned,
        aggregate_sector_daily(cleaned),
        entry_thresholds=[50, 55, 60, 65, 70, 75],
        exit_gaps=[5, 10, 15, 20],
        adjustment_policy="adjusted",
    )

    assert len(grid) == 24
    assert set(grid["阈值差"]) == {5, 10, 15, 20}
    assert set(grid["进场阈值"]) == {50, 55, 60, 65, 70, 75}
    assert (grid["离场阈值"] == grid["进场阈值"] - grid["阈值差"]).all()
    assert {
        "校准段年化收益", "校准段最大回撤", "校准段信号次数", "校准段胜率",
        "校准段等权大盘年化收益", "校准段等权大盘最大回撤",
        "校准段朴素动量年化收益", "校准段朴素动量最大回撤",
    } <= set(grid)
    assert not any(column.startswith("检验段") and column != "检验段起始日（未使用）" for column in grid.columns)


def test_fixed_holding_diagnostics_enumerate_six_entries_times_three_periods_on_train_only() -> None:
    daily, raw_mapping = _stock_daily()
    cleaned = clean_stock_daily(daily, normalize_industry_map(raw_mapping), adjustment_policy="adjusted")
    diagnostics = fixed_holding_diagnostics(
        cleaned,
        aggregate_sector_daily(cleaned),
        entry_thresholds=[50, 55, 60, 65, 70, 75],
    )

    assert len(diagnostics) == 18
    assert set(diagnostics["进场阈值"]) == {50, 55, 60, 65, 70, 75}
    assert set(diagnostics["固定持有期（交易日）"]) == {20, 40, 60}
    assert set(diagnostics["离场规则"]) == {"固定持有期（不使用评分离场）"}
    assert not any(column.startswith("检验段") and column != "检验段起始日（未使用）" for column in diagnostics.columns)


def test_trend_tracking_page_is_statistical_research_without_trading_inputs() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("pages/2_Trend_Tracking.py")
    app.run(timeout=45)

    assert not app.exception
    labels = {widget.label for widget in app.text_input}
    assert "校准后的进场阈值（0–100）" not in labels
    assert "校准后的离场阈值（0–100）" not in labels
    assert any(widget.label == "运行统计检验研究" for widget in app.button)
    assert not any("网格" in widget.label or "回测" in widget.label for widget in app.button)
