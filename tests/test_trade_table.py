import pandas as pd

from rhein.ui.trade_table import build_trade_ranking


def test_trade_ranking_is_one_row_per_trade_and_sorts_without_mutating_source():
    trades = pd.DataFrame([
        {"symbol": "BETA", "signal": "2024-01-02", "entry": "2024-01-03", "exit": "2024-01-08", "days_held": 3, "entry_px": 10.0, "exit_px": 12.0, "ret_pct": 20.0, "pnl_eur": 200.0, "reason": "均线下穿"},
        {"symbol": "ALFA", "signal": "2024-01-05", "entry": "2024-01-08", "exit": "2024-01-09", "days_held": 1, "entry_px": 20.0, "exit_px": 19.0, "ret_pct": -5.0, "pnl_eur": -50.0, "reason": "止损"},
        {"symbol": "BETA", "signal": "2024-01-10", "entry": "2024-01-11", "exit": "2024-01-16", "days_held": 3, "entry_px": 12.0, "exit_px": 13.2, "ret_pct": 10.0, "pnl_eur": 100.0, "reason": "均线下穿"},
    ])
    original = trades.copy(deep=True)

    ranked, display = build_trade_ranking(trades, metric="ret_pct")

    assert len(ranked) == len(trades) == 3
    assert display["标的"].tolist() == ["BETA", "BETA", "ALFA"]
    assert display["交易排名"].tolist() == [1, 2, 3]
    assert display["单笔收益率 (%)"].tolist() == [20.0, 10.0, -5.0]
    assert ranked["_trade_row_id"].tolist() == [0, 2, 1]
    pd.testing.assert_frame_equal(trades, original)


def test_trade_ranking_fills_missing_legacy_fields_without_dropping_trade():
    trades = pd.DataFrame([{"symbol": "ALFA", "entry": "2024-01-08", "ret_pct": 1.5}])

    ranked, display = build_trade_ranking(trades, metric="entry")

    assert len(ranked) == 1
    assert display.loc[0, "标的"] == "ALFA"
    assert pd.isna(display.loc[0, "出场日期"])
    assert pd.isna(display.loc[0, "持仓天数"])
