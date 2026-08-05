import pandas as pd

from alpha_agent.research.reporting import aggregate_gross_pnl, presentation_kpis


def test_aggregate_gross_pnl_is_safe_when_a_zero_trade_run_has_no_optional_columns():
    kpis = pd.DataFrame([{"标的": "AAPL", "n_trades": 0}])
    assert aggregate_gross_pnl(kpis, pd.DataFrame()) == (0.0, 0.0)


def test_aggregate_gross_pnl_prefers_trade_ledger_for_legacy_kpi_rows():
    kpis = pd.DataFrame([{"标的": "AAPL", "n_trades": 2}, {"标的": "MSFT", "n_trades": 0}])
    trades = pd.DataFrame([
        {"symbol": "AAPL", "pnl_eur": 20.0},
        {"symbol": "AAPL", "pnl_eur": -5.0},
        {"symbol": "MSFT", "pnl_eur": 99.0},
    ])
    assert aggregate_gross_pnl(kpis, trades) == (20.0, 5.0)


def test_presentation_kpis_keeps_old_records_sortable_when_profit_factor_is_missing():
    legacy = pd.DataFrame([{
        "标的": "AAPL",
        "n_trades": 2,
        "gross_profit": 20.0,
        "gross_loss": 5.0,
        "cumulative_return_pct": 3.0,
    }])

    display = presentation_kpis(legacy)

    assert display.loc[0, "profit_factor"] == 4.0
    assert display.loc[0, "n_trades"] == 2
    assert "sharpe_ratio" in display
    assert display.sort_values("profit_factor").loc[0, "标的"] == "AAPL"


def test_presentation_kpis_marks_an_unrecoverable_old_profit_factor_unavailable():
    display = presentation_kpis(pd.DataFrame([{"标的": "MSFT", "n_trades": 1}]))

    assert pd.isna(display.loc[0, "profit_factor"])
    assert display.loc[0, "n_trades"] == 1
