import pandas as pd

from alpha_agent.research.reporting import aggregate_gross_pnl


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
