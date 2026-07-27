from __future__ import annotations

import pandas as pd

from rhein.ui.result_runner import annualized_sharpe


def test_annualized_sharpe_marks_positions_to_market_and_includes_exit_return() -> None:
    ohlc = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "Close": [100.0, 110.0, 121.0, 121.0],
    })
    trades = pd.DataFrame([{
        "entry": "2024-01-02", "exit": "2024-01-04", "ret_pct": 21.0,
    }])

    sharpe = annualized_sharpe(ohlc, trades)

    assert sharpe is not None
    assert sharpe > 0


def test_annualized_sharpe_is_not_reported_without_a_completed_trade() -> None:
    ohlc = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "Close": [100.0, 100.0, 100.0],
    })

    assert annualized_sharpe(ohlc, pd.DataFrame()) is None
