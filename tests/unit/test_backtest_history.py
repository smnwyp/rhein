from uuid import uuid4

import pandas as pd

from alpha_agent.backtest_history import JsonBacktestHistory, SavedBacktestRun, dataframe_records, strategy_fingerprint


def run(strategy_id, group_label: str = "组 A") -> SavedBacktestRun:
    return SavedBacktestRun(
        strategy_id=strategy_id,
        strategy_fingerprint=strategy_fingerprint('{"schema_version":"0.3"}'),
        schema_version="0.3",
        group_label=group_label,
        data_path=f"/data/{group_label}",
        input_file_count=2,
        settings={"initial_capital": 10_000.0, "compound": True, "cost_bps": 0.0},
        kpis=dataframe_records(pd.DataFrame([{"标的": "AAPL", "n_trades": 1, "profit_factor": None}])),
        trades=dataframe_records(pd.DataFrame([{"symbol": "AAPL", "signal": "2025-01-01", "ret_pct": 1.2}])),
    )


def test_backtest_history_persists_results_by_strategy_identity(tmp_path):
    history = JsonBacktestHistory(tmp_path / "saved_backtest_results.json")
    strategy_a, strategy_b = uuid4(), uuid4()
    run_a, run_b = history.save(run(strategy_a)), history.save(run(strategy_b, "组 B"))

    restored = JsonBacktestHistory(tmp_path / "saved_backtest_results.json").list_for_strategy(strategy_a)
    assert restored == [run_a]
    assert restored[0].kpis[0]["标的"] == "AAPL"
    assert restored[0].trades[0]["ret_pct"] == 1.2

    history.delete_for_strategy(strategy_a)
    assert history.list_for_strategy(strategy_a) == []
    assert history.list_for_strategy(strategy_b) == [run_b]
