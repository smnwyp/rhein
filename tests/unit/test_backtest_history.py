import json
from uuid import uuid4

import pandas as pd

from alpha_agent.backtest_history import JsonBacktestHistory, SavedBacktestRun, dataframe_records, recalculated_trade_level_kpis, strategy_fingerprint


def run(strategy_id, group_label: str = "组 A") -> SavedBacktestRun:
    return SavedBacktestRun(
        strategy_id=strategy_id,
        strategy_name="测试策略",
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
    assert restored[0].strategy_name == "测试策略"
    assert restored[0].kpis[0]["标的"] == "AAPL"
    assert restored[0].trades[0]["ret_pct"] == 1.2

    history.delete_for_strategy(strategy_a)
    assert history.list_for_strategy(strategy_a) == []
    assert history.list_for_strategy(strategy_b) == [run_b]


def test_backtest_history_deletes_only_the_requested_run(tmp_path):
    history = JsonBacktestHistory(tmp_path / "saved_backtest_results.json")
    strategy_id = uuid4()
    first, second = history.save(run(strategy_id)), history.save(run(strategy_id))

    history.delete(second.run_id)

    assert history.list_for_strategy(strategy_id) == [first]


def test_old_backtest_record_without_a_strategy_name_remains_readable(tmp_path):
    history = JsonBacktestHistory(tmp_path / "saved_backtest_results.json")
    strategy_id = uuid4()
    saved = history.save(run(strategy_id)).model_dump(mode="json")
    saved.pop("strategy_name")
    (tmp_path / "saved_backtest_results.json").write_text(json.dumps([saved]), encoding="utf-8")

    restored = history.list_for_strategy(strategy_id)

    assert restored[0].strategy_name == "未命名策略（旧记录）"


def test_historical_view_recalculates_trade_kpis_without_mutating_saved_records() -> None:
    stored_kpis = [{"标的": "LGND", "max_drawdown_pct": 0.0, "annualized_return_pct": 2.0}]
    stored_trades = [
        {"symbol": "LGND", "signal": "2020-01-01", "entry": "2020-01-01", "exit": "2020-01-02", "ret_pct": -5.0, "pnl_eur": -500.0, "days_held": 1},
        {"symbol": "LGND", "signal": "2020-01-03", "entry": "2020-01-03", "exit": "2020-01-04", "ret_pct": 10.0, "pnl_eur": 950.0, "days_held": 1},
    ]

    refreshed = recalculated_trade_level_kpis(stored_kpis, stored_trades, {"initial_capital": 10_000.0, "compound": True})

    assert refreshed.loc[0, "max_drawdown_pct"] == -5.0
    assert refreshed.loc[0, "annualized_return_pct"] == 2.0
    assert stored_kpis[0]["max_drawdown_pct"] == 0.0
