from pathlib import Path
from uuid import uuid4

import pandas as pd

from alpha_agent.backtest_history import JsonBacktestHistory
from alpha_agent.backtest_jobs import BacktestJob, JsonBacktestJobStore
from rhein.backtest_worker import run_job


def test_job_store_keeps_state_small_and_results_sharded(tmp_path: Path) -> None:
    store = JsonBacktestJobStore(tmp_path / "jobs")
    strategy_id = uuid4()
    job = store.create(BacktestJob(
        status="queued", strategy_id=strategy_id, strategy_name="测试策略",
        strategy_fingerprint="fingerprint", strategy_payload={"schema_version": "0.3"},
        group_label="全部 A 股", data_path="/market/a_share", settings={"compound": True}, total_sources=2,
    ))

    store.write_result(job.job_id, "/market/a_share/SH600001.parquet", kpi={"标的": "SH600001"}, trades=[], error=None)
    updated = store.save(job.model_copy(update={"status": "running", "completed_sources": 1, "current_symbol": "SH600002"}))

    assert store.result_exists(job.job_id, "/market/a_share/SH600001.parquet")
    assert len(store.read_results(job.job_id)) == 1
    assert store.find_resumable(strategy_id=strategy_id, strategy_fingerprint="fingerprint", data_path="/market/a_share") == updated
    assert updated.progress_fraction == .5


def test_detached_worker_runs_a_parquet_job_and_persists_a_saved_run(tmp_path: Path) -> None:
    data = tmp_path / "market"
    data.mkdir()
    pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=4, freq="B"),
        "Open": [10.0, 11.0, 12.0, 13.0], "High": [11.0, 12.0, 13.0, 14.0],
        "Low": [9.0, 10.0, 11.0, 12.0], "Close": [10.0, 11.0, 12.0, 13.0], "Volume": [100.0] * 4,
    }).to_parquet(data / "SH600001.parquet", index=False)
    payload = {
        "schema_version": "0.3", "strategy_name": "后台测试", "symbol": "TEST", "frequency": "1d",
        "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "scalar", "value": 0}}, "constraints": []},
        "entry": {"mode": "fixed", "active_day": {"start_offset_days": 0, "end_offset_days": 0}, "execution": "close"},
        "exit_rules": [{"rule_id": "expiry", "priority": 1, "active_days": {"start_offset_days": 1, "end_offset_days": 1}, "kind": "forced_close", "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    }
    store = JsonBacktestJobStore(tmp_path / "config" / "backtest_jobs")
    job = store.create(BacktestJob(
        status="queued", strategy_id=uuid4(), strategy_name="后台测试", strategy_fingerprint="fingerprint",
        strategy_payload=payload, group_label="测试组", data_path=str(data),
        settings={"initial_capital": 10_000.0, "compound": True, "cost_bps": 0.0}, total_sources=1,
    ))

    run_job(project_root=tmp_path, job_id=job.job_id)

    completed = store.load(job.job_id)
    assert completed.status == "completed"
    assert completed.completed_sources == 1
    assert completed.run_id is not None
    saved = JsonBacktestHistory(tmp_path / "config" / "saved_backtest_results.json").load(completed.run_id)
    assert saved.input_file_count == 1
    assert saved.kpis[0]["标的"] == "SH600001"
