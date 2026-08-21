from pathlib import Path

from alpha_agent.backtest_checkpoint import JsonBacktestCheckpoint


def test_checkpoint_round_trip_and_key_isolation(tmp_path: Path):
    store = JsonBacktestCheckpoint(tmp_path / "checkpoint.json")
    store.save({"key": "run-a", "completed_sources": ["a.csv"], "kpis": [], "trades": []})
    assert store.load("run-a")["completed_sources"] == ["a.csv"]
    assert store.load("run-b") is None
    store.clear("run-a")
    assert store.load("run-a") is None
