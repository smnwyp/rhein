"""Detached, resumable executor for a saved DSL backtest job.

Run with ``python -m rhein.backtest_worker --project-root . --job-id <uuid>``.
It has no Streamlit dependency and persists one source artifact at a time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

import pandas as pd


def _project_imports(project_root: Path) -> None:
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def run_job(*, project_root: Path, job_id: UUID) -> None:
    _project_imports(project_root)
    from alpha_agent.backtest_history import JsonBacktestHistory, SavedBacktestRun, dataframe_records
    from alpha_agent.backtest_jobs import JsonBacktestJobStore
    from alpha_agent.domain.sequence import TimedStrategyDefinition
    from alpha_agent.research.legacy_adapter import compile_timed_strategy
    from alpha_agent.research.market_index import load_market_index_for_files, required_market_index_symbols
    from alpha_agent.research.v03_engine import run_v03_backtest
    from rhein.backtest import run_backtest as run_legacy_backtest
    from rhein.backtest_metrics import annualized_sharpe
    from rhein.data.discovery import input_files
    from rhein.data.ohlc import load_ohlc
    from rhein.ui.history_paths import portable_data_path

    store = JsonBacktestJobStore(project_root / "config" / "backtest_jobs")
    job = store.load(job_id)
    if job.status == "completed":
        return
    job = store.save(job.model_copy(update={"status": "running", "failure_reason": None}))
    try:
        strategy = TimedStrategyDefinition.model_validate(job.strategy_payload)
        paths = input_files(Path(job.data_path))
        capital = float(job.settings.get("initial_capital", 10_000.0))
        compound = job.settings.get("compound", True) is not False
        cost_bps = float(job.settings.get("cost_bps", 0.0))
        market_index = None
        if strategy.schema_version == "0.3":
            runner, params = run_v03_backtest, {"strategy": strategy, "cost_bps": cost_bps}
            if strategy.data_requirement == "daily_ohlcv_with_market_index":
                index_symbols = required_market_index_symbols(strategy.model_dump(mode="json"))
                if len(index_symbols) != 1:
                    raise ValueError("current daily worker requires exactly one declared market-index source")
                market_index = load_market_index_for_files(
                    paths, symbol=next(iter(index_symbols)), cache_directory=project_root / ".cache" / "market_indices",
                )
                params["market_index"] = market_index
        else:
            runner, params = run_legacy_backtest, compile_timed_strategy(strategy, approximate_intraday_with_daily_low=True)
            params["cost_bps"] = cost_bps

        completed = failed = 0
        for index, path in enumerate(paths, start=1):
            source_path = str(path)
            if store.result_exists(job_id, source_path):
                completed += 1
                continue
            job = store.save(job.model_copy(update={"current_symbol": path.stem.upper(), "total_sources": len(paths), "completed_sources": completed, "failed_sources": failed}))
            try:
                ohlc = load_ohlc(path)
                trade_frame, stats = runner(ohlc, capital=capital, compound=compound, **params)
                calendar_days = max(int((ohlc["Date"].iloc[-1] - ohlc["Date"].iloc[0]).days), 1)
                equity_multiple = stats["final_equity"] / stats["initial_capital"]
                kpi = {
                    "标的": path.stem.upper(), "源文件": source_path,
                    "数据覆盖天数": calendar_days,
                    "annualized_return_pct": (equity_multiple ** (365.25 / calendar_days) - 1) * 100,
                    "sharpe_ratio": annualized_sharpe(ohlc, trade_frame), **stats,
                }
                if not trade_frame.empty:
                    trade_frame = trade_frame.copy()
                    trade_frame.insert(0, "symbol", path.stem.upper())
                store.write_result(job_id, source_path, kpi=kpi, trades=dataframe_records(trade_frame), error=None)
            except Exception as error:
                failed += 1
                store.write_result(job_id, source_path, kpi=None, trades=[], error=f"{type(error).__name__}: {error}")
            completed += 1
            job = store.save(job.model_copy(update={"total_sources": len(paths), "completed_sources": completed, "failed_sources": failed, "current_symbol": path.stem.upper()}))

        results = store.read_results(job_id)
        kpis = [result["kpi"] for result in results if isinstance(result.get("kpi"), dict)]
        trades = [trade for result in results if isinstance(result.get("trades"), list) for trade in result["trades"] if isinstance(trade, dict)]
        if not kpis:
            raise ValueError("all input files failed; no empty backtest result was saved")
        history = JsonBacktestHistory(project_root / "config" / "saved_backtest_results.json")
        saved_run = history.save(SavedBacktestRun(
            strategy_id=job.strategy_id, strategy_name=job.strategy_name,
            strategy_fingerprint=job.strategy_fingerprint, schema_version=strategy.schema_version,
            group_label=job.group_label, data_path=portable_data_path(job.data_path, project_root=project_root),
            input_file_count=len(paths), settings=job.settings,
            kpis=kpis, trades=trades,
        ))
        store.save(job.model_copy(update={
            "status": "completed", "completed_sources": completed, "failed_sources": failed,
            "current_symbol": None, "run_id": saved_run.run_id,
            "failure_reason": f"{failed} 个标的运行失败。" if failed else None,
        }))
    except Exception as error:
        store.save(job.model_copy(update={"status": "failed", "current_symbol": None, "failure_reason": f"{type(error).__name__}: {error}"}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted Rhein backtest job")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run_job(project_root=Path(args.project_root).resolve(), job_id=UUID(args.job_id))


if __name__ == "__main__":
    main()
