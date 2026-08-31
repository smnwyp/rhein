"""Run one saved DSL strategy over each installed A-share industry group.

Jobs are deliberately executed serially.  Each group uses the normal durable
job/worker pipeline, so an interrupted batch can resume without recalculating
symbols already written for that group's job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpha_agent.backtest_history import JsonBacktestHistory
from alpha_agent.backtest_jobs import BacktestJob, JsonBacktestJobStore
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.strategy_library import JsonStrategyLibrary
from rhein.backtest_worker import run_job
from rhein.data.a_share_industry import CLASSIFICATION_STANDARD, GROUP_METADATA_FILENAME, GROUP_ROOT_NAME
from rhein.data.discovery import input_files
from rhein.ui.history_paths import same_data_scope
from alpha_agent.backtest_history import strategy_fingerprint


def industry_groups(root: Path) -> list[tuple[str, Path]]:
    groups = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            metadata = json.loads((folder / GROUP_METADATA_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("classification_source") == CLASSIFICATION_STANDARD and metadata.get("industry"):
            groups.append((str(metadata["industry"]), folder))
    if not groups:
        raise ValueError("没有找到可用的 A 股行业分组；请先运行 build_a_share_industry_groups.py")
    return groups


def has_completed_run(history: JsonBacktestHistory, *, strategy_id: UUID, fingerprint: str, data_path: Path) -> bool:
    return any(
        run.strategy_fingerprint == fingerprint
        and same_data_scope(run.data_path, data_path, project_root=PROJECT_ROOT)
        for run in history.list_summaries_for_strategy(strategy_id)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序回测已保存策略的全部 A 股行业分组")
    parser.add_argument("--strategy-id", required=True, help="config/saved_strategies.json 中的策略 UUID")
    parser.add_argument("--groups-root", default=f"data/a_share_ohlcv/{GROUP_ROOT_NAME}")
    parser.add_argument("--rerun", action="store_true", help="即使已有同策略、同分组、同 DSL 的结果也新增一次回测")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy_id = UUID(args.strategy_id)
    library = JsonStrategyLibrary(PROJECT_ROOT / "config" / "saved_strategies.json")
    saved = next((item for item in library.list() if item.strategy_id == strategy_id), None)
    if saved is None or saved.strategy is None:
        raise ValueError("找不到已解析的已保存策略")
    timed_strategy = TimedStrategyDefinition.model_validate(saved.strategy.model_dump(mode="json"))
    if timed_strategy.schema_version not in {"0.2", "0.3"}:
        raise ValueError(f"当前批量回测仅支持时序 DSL v0.2/v0.3，实际为 v{timed_strategy.schema_version}")
    fingerprint = strategy_fingerprint(timed_strategy.model_dump_json())
    groups = industry_groups((PROJECT_ROOT / args.groups_root).resolve())
    if args.dry_run:
        print(f"策略：{saved.strategy_name} · DSL v{timed_strategy.schema_version} · {len(groups)} 个行业")
        for index, (industry, folder) in enumerate(groups, start=1):
            print(f"{index:02d}. {industry} · {len(input_files(folder))} 个标的 · {folder}")
        return

    jobs = JsonBacktestJobStore(PROJECT_ROOT / "config" / "backtest_jobs")
    history = JsonBacktestHistory(PROJECT_ROOT / "config" / "saved_backtest_results.json")
    for index, (industry, folder) in enumerate(groups, start=1):
        path = folder.resolve()
        if not args.rerun and has_completed_run(history, strategy_id=strategy_id, fingerprint=fingerprint, data_path=path):
            print(f"[{index:02d}/{len(groups):02d}] 跳过（已有同 DSL 结果）：{industry}", flush=True)
            continue
        job = jobs.find_resumable(strategy_id=strategy_id, strategy_fingerprint=fingerprint, data_path=str(path))
        if job is None:
            job = jobs.create(BacktestJob(
                status="queued", strategy_id=strategy_id, strategy_name=saved.strategy_name,
                strategy_fingerprint=fingerprint, strategy_payload=timed_strategy.model_dump(mode="json"),
                group_label=f"A 股行业 · {industry}", data_path=str(path), total_sources=len(input_files(path)),
                settings={"initial_capital": 10_000.0, "compound": True, "cost_bps": 0.0,
                          "runner": f"DSL v{timed_strategy.schema_version} 分组回测"},
            ))
        print(f"[{index:02d}/{len(groups):02d}] 运行：{industry} · {job.total_sources} 个标的 · job={job.job_id}", flush=True)
        run_job(project_root=PROJECT_ROOT, job_id=job.job_id)
        completed = jobs.load(job.job_id)
        if completed.status != "completed":
            raise RuntimeError(f"{industry} 回测未完成：{completed.failure_reason or completed.status}")
        print(f"[{index:02d}/{len(groups):02d}] 完成：{industry} · 失败标的 {completed.failed_sources}", flush=True)


if __name__ == "__main__":
    main()
