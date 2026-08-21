"""Resumable orchestration; all financial evidence remains deterministic."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from alpha_agent.backtest_history import SavedBacktestRun
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.errors import AlphaAgentError

from .audit import attach_replay_audits
from .charts import render_trade_chart
from .client import OpenAIResearchClient, ResearchModelFailure
from .definition import ResearchSkillDefinition
from .evidence import build_statistical_evidence, select_visual_samples
from .history import JsonResearchReportHistory
from .models import ResearchSynthesis, SavedResearchReport, TokenUsage


class ResearchExecutionBlocked(AlphaAgentError):
    code = "research_execution_blocked"


@dataclass(frozen=True)
class ResearchProgress:
    """One user-visible state update; it contains no financial calculation."""

    stage: Literal["evidence", "visual", "synthesis", "complete"]
    completed: int
    total: int
    message: str


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(input_tokens=left.input_tokens + right.input_tokens, output_tokens=left.output_tokens + right.output_tokens)


def _source_paths(run: SavedBacktestRun, resolve_path: Callable[[str], Path | None]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in run.kpis:
        symbol, source = row.get("标的"), row.get("源文件")
        if isinstance(symbol, str) and isinstance(source, str):
            resolved = resolve_path(source)
            if resolved is not None:
                paths[symbol.upper()] = resolved
    return paths


def _allowed_dsl_paths(report: SavedResearchReport) -> list[str]:
    if report.evidence is None:
        return []
    paths: set[str] = set()
    for trade in report.evidence.trades:
        for audit in (trade.entry_audit, trade.exit_audit):
            if isinstance(audit, dict) and isinstance(audit.get("rows"), list):
                paths.update(str(row["dsl_path"]) for row in audit["rows"] if isinstance(row, dict) and isinstance(row.get("dsl_path"), str))
    return sorted(paths)


def _synthesis_evidence(report: SavedResearchReport) -> dict[str, object]:
    """Keep the synthesis input evidence-bound without sending every audit row.

    The full deterministic evidence stays in the saved artifact.  The model
    receives complete cohort/distribution summaries plus only the selected
    chart trades' ledger context; DSL paths are passed separately for
    experiment-card validation.
    """
    if report.evidence is None:
        return {}
    selected_ids = {sample.trade_id for sample in report.samples}
    selected_trades = [
        {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "signal_date": trade.signal_date,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "return_pct": trade.return_pct,
            "days_held": trade.days_held,
            "exit_reason": trade.exit_reason,
            "period": trade.period,
        }
        for trade in report.evidence.trades
        if trade.trade_id in selected_ids
    ]
    return {
        "trade_count": report.evidence.trade_count,
        "positive_return_q75": report.evidence.positive_return_q75,
        "cohorts": {key: value.model_dump(mode="json") for key, value in report.evidence.cohorts.items()},
        "distributions": [item.model_dump(mode="json") for item in report.evidence.distributions],
        "source_coverage": report.evidence.source_coverage.model_dump(mode="json"),
        "selected_trade_context": selected_trades,
    }


class ResearchSkillService:
    def __init__(
        self, *, history: JsonResearchReportHistory, definition: ResearchSkillDefinition,
        load_ohlc: Callable[[Path], pd.DataFrame], resolve_path: Callable[[str], Path | None],
        model_client: OpenAIResearchClient | None, progress: Callable[[ResearchProgress], None] | None = None,
        market_index: pd.DataFrame | None = None,
    ) -> None:
        self._history, self._definition = history, definition
        self._load_ohlc, self._resolve_path = load_ohlc, resolve_path
        self._model_client, self._progress, self._market_index = model_client, progress, market_index

    def start_or_resume(self, *, run: SavedBacktestRun, strategy: TimedStrategyDefinition, research_focus: str) -> SavedResearchReport:
        if run.schema_version != "0.3" or strategy.schema_version != "0.3":
            raise ResearchExecutionBlocked("research skill only supports saved DSL v0.3 runs")
        if run.strategy_fingerprint != self._fingerprint(strategy):
            raise ResearchExecutionBlocked("saved run fingerprint does not match the currently loaded strategy")
        proposed = SavedResearchReport.new(
            run_id=run.run_id, strategy_id=run.strategy_id, strategy_fingerprint=run.strategy_fingerprint,
            skill_id=self._definition.skill_id, skill_version=self._definition.version, research_focus=research_focus,
        )
        report = self._history.find_by_input(proposed.input_fingerprint) or proposed
        if report.status == "completed":
            return report
        if report.evidence is None or (
            report.evidence.trade_count and report.evidence.source_coverage.readable_trade_count == 0
        ):
            self._emit_progress(
                stage="evidence", completed=0, total=len(run.trades),
                message="正在从已保存交易账本计算全量确定性证据…",
            )
            evidence = build_statistical_evidence(run.trades)
            evidence = attach_replay_audits(
                evidence, strategy=strategy, source_paths=_source_paths(run, self._resolve_path),
                load_ohlc=self._load_ohlc, market_index=self._market_index,
                progress=lambda completed, total, symbol: self._emit_progress(
                    stage="evidence", completed=completed, total=total,
                    message=f"正在重放 DSL 入/出场审计：{completed}/{total} 笔（{symbol}）",
                ),
            )
            report = report.evolve(status="mapping", failure_reason=None, evidence=evidence, samples=select_visual_samples(evidence, target_count=self._definition.visual_sample_target))
            self._history.save(report)
            self._emit_progress(
                stage="evidence", completed=len(evidence.trades), total=len(evidence.trades),
                message="全量统计与 DSL 审计已保存。",
            )
            if evidence.trade_count and evidence.source_coverage.readable_trade_count == 0:
                report = report.evolve(failure_reason="No source OHLC could be read; deterministic coverage has been saved and can be resumed after data is restored.")
                self._history.save(report)
                raise ResearchExecutionBlocked(report.failure_reason)
        if self._model_client is None:
            report = report.evolve(status="mapping", failure_reason="OPENAI_API_KEY is not configured; deterministic evidence has been saved and can be resumed.")
            self._history.save(report)
            raise ResearchExecutionBlocked(report.failure_reason)
        report = self._review_visual_batches(report)
        if report.synthesis is None:
            self._emit_progress(stage="synthesis", completed=0, total=1, message="正在综合已保存的统计证据和逐图观察…")
            report = report.evolve(status="synthesizing", failure_reason=None)
            self._history.save(report)
            try:
                synthesis, usage = self._model_client.synthesize(
                    evidence=_synthesis_evidence(report), observations=report.observations,
                    research_focus=report.research_focus, allowed_dsl_paths=_allowed_dsl_paths(report),
                )
                self._validate_synthesis(synthesis, report)
            except ResearchModelFailure as error:
                report = report.evolve(status="synthesizing", failure_reason=error.message)
                self._history.save(report)
                raise
            report = report.evolve(
                status="completed", synthesis=synthesis, usage=_add_usage(report.usage, usage),
                model_snapshots=[*report.model_snapshots, {
                    "stage": "synthesis", "model": self._definition.synthesis_model,
                    "output": synthesis.model_dump(mode="json"), "usage": usage.model_dump(mode="json"),
                }],
            )
            self._history.save(report)
            self._emit_progress(stage="complete", completed=1, total=1, message="复盘报告已保存。")
        return report

    def _review_visual_batches(self, report: SavedResearchReport) -> SavedResearchReport:
        assert report.evidence is not None
        by_id = {trade.trade_id: trade for trade in report.evidence.trades}
        sample_updates = {sample.trade_id: sample for sample in report.samples}
        completed = set(report.completed_batch_numbers)
        total_batches = (len(report.samples) + self._definition.visual_batch_size - 1) // self._definition.visual_batch_size
        for batch_number, start in enumerate(range(0, len(report.samples), self._definition.visual_batch_size)):
            if batch_number in completed:
                continue
            self._emit_progress(
                stage="visual", completed=len(completed), total=total_batches,
                message=f"正在准备图表批次 {batch_number + 1}/{total_batches}…",
            )
            samples = report.samples[start:start + self._definition.visual_batch_size]
            charts = []
            for sample in samples:
                trade = by_id[sample.trade_id]
                if not trade.source_path:
                    continue
                try:
                    chart = render_trade_chart(trade, self._load_ohlc(Path(trade.source_path)))
                    charts.append(chart)
                    sample_updates[sample.trade_id] = sample.model_copy(update={"chart_checksum": chart.checksum, "chart_available": True})
                except Exception:
                    sample_updates[sample.trade_id] = sample.model_copy(update={"chart_checksum": None, "chart_available": False})
            report = report.evolve(samples=[sample_updates[item.trade_id] for item in report.samples])
            self._history.save(report)
            if charts:
                self._emit_progress(
                    stage="visual", completed=len(completed), total=total_batches,
                    message=f"正在观察图表批次 {batch_number + 1}/{total_batches}…",
                )
                try:
                    cached = self._history.observations_by_checksum({chart.checksum for chart in charts})
                    observations = [cached[chart.checksum] for chart in charts if chart.checksum in cached]
                    missing_charts = [chart for chart in charts if chart.checksum not in cached]
                    usage = TokenUsage()
                    cache_hit_only = not missing_charts
                    if missing_charts:
                        observations, usage = self._model_client.observe_batch(missing_charts, research_focus=report.research_focus)  # type: ignore[union-attr]
                        observations = [*cached.values(), *observations]
                    expected = {chart.trade_id: chart.checksum for chart in charts}
                    if len(observations) != len(expected) or {item.trade_id: item.chart_checksum for item in observations} != expected:
                        raise ResearchModelFailure("visual observation schema did not cover exactly the rendered charts")
                except ResearchModelFailure as error:
                    report = report.evolve(status="mapping", failure_reason=error.message)
                    self._history.save(report)
                    raise
                prior = {item.trade_id: item for item in report.observations}
                prior.update({item.trade_id: item for item in observations})
                report = report.evolve(
                    observations=[prior[key] for key in sorted(prior)], usage=_add_usage(report.usage, usage),
                    model_snapshots=[*report.model_snapshots, {
                        "stage": "visual_cache" if cache_hit_only else "visual", "batch_number": batch_number,
                        "model": None if cache_hit_only else self._definition.observation_model,
                        "output": [item.model_dump(mode="json") for item in observations],
                        "usage": usage.model_dump(mode="json"),
                    }],
                )
            if charts:
                completed.add(batch_number)
            report = report.evolve(status="mapping", completed_batch_numbers=sorted(completed), failure_reason=None)
            self._history.save(report)
            self._emit_progress(
                stage="visual", completed=len(completed), total=total_batches,
                message=f"图表批次已保存：{len(completed)}/{total_batches}。",
            )
        if report.samples and not any(sample.chart_available for sample in report.samples):
            report = report.evolve(status="mapping", failure_reason="All selected trade charts are unavailable; no visual-model call was made.")
            self._history.save(report)
            raise ResearchExecutionBlocked(report.failure_reason)
        return report

    @staticmethod
    def _fingerprint(strategy: TimedStrategyDefinition) -> str:
        from alpha_agent.backtest_history import strategy_fingerprint
        return strategy_fingerprint(strategy.model_dump_json())

    @staticmethod
    def _validate_synthesis(synthesis: ResearchSynthesis, report: SavedResearchReport) -> None:
        allowed_paths = set(_allowed_dsl_paths(report))
        selected_trade_ids = {item.trade_id for item in report.samples}
        finding_groups = (
            synthesis.facts,
            synthesis.visual_observations,
            synthesis.hypotheses_to_verify,
        )
        findings = [item for group in finding_groups for item in group]
        if (
            len({item.finding_id for item in findings}) != len(findings)
            or any(not item.evidence_trade_ids or not set(item.evidence_trade_ids).issubset(selected_trade_ids) for item in findings)
        ):
            raise ResearchModelFailure("synthesis finding references must identify selected saved evidence trades")
        if len(synthesis.experiment_cards) > 3 or any(card.dsl_path not in allowed_paths or not set(card.evidence_trade_ids).issubset(selected_trade_ids) for card in synthesis.experiment_cards):
            raise ResearchModelFailure("synthesis contains an experiment card outside the saved DSL/evidence boundary")

    def _emit_progress(
        self, *, stage: Literal["evidence", "visual", "synthesis", "complete"],
        completed: int, total: int, message: str,
    ) -> None:
        if self._progress is not None:
            self._progress(ResearchProgress(stage=stage, completed=completed, total=total, message=message))
