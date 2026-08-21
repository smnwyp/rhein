"""Immutable contracts for the saved-backtest research skill.

The models intentionally keep financial calculations out of model output.  A
model may describe evidence that Python prepared, but cannot supply a KPI or a
new strategy parameter.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from alpha_agent.domain.indicators import DSLModel


ResearchStatus = Literal["preparing", "mapping", "synthesizing", "completed", "failed"]


class CohortSummary(DSLModel):
    trade_count: int = Field(ge=0)
    trade_ids: list[str] = Field(default_factory=list)
    winning_trade_count: int = Field(ge=0)
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    mean_days_held: float | None = None


class DistributionCell(DSLModel):
    dimension: Literal["exit_reason", "symbol", "period"]
    key: str = Field(min_length=1)
    trade_count: int = Field(ge=0)
    winning_trade_count: int = Field(ge=0)
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    mean_days_held: float | None = None
    evidence_sufficient: bool


class SourceCoverage(DSLModel):
    total_trades: int = Field(ge=0)
    readable_trade_count: int = Field(ge=0)
    unreadable_trade_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TradeEvidence(DSLModel):
    trade_id: str = Field(min_length=1)
    ledger_index: int = Field(ge=0)
    symbol: str = Field(min_length=1)
    signal_date: str = Field(min_length=1)
    entry_date: str = Field(min_length=1)
    exit_date: str = Field(min_length=1)
    return_pct: float
    pnl_eur: float | None = None
    days_held: int = Field(ge=0)
    exit_reason: str = Field(min_length=1)
    period: Literal["period_1", "period_2", "period_3"]
    source_path: str | None = None
    entry_audit: dict[str, object] | None = None
    exit_audit: dict[str, object] | None = None


class StatisticalEvidence(DSLModel):
    trade_count: int = Field(ge=0)
    positive_return_q75: float | None = None
    cohorts: dict[str, CohortSummary]
    distributions: list[DistributionCell]
    trades: list[TradeEvidence]
    source_coverage: SourceCoverage


class VisualSample(DSLModel):
    trade_id: str = Field(min_length=1)
    bucket: Literal["highest_return", "lowest_return", "fast_negative_exit", "stratified", "supplement"]
    order: int = Field(ge=0)
    chart_checksum: str | None = None
    chart_available: bool = False


class VisualObservation(DSLModel):
    trade_id: str = Field(min_length=1)
    chart_checksum: str = Field(min_length=1)
    entry_context: str = Field(min_length=1)
    post_entry_path: str = Field(min_length=1)
    exit_context: str = Field(min_length=1)
    display_indicator_observation: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


class ExperimentCard(DSLModel):
    """A non-executable, one-variable experiment proposal.

    There is deliberately no numeric value field.  The next bounded workflow
    may turn this into a draft only after schema/semantic validation and user
    confirmation.
    """
    title: str = Field(min_length=1, max_length=160)
    dsl_path: str = Field(min_length=1)
    variable: str = Field(min_length=1)
    direction: Literal["tighten", "loosen", "enable", "disable"]
    evaluation_metric: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    evidence_trade_ids: list[str] = Field(min_length=1)


class ResearchFinding(DSLModel):
    """One report statement and the saved trades that substantiate it.

    The text is model-authored, but references are constrained to the
    deterministically selected visual-review trades.  This gives the UI a
    stable, inspectable path from a claim back to its chart and DSL audit.
    """

    finding_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_trade_ids: list[str]


class ResearchSynthesis(DSLModel):
    # OpenAI strict structured outputs require every root property to be
    # listed in JSON Schema ``required``.  The model must therefore return
    # empty arrays explicitly when a category has no evidence.
    facts: list[ResearchFinding]
    visual_observations: list[ResearchFinding]
    hypotheses_to_verify: list[ResearchFinding]
    experiment_cards: list[ExperimentCard] = Field(max_length=3)

    @field_validator("facts", "visual_observations", "hypotheses_to_verify", mode="before")
    @classmethod
    def migrate_legacy_string_findings(cls, value: object, info: object) -> object:
        """Read v1 artifacts whose findings were plain strings.

        Old reports cannot acquire evidence references retroactively, so their
        lists intentionally remain empty and the UI labels them accordingly.
        """
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return value
        field_name = getattr(info, "field_name", "finding")
        return [
            {
                "finding_id": f"legacy-{field_name}-{index + 1}",
                "statement": item,
                "evidence_trade_ids": [],
            }
            for index, item in enumerate(value)
        ]


class TokenUsage(DSLModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class SavedResearchReport(DSLModel):
    report_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    strategy_id: UUID
    strategy_fingerprint: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    skill_version: str = Field(min_length=1)
    research_focus: str = Field(default="", max_length=2_000)
    input_fingerprint: str = Field(min_length=1)
    status: ResearchStatus = "preparing"
    failure_reason: str | None = None
    evidence: StatisticalEvidence | None = None
    samples: list[VisualSample] = Field(default_factory=list)
    observations: list[VisualObservation] = Field(default_factory=list)
    synthesis: ResearchSynthesis | None = None
    model_snapshots: list[dict[str, object]] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    completed_batch_numbers: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(
        cls, *, run_id: UUID, strategy_id: UUID, strategy_fingerprint: str,
        skill_id: str, skill_version: str, research_focus: str,
    ) -> "SavedResearchReport":
        from hashlib import sha256

        focus = research_focus.strip()
        fingerprint = sha256(
            f"{run_id}|{strategy_fingerprint}|{skill_id}|{skill_version}|{focus}".encode("utf-8")
        ).hexdigest()
        return cls(
            run_id=run_id, strategy_id=strategy_id, strategy_fingerprint=strategy_fingerprint,
            skill_id=skill_id, skill_version=skill_version, research_focus=focus,
            input_fingerprint=fingerprint,
        )

    def evolve(self, **changes: object) -> "SavedResearchReport":
        return self.model_copy(update={**changes, "updated_at": datetime.now(timezone.utc)})


class SavedResearchReportSummary(DSLModel):
    report_id: UUID
    run_id: UUID
    strategy_id: UUID
    strategy_fingerprint: str
    skill_id: str
    skill_version: str
    research_focus: str
    input_fingerprint: str
    status: ResearchStatus
    artifact_file: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_report(cls, report: SavedResearchReport) -> "SavedResearchReportSummary":
        return cls(
            report_id=report.report_id, run_id=report.run_id, strategy_id=report.strategy_id,
            strategy_fingerprint=report.strategy_fingerprint, skill_id=report.skill_id,
            skill_version=report.skill_version, research_focus=report.research_focus,
            input_fingerprint=report.input_fingerprint, status=report.status,
            artifact_file=f"{report.report_id}.json.gz", created_at=report.created_at,
            updated_at=report.updated_at,
        )
