"""Durable, file-backed requests for out-of-process backtests.

The Streamlit process only creates and reads jobs.  A worker process owns the
CPU-intensive execution and writes one result artifact per source file, so a
browser rerun never owns the lifetime of an all-universe backtest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

from pydantic import Field, JsonValue

from alpha_agent.domain.indicators import DSLModel
from alpha_agent.errors import AlphaAgentError


class BacktestJobError(AlphaAgentError):
    code = "backtest_job_error"


class BacktestJob(DSLModel):
    job_id: UUID = Field(default_factory=uuid4)
    status: str = Field(pattern="^(queued|running|completed|failed)$")
    strategy_id: UUID
    strategy_name: str = Field(min_length=1)
    strategy_fingerprint: str = Field(min_length=1)
    strategy_payload: dict[str, JsonValue]
    group_label: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    settings: dict[str, JsonValue]
    total_sources: int = Field(ge=0)
    completed_sources: int = Field(default=0, ge=0)
    failed_sources: int = Field(default=0, ge=0)
    current_symbol: str | None = None
    failure_reason: str | None = None
    run_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def progress_fraction(self) -> float:
        return self.completed_sources / self.total_sources if self.total_sources else 0.0


class JsonBacktestJobStore:
    """Atomically persists job state and sharded per-symbol artifacts."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def create(self, job: BacktestJob) -> BacktestJob:
        if self.job_path(job.job_id).exists():
            raise BacktestJobError("backtest job already exists", details={"job_id": str(job.job_id)})
        self._write_json(self.job_path(job.job_id), job.model_dump(mode="json"))
        return job

    def load(self, job_id: UUID) -> BacktestJob:
        try:
            return BacktestJob.model_validate_json(self.job_path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BacktestJobError("backtest job cannot be read", details={"job_id": str(job_id)}) from error

    def save(self, job: BacktestJob) -> BacktestJob:
        updated = job.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self._write_json(self.job_path(updated.job_id), updated.model_dump(mode="json"))
        return updated

    def find_resumable(self, *, strategy_id: UUID, strategy_fingerprint: str, data_path: str) -> BacktestJob | None:
        candidates: list[BacktestJob] = []
        if not self.directory.exists():
            return None
        for path in self.directory.glob("*.json"):
            try:
                candidate = BacktestJob.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                candidate.strategy_id == strategy_id
                and candidate.strategy_fingerprint == strategy_fingerprint
                and candidate.data_path == data_path
                and candidate.status in {"queued", "running", "failed"}
            ):
                candidates.append(candidate)
        return max(candidates, key=lambda item: item.updated_at) if candidates else None

    def result_path(self, job_id: UUID, source_path: str) -> Path:
        # The hash gives each source a stable, path-safe artifact name.  The
        # original source string remains inside the JSON for auditability.
        return self.directory / str(job_id) / "results" / f"{sha256(source_path.encode()).hexdigest()}.json"

    def result_exists(self, job_id: UUID, source_path: str) -> bool:
        path = self.result_path(job_id, source_path)
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("source_path") == source_path
        except (OSError, ValueError):
            return False

    def write_result(self, job_id: UUID, source_path: str, *, kpi: dict[str, JsonValue] | None, trades: list[dict[str, JsonValue]], error: str | None) -> None:
        self._write_json(self.result_path(job_id, source_path), {
            "source_path": source_path, "kpi": kpi, "trades": trades, "error": error,
        })

    def read_results(self, job_id: UUID) -> list[dict[str, object]]:
        directory = self.directory / str(job_id) / "results"
        results: list[dict[str, object]] = []
        for path in sorted(directory.glob("*.json")) if directory.exists() else []:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise BacktestJobError("backtest job result cannot be read", details={"path": str(path)}) from error
            if not isinstance(value, dict) or not isinstance(value.get("source_path"), str):
                raise BacktestJobError("backtest job result is invalid", details={"path": str(path)})
            results.append(value)
        return results

    def job_path(self, job_id: UUID) -> Path:
        return self.directory / f"{job_id}.json"

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
                json.dump(value, temporary, ensure_ascii=False, separators=(",", ":"))
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        except OSError as error:
            raise BacktestJobError("backtest job cannot be written", details={"path": str(path)}) from error
