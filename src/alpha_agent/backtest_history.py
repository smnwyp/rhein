"""File-backed, auditable backtest history for saved strategies."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

import pandas as pd
from pydantic import Field, JsonValue

from alpha_agent.domain.indicators import DSLModel
from alpha_agent.errors import AlphaAgentError
from rhein.engine.kpis import calculate_kpis


class SavedBacktestRun(DSLModel):
    """One immutable result for one saved strategy and one data group."""

    run_id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    # Kept alongside the stable UUID so a historical result remains legible
    # even when it is exported or viewed outside the strategy library.
    # The default makes previously saved JSON records backward compatible.
    strategy_name: str = Field(default="未命名策略（旧记录）", min_length=1)
    strategy_fingerprint: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    input_file_count: int = Field(ge=0)
    settings: dict[str, JsonValue]
    kpis: list[dict[str, JsonValue]]
    trades: list[dict[str, JsonValue]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SavedBacktestRunSummary(DSLModel):
    """Small, eagerly-readable descriptor for a saved result artifact."""

    run_id: UUID
    strategy_id: UUID
    strategy_name: str = Field(default="未命名策略（旧记录）", min_length=1)
    strategy_fingerprint: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    input_file_count: int = Field(ge=0)
    settings: dict[str, JsonValue]
    kpi_count: int = Field(ge=0)
    trade_count: int = Field(ge=0)
    artifact_file: str = Field(min_length=1)
    created_at: datetime

    @classmethod
    def from_run(cls, run: SavedBacktestRun) -> "SavedBacktestRunSummary":
        return cls(
            run_id=run.run_id,
            strategy_id=run.strategy_id,
            strategy_name=run.strategy_name,
            strategy_fingerprint=run.strategy_fingerprint,
            schema_version=run.schema_version,
            group_label=run.group_label,
            data_path=run.data_path,
            input_file_count=run.input_file_count,
            settings=run.settings,
            kpi_count=len(run.kpis),
            trade_count=len(run.trades),
            artifact_file=f"{run.run_id}.json.gz",
            created_at=run.created_at,
        )


class BacktestHistoryError(AlphaAgentError):
    code = "backtest_history_error"


class JsonBacktestHistory:
    """Repository-local history keyed by immutable saved-strategy ID.

    The index is deliberately tiny.  It is read on every Streamlit rerun to
    populate the historical-run selector.  Each potentially large KPI/trade
    payload lives in its own gzip-compressed artifact and is only read after a
    user selects that exact run.  ``path`` remains the legacy monolithic JSON
    location for backward-compatible imports.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._index_path = path.with_name(f"{path.stem}_index.json")
        self._artifact_directory = path.parent / "saved_backtest_artifacts"

    def list_for_strategy(self, strategy_id: UUID) -> list[SavedBacktestRun]:
        """Return fully hydrated runs; use summaries in UI selector paths."""
        if not self._index_path.exists():
            return sorted(
                (entry for entry in self._read_legacy() if entry.strategy_id == strategy_id),
                key=lambda entry: entry.created_at,
                reverse=True,
            )
        return sorted(
            (self._load_summary(entry) for entry in self.list_summaries_for_strategy(strategy_id)),
            key=lambda entry: entry.created_at,
            reverse=True,
        )

    def list_summaries_for_strategy(self, strategy_id: UUID) -> list[SavedBacktestRunSummary]:
        return sorted(
            (entry for entry in self._read_summaries() if entry.strategy_id == strategy_id),
            key=lambda entry: entry.created_at,
            reverse=True,
        )

    def load(self, run_id: UUID) -> SavedBacktestRun:
        """Hydrate precisely one previously selected historical run."""
        if not self._index_path.exists():
            legacy = next((entry for entry in self._read_legacy() if entry.run_id == run_id), None)
            if legacy is not None:
                return legacy
            raise BacktestHistoryError("saved backtest run was not found", details={"run_id": str(run_id)})
        summary = next((entry for entry in self._read_summaries() if entry.run_id == run_id), None)
        if summary is None:
            raise BacktestHistoryError("saved backtest run was not found", details={"run_id": str(run_id)})
        return self._load_summary(summary)

    def save(self, run: SavedBacktestRun) -> SavedBacktestRun:
        entries = self._ensure_index()
        summary = SavedBacktestRunSummary.from_run(run)
        self._write_artifact(run, summary)
        entries.append(summary)
        self._write_summaries(entries)
        return run

    def delete_for_strategy(self, strategy_id: UUID) -> None:
        entries = self._ensure_index()
        removed = [entry for entry in entries if entry.strategy_id == strategy_id]
        self._delete_artifacts(removed)
        self._write_summaries([entry for entry in entries if entry.strategy_id != strategy_id])

    def delete(self, run_id: UUID) -> None:
        """Remove one known run without affecting other strategy history."""
        entries = self._ensure_index()
        remaining = [entry for entry in entries if entry.run_id != run_id]
        if len(remaining) == len(entries):
            raise BacktestHistoryError("saved backtest run was not found", details={"run_id": str(run_id)})
        self._delete_artifacts([entry for entry in entries if entry.run_id == run_id])
        self._write_summaries(remaining)

    def migrate_legacy_to_artifacts(self) -> int:
        """Convert the old monolithic JSON file to lazy-loadable artifacts.

        This is intentionally explicit so a migration can be verified before
        the old file is removed from the repository.
        """
        if self._index_path.exists():
            return len(self._read_summaries())
        legacy_entries = self._read_legacy()
        summaries = [SavedBacktestRunSummary.from_run(entry) for entry in legacy_entries]
        for entry, summary in zip(legacy_entries, summaries, strict=True):
            self._write_artifact(entry, summary)
        self._write_summaries(summaries)
        return len(summaries)

    def _ensure_index(self) -> list[SavedBacktestRunSummary]:
        if self._index_path.exists():
            return self._read_summaries()
        self.migrate_legacy_to_artifacts()
        return self._read_summaries()

    def _read_summaries(self) -> list[SavedBacktestRunSummary]:
        if self._index_path.exists():
            try:
                payload = json.loads(self._index_path.read_text(encoding="utf-8"))
                return [SavedBacktestRunSummary.model_validate_json(json.dumps(item)) for item in payload]
            except (OSError, ValueError) as error:
                raise BacktestHistoryError("saved backtest history index cannot be read", details={"path": str(self._index_path)}) from error
        # Retain API compatibility for local users whose history has not yet
        # been migrated.  This fallback is deliberately not used in the
        # published repository, where the lightweight index is committed.
        return [SavedBacktestRunSummary.from_run(entry) for entry in self._read_legacy()]

    def _read_legacy(self) -> list[SavedBacktestRun]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            # The domain models are strict in Python mode. Reload through JSON
            # mode so UUID and timestamp strings are restored to their typed
            # values, exactly as the strategy library does.
            return [SavedBacktestRun.model_validate_json(json.dumps(item)) for item in payload]
        except (OSError, ValueError) as error:
            raise BacktestHistoryError("saved backtest history cannot be read", details={"path": str(self._path)}) from error

    def _load_summary(self, summary: SavedBacktestRunSummary) -> SavedBacktestRun:
        artifact_path = self._artifact_path(summary)
        try:
            with gzip.open(artifact_path, "rt", encoding="utf-8") as artifact:
                payload = json.load(artifact)
            return SavedBacktestRun.model_validate_json(json.dumps({
                **summary.model_dump(mode="json", exclude={"kpi_count", "trade_count", "artifact_file"}),
                "kpis": payload["kpis"],
                "trades": payload["trades"],
            }))
        except (OSError, ValueError, KeyError) as error:
            raise BacktestHistoryError("saved backtest artifact cannot be read", details={"run_id": str(summary.run_id), "path": str(artifact_path)}) from error

    def _write_summaries(self, entries: list[SavedBacktestRunSummary]) -> None:
        serialized = json.dumps([entry.model_dump(mode="json") for entry in entries], ensure_ascii=False, indent=2)
        self._atomic_write_text(self._index_path, serialized)

    def _write_artifact(self, run: SavedBacktestRun, summary: SavedBacktestRunSummary) -> None:
        artifact_path = self._artifact_path(summary)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"kpis": run.kpis, "trades": run.trades}, ensure_ascii=False, separators=(",", ":"))
        try:
            with NamedTemporaryFile("wb", dir=artifact_path.parent, delete=False) as temporary:
                with gzip.GzipFile(fileobj=temporary, mode="wb", compresslevel=9, mtime=0) as compressed:
                    compressed.write(payload.encode("utf-8"))
                temporary_path = Path(temporary.name)
            temporary_path.replace(artifact_path)
        except OSError as error:
            raise BacktestHistoryError("saved backtest artifact cannot be written", details={"path": str(artifact_path)}) from error

    def _delete_artifacts(self, entries: list[SavedBacktestRunSummary]) -> None:
        for entry in entries:
            try:
                self._artifact_path(entry).unlink(missing_ok=True)
            except OSError as error:
                raise BacktestHistoryError("saved backtest artifact cannot be deleted", details={"run_id": str(entry.run_id)}) from error

    def _artifact_path(self, summary: SavedBacktestRunSummary) -> Path:
        # Artifact names are generated from UUIDs; reject crafted index files
        # that try to escape the history directory.
        name = Path(summary.artifact_file).name
        if name != summary.artifact_file or not name.endswith(".json.gz"):
            raise BacktestHistoryError("saved backtest artifact path is invalid", details={"artifact_file": summary.artifact_file})
        return self._artifact_directory / name

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
                temporary.write(text)
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        except OSError as error:
            raise BacktestHistoryError("saved backtest history index cannot be written", details={"path": str(path)}) from error


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, JsonValue]]:
    """Produce finite JSON values only; pandas turns NaN/NaT into JSON null."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def recalculated_trade_level_kpis(
    stored_kpis: list[dict[str, JsonValue]],
    stored_trades: list[dict[str, JsonValue]],
    settings: dict[str, JsonValue],
) -> pd.DataFrame:
    """Rebuild trade-derived KPIs from an immutable historical trade ledger.

    Historical runs retain their original payload.  This view-level migration
    fixes calculation defects such as a missing initial-equity drawdown point
    using the authoritative, saved trades.  It intentionally leaves metrics
    requiring OHLC data (annualized return and Sharpe) untouched.
    """
    kpis = pd.DataFrame(stored_kpis)
    trades = pd.DataFrame(stored_trades)
    required_trade_columns = {"symbol", "ret_pct", "pnl_eur", "days_held"}
    if kpis.empty or "标的" not in kpis or not required_trade_columns.issubset(trades.columns):
        return kpis
    try:
        capital = float(settings.get("initial_capital", 10_000.0))
    except (TypeError, ValueError):
        capital = 10_000.0
    compound = settings.get("compound", True) is not False
    for row_index, symbol in kpis["标的"].items():
        symbol_trades = trades[trades["symbol"] == symbol].copy()
        if symbol_trades.empty:
            continue
        sort_columns = [column for column in ("exit", "entry", "signal") if column in symbol_trades]
        if sort_columns:
            symbol_trades = symbol_trades.sort_values(sort_columns, kind="stable")
        refreshed = calculate_kpis(symbol_trades, capital=capital, compound=compound)
        for key, value in refreshed.items():
            kpis.at[row_index, key] = value
        kpis.at[row_index, "cumulative_return_pct"] = (refreshed["final_equity"] / capital - 1) * 100
    return kpis


def strategy_fingerprint(strategy_json: str) -> str:
    return sha256(strategy_json.encode("utf-8")).hexdigest()
