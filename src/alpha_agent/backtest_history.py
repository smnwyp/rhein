"""File-backed, auditable backtest history for saved strategies."""
from __future__ import annotations

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


class SavedBacktestRun(DSLModel):
    """One immutable result for one saved strategy and one data group."""

    run_id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    strategy_fingerprint: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    input_file_count: int = Field(ge=0)
    settings: dict[str, JsonValue]
    kpis: list[dict[str, JsonValue]]
    trades: list[dict[str, JsonValue]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestHistoryError(AlphaAgentError):
    code = "backtest_history_error"


class JsonBacktestHistory:
    """Repository-local history keyed by immutable saved-strategy ID."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def list_for_strategy(self, strategy_id: UUID) -> list[SavedBacktestRun]:
        return sorted(
            (entry for entry in self._read() if entry.strategy_id == strategy_id),
            key=lambda entry: entry.created_at,
            reverse=True,
        )

    def save(self, run: SavedBacktestRun) -> SavedBacktestRun:
        entries = self._read()
        entries.append(run)
        self._write(entries)
        return run

    def delete_for_strategy(self, strategy_id: UUID) -> None:
        entries = self._read()
        self._write([entry for entry in entries if entry.strategy_id != strategy_id])

    def _read(self) -> list[SavedBacktestRun]:
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

    def _write(self, entries: list[SavedBacktestRun]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps([entry.model_dump(mode="json") for entry in entries], ensure_ascii=False, indent=2)
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self._path.parent, delete=False) as temporary:
                temporary.write(serialized)
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._path)
        except OSError as error:
            raise BacktestHistoryError("saved backtest history cannot be written", details={"path": str(self._path)}) from error


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, JsonValue]]:
    """Produce finite JSON values only; pandas turns NaN/NaT into JSON null."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def strategy_fingerprint(strategy_json: str) -> str:
    return sha256(strategy_json.encode("utf-8")).hexdigest()
