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

    def delete(self, run_id: UUID) -> None:
        """Remove one known run without affecting other strategy history."""
        entries = self._read()
        remaining = [entry for entry in entries if entry.run_id != run_id]
        if len(remaining) == len(entries):
            raise BacktestHistoryError("saved backtest run was not found", details={"run_id": str(run_id)})
        self._write(remaining)

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
