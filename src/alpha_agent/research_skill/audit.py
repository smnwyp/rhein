"""Replay each selected/all trade using the existing deterministic v0.3 engine."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.research.v03_engine import build_trade_event_audit

from .models import SourceCoverage, StatisticalEvidence


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def attach_replay_audits(
    evidence: StatisticalEvidence, *, strategy: TimedStrategyDefinition,
    source_paths: dict[str, Path], load_ohlc: Callable[[Path], pd.DataFrame],
    market_index: pd.DataFrame | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> StatisticalEvidence:
    """Add actual entry/exit condition paths; failed source reads stay visible."""
    enriched = []
    unreadable: list[str] = []
    notes: list[str] = []
    frames_by_path: dict[Path, pd.DataFrame] = {}
    total = len(evidence.trades)
    report_every = max(1, total // 100)

    def report(completed: int, symbol: str) -> None:
        if progress is not None and (completed == 0 or completed == total or completed % report_every == 0):
            progress(completed, total, symbol)

    for index, trade in enumerate(evidence.trades):
        report(index, trade.symbol)
        path = source_paths.get(trade.symbol)
        if path is None:
            unreadable.append(trade.trade_id); notes.append(f"{trade.symbol}: no source path in saved run")
            enriched.append(trade)
        else:
            try:
                frame = frames_by_path.get(path)
                if frame is None:
                    frame = load_ohlc(path)
                    frames_by_path[path] = frame
                common = dict(strategy=strategy, signal_date=trade.signal_date, entry_date=trade.entry_date, exit_date=trade.exit_date, reason=trade.exit_reason, market_index=market_index)
                enriched.append(trade.model_copy(update={
                    "source_path": str(path),
                    "entry_audit": _json_safe(build_trade_event_audit(frame, event="entry", **common)),
                    "exit_audit": _json_safe(build_trade_event_audit(frame, event="exit", **common)),
                }))
            except Exception as error:  # Coverage, rather than invented observations, is the evidence.
                unreadable.append(trade.trade_id); notes.append(f"{trade.symbol}: {type(error).__name__}")
                enriched.append(trade.model_copy(update={"source_path": str(path)}))
        report(index + 1, trade.symbol)
    return evidence.model_copy(update={
        "trades": enriched,
        "source_coverage": SourceCoverage(total_trades=len(enriched), readable_trade_count=len(enriched) - len(unreadable), unreadable_trade_ids=unreadable, notes=notes),
    })
