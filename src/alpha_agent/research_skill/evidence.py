"""Deterministic evidence and sampling for the fixed v1 research protocol."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from statistics import median
from typing import Iterable

import numpy as np

from .models import (
    CohortSummary, DistributionCell, SourceCoverage, StatisticalEvidence,
    TradeEvidence, VisualSample,
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _integer(value: object) -> int:
    return max(0, int(_number(value)))


def _trade_id(row: dict[str, object], ordinal: int) -> str:
    identity = "|".join(str(row.get(key, "")) for key in ("symbol", "signal", "entry", "exit", "reason"))
    return sha256(f"{ordinal}|{identity}".encode("utf-8")).hexdigest()[:20]


def _periods(rows: list[dict[str, object]]) -> list[str]:
    """Assign three continuous chronological partitions without future data."""
    ordered = sorted(range(len(rows)), key=lambda index: (str(rows[index].get("entry", "")), index))
    result = ["period_1"] * len(rows)
    for position, index in enumerate(ordered):
        result[index] = ("period_1", "period_2", "period_3")[min(2, position * 3 // max(1, len(ordered)))]
    return result


def _summary(trades: list[TradeEvidence]) -> CohortSummary:
    values = [trade.return_pct for trade in trades]
    return CohortSummary(
        trade_count=len(trades), trade_ids=[trade.trade_id for trade in trades],
        winning_trade_count=sum(trade.return_pct > 0 for trade in trades),
        mean_return_pct=float(np.mean(values)) if values else None,
        median_return_pct=float(median(values)) if values else None,
        mean_days_held=float(np.mean([trade.days_held for trade in trades])) if trades else None,
    )


def _distribution(dimension: str, values: Iterable[tuple[str, TradeEvidence]]) -> list[DistributionCell]:
    grouped: dict[str, list[TradeEvidence]] = defaultdict(list)
    for key, trade in values:
        grouped[key].append(trade)
    return [
        DistributionCell(
            dimension=dimension, key=key, trade_count=len(rows),
            winning_trade_count=sum(row.return_pct > 0 for row in rows),
            mean_return_pct=float(np.mean([row.return_pct for row in rows])),
            median_return_pct=float(median([row.return_pct for row in rows])),
            mean_days_held=float(np.mean([row.days_held for row in rows])),
            evidence_sufficient=len(rows) >= 5,
        )
        for key, rows in sorted(grouped.items())
    ]


def build_statistical_evidence(trades: list[dict[str, object]]) -> StatisticalEvidence:
    """Calculate every v1 KPI/cohort in Python from the immutable ledger."""
    periods = _periods(trades)
    normalized = [
        TradeEvidence(
            trade_id=_trade_id(row, ordinal), ledger_index=ordinal, symbol=str(row.get("symbol", "未知标的")),
            signal_date=str(row.get("signal", "")), entry_date=str(row.get("entry", "")),
            exit_date=str(row.get("exit", "")), return_pct=_number(row.get("ret_pct")),
            pnl_eur=_number(row.get("pnl_eur")) if row.get("pnl_eur") is not None else None,
            days_held=_integer(row.get("days_held")), exit_reason=str(row.get("reason", "未知原因")),
            period=periods[ordinal], source_path=str(row["source_path"]) if row.get("source_path") else None,
        )
        for ordinal, row in enumerate(trades)
    ]
    positive = [trade.return_pct for trade in normalized if trade.return_pct > 0]
    q75 = float(np.quantile(positive, 0.75)) if positive else None
    early_loss = [trade for trade in normalized if trade.days_held <= 2 and trade.return_pct <= 0]
    trend_winner = [
        trade for trade in normalized
        if q75 is not None and trade.return_pct >= q75 and trade.days_held >= 5
    ]
    distributions = (
        _distribution("exit_reason", ((trade.exit_reason, trade) for trade in normalized))
        + _distribution("symbol", ((trade.symbol, trade) for trade in normalized))
        + _distribution("period", ((trade.period, trade) for trade in normalized))
    )
    return StatisticalEvidence(
        trade_count=len(normalized), positive_return_q75=q75,
        cohorts={"early_loss": _summary(early_loss), "trend_winner": _summary(trend_winner)},
        distributions=distributions, trades=normalized,
        source_coverage=SourceCoverage(total_trades=len(normalized), readable_trade_count=0),
    )


def select_visual_samples(evidence: StatisticalEvidence, *, target_count: int = 60) -> list[VisualSample]:
    """Apply the fixed v1 15/15/15/15 sampling rule with stable tie breaks."""
    rows = evidence.trades
    target = min(target_count, len(rows))
    selected: list[tuple[str, TradeEvidence]] = []
    seen: set[str] = set()

    def take(bucket: str, candidates: list[TradeEvidence], limit: int) -> None:
        for trade in candidates:
            if len(selected) >= target or len([item for item in selected if item[0] == bucket]) >= limit:
                return
            if trade.trade_id not in seen:
                selected.append((bucket, trade)); seen.add(trade.trade_id)

    take("highest_return", sorted(rows, key=lambda row: (-row.return_pct, row.trade_id)), 15)
    take("lowest_return", sorted(rows, key=lambda row: (row.return_pct, row.trade_id)), 15)
    take("fast_negative_exit", sorted((row for row in rows if row.return_pct < 0), key=lambda row: (row.days_held, row.return_pct, row.trade_id)), 15)

    remaining = [row for row in rows if row.trade_id not in seen]
    strata: dict[tuple[str, str], list[TradeEvidence]] = defaultdict(list)
    for row in sorted(remaining, key=lambda item: (item.exit_reason, item.period, item.entry_date, item.trade_id)):
        strata[(row.exit_reason, row.period)].append(row)
    while strata and len(selected) < target and len([item for item in selected if item[0] == "stratified"]) < 15:
        for key in sorted(list(strata)):
            options = strata[key]
            if options:
                trade = options.pop(0)
                selected.append(("stratified", trade)); seen.add(trade.trade_id)
            if not options:
                del strata[key]
            if len(selected) >= target or len([item for item in selected if item[0] == "stratified"]) >= 15:
                break
    for trade in sorted((row for row in rows if row.trade_id not in seen), key=lambda row: (row.entry_date, row.symbol, row.trade_id)):
        if len(selected) >= target:
            break
        selected.append(("supplement", trade)); seen.add(trade.trade_id)
    return [VisualSample(trade_id=trade.trade_id, bucket=bucket, order=index) for index, (bucket, trade) in enumerate(selected)]
