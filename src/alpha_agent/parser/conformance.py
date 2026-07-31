"""Source-aware checks that prevent a valid DSL from silently changing intent."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from alpha_agent.domain.interpretation import ClauseCoverage, SourceClause
from alpha_agent.errors import SemanticStrategyValidationFailure
from alpha_agent.parser.completeness import _resolve_dsl_path


def _has(node: object, key: str, value: object) -> bool:
    if isinstance(node, Mapping):
        return node.get(key) == value or any(_has(child, key, value) for child in node.values())
    if isinstance(node, list):
        return any(_has(child, key, value) for child in node)
    return False


def _mapped_subtrees(clause_id: str, coverage: Iterable[ClauseCoverage], strategy_payload: dict[str, Any]) -> list[object]:
    values: list[object] = []
    for entry in coverage:
        if entry.clause_id != clause_id or entry.disposition not in {"mapped", "assumption"}:
            continue
        for path in entry.dsl_paths:
            try:
                values.append(_resolve_dsl_path(strategy_payload, path))
            except (KeyError, IndexError, ValueError):
                # Coverage validation emits the precise missing-path failure.
                continue
    return values


def validate_source_conformance(source_clauses: Iterable[SourceClause], coverage: list[ClauseCoverage], strategy_payload: dict[str, Any]) -> None:
    """Enforce a small set of explicit source-to-DSL semantic invariants.

    These checks intentionally cover expressions whose DSL representations have
    a distinct semantic node.  They do not try to infer general natural-language
    equivalence, which remains the model's responsibility.
    """
    issues: list[dict[str, str]] = []
    for clause in source_clauses:
        mapped = _mapped_subtrees(clause.clause_id, coverage, strategy_payload)
        text = clause.text
        if re.search(r"(?:上穿|向上穿越)", text):
            if not any(_has(value, "operator", "cross_above") for value in mapped):
                issues.append({"path": f"source.{clause.clause_id}", "rule": "cross_requires_cross_node", "message": "an ‘上穿’ source clause must map to a cross_above node"})
        if re.search(r"(?:下穿|向下穿越)", text):
            if not any(_has(value, "operator", "cross_below") for value in mapped):
                issues.append({"path": f"source.{clause.clause_id}", "rule": "cross_requires_cross_node", "message": "a ‘下穿’ source clause must map to a cross_below node"})
        if re.search(r"(?:相对(?:前一|上一)(?:个)?交易日(?:收盘价)?|单日).*?(?:涨幅|收益|回报)", text):
            if not any(_has(value, "indicator", "rolling_return") and _has(value, "window", 1) for value in mapped):
                issues.append({"path": f"source.{clause.clause_id}", "rule": "daily_return_requires_rolling_return", "message": "a one-day return clause must map to rolling_return(window=1), not a price-level comparison"})
    if issues:
        raise SemanticStrategyValidationFailure(issues)
