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


def _macd_operand(operand: object, *, component: str, lagged: bool) -> bool:
    if not isinstance(operand, Mapping):
        return False
    if lagged:
        if operand.get("kind") != "lagged_indicator" or operand.get("offset_days") != -1:
            return False
    elif operand.get("kind") != "indicator":
        return False
    indicator = operand.get("indicator")
    return isinstance(indicator, Mapping) and indicator.get("indicator") == component


def _comparison_nodes(node: object) -> list[Mapping[str, object]]:
    if isinstance(node, Mapping):
        current = [node] if node.get("node_type") == "comparison" else []
        return current + [item for value in node.values() for item in _comparison_nodes(value)]
    if isinstance(node, list):
        return [item for value in node for item in _comparison_nodes(value)]
    return []


def _has_explicit_inclusive_macd_pair(node: object, *, direction: str) -> bool:
    """Recognise the source's equality-inclusive DIF/DEA two-day definition.

    A generic ``cross_*`` node is intentionally strict on the current bar.
    When the source instead explicitly defines a touch-inclusive crossover
    (prior DIF > DEA and current DIF <= DEA, or vice versa), a pair of typed
    comparisons preserves the requested semantics more faithfully.
    """
    comparisons = _comparison_nodes(node)
    if direction == "below":
        current_operator, prior_operator = "less_than_or_equal", "greater_than"
    else:
        current_operator, prior_operator = "greater_than_or_equal", "less_than"
    current_found = any(
        item.get("operator") == current_operator
        and _macd_operand(item.get("left"), component="macd_line", lagged=False)
        and _macd_operand(item.get("right"), component="macd_signal", lagged=False)
        for item in comparisons
    )
    prior_found = any(
        item.get("operator") == prior_operator
        and _macd_operand(item.get("left"), component="macd_line", lagged=True)
        and _macd_operand(item.get("right"), component="macd_signal", lagged=True)
        for item in comparisons
    )
    return current_found and prior_found


def _has_macd_zero_axis_cross_pair(node: object) -> bool:
    """Recognise DIF crossing zero, which cannot be a CrossCondition.

    Cross nodes require two series operands. Zero is a scalar, so the faithful
    form is the explicit two-day pair: prior DIF <= 0 and current DIF > 0.
    """
    comparisons = _comparison_nodes(node)
    current_found = any(
        item.get("operator") == "greater_than"
        and _macd_operand(item.get("left"), component="macd_line", lagged=False)
        and isinstance(item.get("right"), Mapping)
        and item["right"].get("kind") == "scalar"
        and item["right"].get("value") == 0
        for item in comparisons
    )
    prior_found = any(
        item.get("operator") == "less_than_or_equal"
        and _macd_operand(item.get("left"), component="macd_line", lagged=True)
        and isinstance(item.get("right"), Mapping)
        and item["right"].get("kind") == "scalar"
        and item["right"].get("value") == 0
        for item in comparisons
    )
    return current_found and prior_found


def _has_reciprocal_price_ma_cross(node: object, *, direction: str) -> bool:
    """Recognise a price/MA cross written with the MA as the left operand.

    Formula languages often write ``CROSS(MA20, Close)`` for a price moving
    below MA20.  Its typed cross direction is necessarily ``cross_above``
    because the *left* series (MA20) moves above the right series (Close).
    This is semantically equivalent to ``CROSS(Close, MA20)`` with
    ``cross_below`` and must not be rejected solely because the operands are
    written in the reciprocal order.
    """
    if not isinstance(node, Mapping):
        if isinstance(node, list):
            return any(_has_reciprocal_price_ma_cross(item, direction=direction) for item in node)
        return False
    expected_operator = "cross_above" if direction == "below" else "cross_below"
    if (
        node.get("node_type") == "cross"
        and node.get("operator") == expected_operator
        and isinstance(node.get("left"), Mapping)
        and node["left"].get("kind") == "indicator"
        and isinstance(node.get("right"), Mapping)
        and node["right"].get("kind") == "market_field"
        and node["right"].get("field") == "close"
    ):
        return True
    return any(_has_reciprocal_price_ma_cross(value, direction=direction) for value in node.values())


def _source_defines_inclusive_macd_pair(text: str, *, direction: str) -> bool:
    if "DIF" not in text or "DEA" not in text or "前一日" not in text or "当日" not in text:
        return False
    return ("≤" in text or "<=" in text) if direction == "below" else ("≥" in text or ">=" in text)


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
            has_strict_cross = any(_has(value, "operator", "cross_above") for value in mapped)
            has_inclusive_pair = _source_defines_inclusive_macd_pair(text, direction="above") and any(
                _has_explicit_inclusive_macd_pair(value, direction="above") for value in mapped
            )
            has_zero_axis_pair = "零轴" in text and any(_has_macd_zero_axis_cross_pair(value) for value in mapped)
            if not has_strict_cross and not has_inclusive_pair and not has_zero_axis_pair:
                issues.append({"path": f"source.{clause.clause_id}", "rule": "cross_requires_cross_node", "message": "an ‘上穿’ source clause must map to a cross_above node"})
        if re.search(r"(?:下穿|向下穿越)", text):
            has_strict_cross = any(_has(value, "operator", "cross_below") for value in mapped)
            has_reciprocal_price_ma_cross = any(_has_reciprocal_price_ma_cross(value, direction="below") for value in mapped)
            has_inclusive_pair = _source_defines_inclusive_macd_pair(text, direction="below") and any(
                _has_explicit_inclusive_macd_pair(value, direction="below") for value in mapped
            )
            if not has_strict_cross and not has_reciprocal_price_ma_cross and not has_inclusive_pair:
                issues.append({"path": f"source.{clause.clause_id}", "rule": "cross_requires_cross_node", "message": "a ‘下穿’ source clause must map to a cross_below node"})
        if re.search(r"(?:相对(?:前一|上一)(?:个)?交易日(?:收盘价)?|单日).*?(?:涨幅|收益|回报)", text):
            if not any(_has(value, "indicator", "rolling_return") and _has(value, "window", 1) for value in mapped):
                issues.append({"path": f"source.{clause.clause_id}", "rule": "daily_return_requires_rolling_return", "message": "a one-day return clause must map to rolling_return(window=1), not a price-level comparison"})
    if issues:
        raise SemanticStrategyValidationFailure(issues)
