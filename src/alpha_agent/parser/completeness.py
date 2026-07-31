"""Deterministic source-coverage checks for interpreter output."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.interpretation import (
    ClauseCoverage,
    ClarificationRequired,
    SourceClause,
    StrategyInterpretationRequest,
)
from alpha_agent.errors import ModelResponseParsingFailure


def segment_source_clauses(strategy_text: str) -> list[SourceClause]:
    """Create stable review IDs; quoted Chinese condition lists split by item."""
    texts: list[str] = []
    for sentence in (part.strip() for part in re.split(r"[。；;\n]+", strategy_text)):
        if not sentence:
            continue
        quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", sentence)
        if not quoted:
            texts.append(sentence)
            continue
        prefix = re.split(r"[“\"]", sentence, maxsplit=1)[0].strip(" ：:")
        for quote in quoted:
            for item in re.split(r"[、，,]", quote):
                if item.strip():
                    texts.append(f"{prefix}：{item.strip()}" if prefix else item.strip())
        tail = re.split(r"[”\"]", sentence)[-1].strip(" 。；;")
        if tail:
            texts.append(tail)
    return [SourceClause(clause_id=f"C{index:02d}", text=text) for index, text in enumerate(texts, start=1)]


def enrich_request(request: StrategyInterpretationRequest) -> StrategyInterpretationRequest:
    if request.source_clauses:
        return request
    return request.model_copy(update={"source_clauses": segment_source_clauses(request.strategy_text)})


def _answered(request: StrategyInterpretationRequest, question_id: str) -> bool:
    return any(answer.question_id == question_id and answer.answer.strip() for answer in request.clarification_answers)


def preflight_clarifications(request: StrategyInterpretationRequest) -> ClarificationRequired | None:
    """Block three unsafe market-semantic defaults before calling a provider."""
    text = request.strategy_text
    questions: list[ClarificationQuestion] = []
    ambiguous: list[str] = []
    if "低点" in text and not _answered(request, "anchor_low_reference_field"):
        questions.append(ClarificationQuestion(
            question_id="anchor_low_reference_field",
            question="“t0-15 至 t0 的低点”应按哪一个字段定义？",
            target_path="anchor.constraints",
            suggested_answers=["区间最低价（Low）", "区间最低收盘价（Close）"],
            answer_kind="choice",
        ))
        ambiguous.append("低点")
    if re.search(r"(?:t\d+)?收盘不低于\s*t0(?![的日]?收盘)", text, flags=re.IGNORECASE) and not _answered(request, "anchor_price_reference"):
        questions.append(ClarificationQuestion(
            question_id="anchor_price_reference",
            question="“不低于 t0”中的 t0 指哪一个参考价格？",
            target_path="entry.condition",
            suggested_answers=["t0 收盘价", "t0 开盘价", "t0 最高价", "t0 最低价"],
            answer_kind="choice",
        ))
        ambiguous.append("不低于 t0")
    if re.search(r"(?:MA|均线)\s*20\s*\(?(?:t0)?\)?\s*较\s*t0\s*-?\s*2", text, flags=re.IGNORECASE) and not _answered(request, "anchor_indicator_change_basis"):
        questions.append(ClarificationQuestion(
            question_id="anchor_indicator_change_basis",
            question="“MA20(t0) 较 t0-2 至少高 0.01%”中的 0.01% 是相对变化率还是绝对点数差？",
            target_path="anchor.constraints",
            suggested_answers=["相对变化率：MA20(t0) / MA20(t0-2) - 1", "绝对点数差：MA20(t0) - MA20(t0-2)"],
            answer_kind="choice",
        ))
        ambiguous.append("较 t0-2 至少高 0.01%")
    if not questions:
        return None
    coverage = [
        ClauseCoverage(clause_id=clause.clause_id, disposition="clarification_required", dsl_paths=[], explanation="等待逐条解释或澄清。")
        for clause in request.source_clauses
    ]
    return ClarificationRequired(status="clarification_required", partial_strategy=None, questions=questions, ambiguous_terms=ambiguous, source_clauses=request.source_clauses, coverage=coverage)


def _resolve_dsl_path(payload: Any, path: str) -> Any:
    """Resolve a conservative ``foo.bar[0]`` path, never by evaluating code."""
    current = payload
    segments = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", path)
    if not segments:
        raise KeyError(path)
    for attribute, index in segments:
        if not isinstance(current, dict) or attribute not in current:
            raise KeyError(path)
        current = current[attribute]
        if index:
            if not isinstance(current, list):
                raise KeyError(path)
            current = current[int(index)]
    return current


def validate_coverage(source_clauses: Iterable[SourceClause], coverage: list[ClauseCoverage], *, strategy_payload: dict[str, Any] | None, parsed: bool) -> None:
    """Reject incomplete, unresolved parsed outputs, and dangling DSL pointers."""
    expected = {clause.clause_id for clause in source_clauses}
    supplied = [entry.clause_id for entry in coverage]
    issues: list[dict[str, str]] = []
    if set(supplied) != expected or len(supplied) != len(set(supplied)):
        issues.append({"path": "coverage", "rule": "complete_source_coverage", "message": "coverage must contain every source clause exactly once"})
    for index, entry in enumerate(coverage):
        path = f"coverage[{index}]"
        if parsed and entry.disposition not in {"mapped", "assumption"}:
            issues.append({"path": path, "rule": "parsed_requires_resolved_clauses", "message": "a parsed result cannot retain unresolved or unsupported source clauses"})
        if entry.disposition in {"mapped", "assumption"}:
            if not entry.dsl_paths:
                issues.append({"path": path, "rule": "mapped_clause_has_dsl_path", "message": "mapped or assumed clauses must identify at least one DSL path"})
            elif strategy_payload is not None:
                for dsl_path in entry.dsl_paths:
                    try:
                        _resolve_dsl_path(strategy_payload, dsl_path)
                    except (KeyError, IndexError, ValueError):
                        issues.append({"path": path, "rule": "existing_dsl_path", "message": f"coverage points to a missing DSL path: {dsl_path}"})
    if issues:
        raise ModelResponseParsingFailure("model response does not satisfy the source-coverage contract", details={"validation_errors": issues})
