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


_NUMBERED_SECTION = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")
_DEFINITION_SECTION = re.compile(r"^\s*(?:定义|术语|数据口径)\s*[:：]\s*$")


def _numbered_strategy_sections(strategy_text: str) -> list[str] | None:
    """Keep a specification's meaningful numbered blocks together.

    A formal strategy often has one formula per line. Treating every formula,
    heading, and explanatory sentence as a separate coverage item makes the
    contract noisy and can exhaust a provider's structured-output budget. A
    numbered subsection is the smallest reliable deterministic unit here.
    """
    lines = [line.strip() for line in strategy_text.splitlines() if line.strip()]
    if sum(bool(_NUMBERED_SECTION.match(line)) for line in lines) < 2:
        return None
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        if (_NUMBERED_SECTION.match(line) or _DEFINITION_SECTION.match(line)) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def segment_source_clauses(strategy_text: str) -> list[SourceClause]:
    """Create stable review IDs at a useful semantic granularity."""
    numbered_sections = _numbered_strategy_sections(strategy_text)
    if numbered_sections is not None:
        return [SourceClause(clause_id=f"C{index:02d}", text=text) for index, text in enumerate(numbered_sections, start=1)]
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
    if re.search(r"(?:t0\s*-\s*\d+[^。；;]*低点|低点[^。；;]*t0)", text, flags=re.IGNORECASE) and not _answered(request, "anchor_low_reference_field"):
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
    if re.search(r"样本结束.*未平仓", text) and not _answered(request, "sample_end_open_position"):
        questions.append(ClarificationQuestion(
            question_id="sample_end_open_position",
            question="样本结束时仍未平仓的头寸如何处理？",
            target_path="lifecycle_policy.sample_end_open_position",
            suggested_answers=["样本最后一个交易日按收盘价强制平仓", "保留未平仓头寸，并从已平仓 KPI 中排除"],
            answer_kind="choice",
        ))
        ambiguous.append("样本结束未平仓头寸的处理")
    if re.search(r"卖出后.*重新寻找.*C\s*点", text, flags=re.IGNORECASE) and not _answered(request, "allow_reentry_after_exit"):
        questions.append(ClarificationQuestion(
            question_id="allow_reentry_after_exit",
            question="一笔交易卖出后，是否允许继续寻找下一次 C 点并再次入场？",
            target_path="lifecycle_policy.allow_reentry_after_exit",
            suggested_answers=["允许继续寻找下一次 C 点", "不允许；每个标的只执行第一笔交易"],
            answer_kind="choice",
        ))
        ambiguous.append("卖出后是否重新寻找 C 点")
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
