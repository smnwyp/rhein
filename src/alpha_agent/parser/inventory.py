"""Deterministic validation and conversion for the semantic inventory phase."""
from __future__ import annotations

import re

from alpha_agent.domain.interpretation import ClauseCoverage, ClarificationRequired, StrategyInterpretationRequest
from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.semantic_inventory import SemanticInventory
from alpha_agent.errors import ModelResponseParsingFailure, UnsupportedStrategyFeature


_THIRD_DAY_OPEN_SEQUENCE = re.compile(
    r"第二天.*?收盘.*?(?:符合|满足).*?(?:所有)?(?:买入)?条件.*?第三天.*?开盘.*?(?:买入|入场)",
    flags=re.DOTALL,
)
_OR_WAIT_CONDITION = re.compile(
    r"[^。；;]*(?:或[^。；;]*){2}则[^。；;]*观望",
    flags=re.DOTALL,
)
_BOUNDED_DD_DECLINE_EXIT = re.compile(
    r"自买入点(?:的)?\s*\d+\s*个交易日内.*?DD.*?连续(?:二|两)天(?:下)?降.*?(?:抛出|卖出)",
    flags=re.DOTALL | re.IGNORECASE,
)
_EXPLICIT_INSTRUMENT = re.compile(
    r"(?:股票|标的|证券|代码|ticker)\s*[:：]?\s*(?:[A-Za-z][A-Za-z0-9.^_-]{0,11}|(?:SH|SZ)?\d{6})",
    flags=re.IGNORECASE,
)
_REENTRY_WORDING = re.compile(r"重新买入|再(?:次)?入场|再(?:次)?交易|重新寻找", flags=re.IGNORECASE)


def source_resolved_inventory_questions(
    questions: list[ClarificationQuestion], request: StrategyInterpretationRequest,
) -> tuple[list[ClarificationQuestion], list[str]]:
    """Remove only questions directly contradicted by explicit source wording.

    The semantic inventory is a planning aid, not an authority to turn a stated
    time, window, or logical connector into a user choice.  This narrow guard
    leaves every genuinely absent parameter untouched and merely prevents a
    model's conservative uncertainty from blocking the DSL compiler.
    """
    text = request.strategy_text
    has_third_day_open = bool(_THIRD_DAY_OPEN_SEQUENCE.search(text))
    has_or_wait = bool(_OR_WAIT_CONDITION.search(text))
    has_bounded_dd_exit = bool(_BOUNDED_DD_DECLINE_EXIT.search(text))
    uses_current_input_group = request.symbol is None and not _EXPLICIT_INSTRUMENT.search(text)
    remaining: list[ClarificationQuestion] = []
    suppressed_ids: list[str] = []
    for question in questions:
        wording = f"{question.question} {' '.join(question.suggested_answers)}"
        target = question.target_path.lower()
        is_third_day_price_question = (
            has_third_day_open
            and target.startswith("entry")
            and bool(re.search(r"第三天|开盘|收盘|入场价格|next[_ ]?open|daily[_ ]?close", wording, flags=re.IGNORECASE))
        )
        is_or_wait_question = (
            has_or_wait
            and "观望" in wording
            and bool(re.search(r"\bOR\b|\bAND\b|任意|同时|三(?:个|项)", wording, flags=re.IGNORECASE))
        )
        is_dd_window_question = (
            has_bounded_dd_exit
            and (target.startswith("exit") or "DD" in wording.upper())
            and bool(re.search(r"DD|连续|相邻|窗口|买入后|\d+日", wording, flags=re.IGNORECASE))
        )
        is_group_symbol_question = (
            uses_current_input_group
            and (target.startswith("symbol") or target.startswith("universe"))
            and bool(re.search(r"股票代码|标的代码|ticker|证券代码|AAPL|TSLA|适用.*?(?:股票|标的)", wording, flags=re.IGNORECASE))
        )
        is_default_reentry_question = (
            request.policy.allow_reentry_after_exit
            and target == "lifecycle_policy.allow_reentry_after_exit"
            and bool(_REENTRY_WORDING.search(wording))
        )
        if is_third_day_price_question or is_or_wait_question or is_dd_window_question or is_group_symbol_question or is_default_reentry_question:
            suppressed_ids.append(question.question_id)
        else:
            remaining.append(question)
    return remaining, suppressed_ids


def validate_inventory(inventory: SemanticInventory, request: StrategyInterpretationRequest) -> None:
    """Ensure the model's semantic plan remains rooted in every source clause."""
    known = {clause.clause_id for clause in request.source_clauses}
    # An inventory has several semantic surfaces.  A definition can legitimately
    # be represented by the symbol table, and a policy/capability statement by a
    # closure or unsupported finding, without duplicating it as an executable
    # item.  Requiring every clause to appear only in ``items`` falsely rejects
    # well-grounded structured specifications.
    referenced = {
        clause_id
        for clause_ids in (
            *(item.source_clause_ids for item in inventory.items),
            *(symbol.source_clause_ids for symbol in inventory.symbols),
            *(finding.source_clause_ids for finding in inventory.closure),
            *(feature.source_clause_ids for feature in inventory.unsupported_features),
        )
        for clause_id in clause_ids
    }
    unknown = sorted(referenced - known)
    missing = sorted(known - referenced)
    if unknown or missing:
        raise ModelResponseParsingFailure(
            "semantic inventory does not cover the source clauses",
            details={
                "validation_errors": [
                    *[{"path": "items", "rule": "known_source_clause", "message": f"unknown source clause: {clause_id}"} for clause_id in unknown],
                    *[{"path": "items", "rule": "complete_source_coverage", "message": f"source clause has no semantic item: {clause_id}"} for clause_id in missing],
                ]
            },
        )


def inventory_clarification(inventory: SemanticInventory, request: StrategyInterpretationRequest) -> ClarificationRequired:
    coverage = [
        ClauseCoverage(
            clause_id=clause.clause_id,
            disposition="clarification_required",
            dsl_paths=[],
            explanation="语义清单显示该策略尚未闭合；等待必要澄清。",
        )
        for clause in request.source_clauses
    ]
    return ClarificationRequired(
        status="clarification_required",
        partial_strategy=None,
        questions=inventory.questions,
        ambiguous_terms=inventory.ambiguous_terms or ["策略语义尚未闭合"],
        source_clauses=request.source_clauses,
        coverage=coverage,
    )


def raise_inventory_unsupported(inventory: SemanticInventory) -> None:
    raise UnsupportedStrategyFeature(
        "strategy requires capabilities that the current DSL or deterministic engine cannot express faithfully",
        details={
            "unsupported_features": [feature.model_dump(mode="json") for feature in inventory.unsupported_features],
            "closure": [finding.model_dump(mode="json") for finding in inventory.closure],
            "semantic_items": [item.model_dump(mode="json") for item in inventory.items],
        },
    )
