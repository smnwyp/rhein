"""Deterministic validation and conversion for the semantic inventory phase."""
from __future__ import annotations

from alpha_agent.domain.interpretation import ClauseCoverage, ClarificationRequired, StrategyInterpretationRequest
from alpha_agent.domain.semantic_inventory import SemanticInventory
from alpha_agent.errors import ModelResponseParsingFailure, UnsupportedStrategyFeature


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
