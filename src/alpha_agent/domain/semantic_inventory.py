"""A compact semantic intermediate representation for strategy interpretation.

The inventory deliberately precedes the recursive executable DSL.  It gives the
interpreter a typed place to record symbols, state, data requirements, and
unresolved intent before asking a model to emit every low-level DSL node.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.indicators import DSLModel


ClosureDimension = Literal[
    "vocabulary",
    "parameters",
    "temporal",
    "state",
    "execution",
    "data",
    "lifecycle",
    "capability",
]


class InventorySymbol(DSLModel):
    name: str = Field(min_length=1, max_length=40)
    meaning: str = Field(min_length=1, max_length=120)
    source_clause_ids: list[str] = Field(min_length=1, max_length=8)


class SemanticItem(DSLModel):
    """One source-grounded, DSL-like semantic requirement—not executable code."""

    semantic_id: str = Field(pattern=r"^I-[A-Z0-9_-]+$")
    source_clause_ids: list[str] = Field(min_length=1, max_length=12)
    category: Literal[
        "data_policy",
        "execution_policy",
        "anchor_condition",
        "anchor_constraint",
        "entry_rule",
        "exit_rule",
        "state_transition",
        "lifecycle_policy",
        "order_failure_policy",
        "definition",
    ]
    pseudo_dsl: str = Field(min_length=1, max_length=240)
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    required_capabilities: list[str] = Field(default_factory=list, max_length=12)
    disposition: Literal["resolved", "clarification_required", "unsupported"]


class ClosureFinding(DSLModel):
    dimension: ClosureDimension
    status: Literal["closed", "clarification_required", "unsupported", "contradiction"]
    explanation: str = Field(min_length=1, max_length=180)
    source_clause_ids: list[str] = Field(default_factory=list, max_length=12)
    semantic_ids: list[str] = Field(default_factory=list, max_length=16)


class InventoryUnsupportedFeature(DSLModel):
    capability_id: str = Field(min_length=1)
    source_clause_ids: list[str] = Field(min_length=1, max_length=12)
    source_concept: str = Field(min_length=1, max_length=180)
    required_extension: str = Field(min_length=1, max_length=180)


class SemanticInventory(DSLModel):
    """Model-produced, locally validated interpretation plan."""

    status: Literal["compile_eligible", "clarification_required", "unsupported"]
    symbols: list[InventorySymbol] = Field(default_factory=list, max_length=16)
    items: list[SemanticItem] = Field(min_length=1, max_length=20)
    closure: list[ClosureFinding] = Field(min_length=1, max_length=8)
    questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=8)
    ambiguous_terms: list[str] = Field(default_factory=list, max_length=12)
    unsupported_features: list[InventoryUnsupportedFeature] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def status_matches_findings(self) -> "SemanticInventory":
        if len({item.semantic_id for item in self.items}) != len(self.items):
            raise ValueError("semantic item IDs must be unique")
        blocked = {finding.status for finding in self.closure if finding.status != "closed"}
        if self.status == "compile_eligible" and (blocked or self.questions or self.unsupported_features):
            raise ValueError("compile-eligible inventory cannot contain unresolved findings")
        if self.status == "clarification_required" and not self.questions:
            raise ValueError("clarification-required inventory must contain questions")
        if self.status == "unsupported" and not self.unsupported_features:
            raise ValueError("unsupported inventory must identify unsupported features")
        return self
