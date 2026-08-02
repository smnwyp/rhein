"""The interpreter's typed request, policy, and discriminated results."""
from typing import Annotated, Literal
from pydantic import Field, PositiveInt
from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.strategy import AnyStrategyDefinition
class InterpretationPolicy(DSLModel):
    default_rsi_window: PositiveInt | None = None
    # Product-level research policy confirmed by the user: daily backtests use
    # the final daily close, even when source prose describes a pre-close order.
    execution_price_policy: Literal["daily_close"] = "daily_close"


class ClarificationAnswer(DSLModel):
    question_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SourceClause(DSLModel):
    """A deterministic, user-visible fragment of the source strategy."""

    clause_id: str = Field(pattern=r"^C[0-9]{2,}$")
    text: str = Field(min_length=1)


class ClauseCoverage(DSLModel):
    """The interpreter's audit trail for one source clause."""

    clause_id: str = Field(pattern=r"^C[0-9]{2,}$")
    disposition: Literal["mapped", "assumption", "clarification_required", "unsupported"]
    dsl_paths: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class StrategyInterpretationRequest(DSLModel):
    strategy_text: str = Field(min_length=1)
    symbol: str | None = Field(default=None, min_length=1)
    policy: InterpretationPolicy = Field(default_factory=InterpretationPolicy)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)
    source_clauses: list[SourceClause] = Field(default_factory=list)
    repair_instruction: str | None = Field(default=None, exclude=True)
class InterpretationNote(DSLModel): code: str = Field(min_length=1); message: str = Field(min_length=1)
class ParsedStrategy(DSLModel):
    status: Literal["parsed"]; strategy: AnyStrategyDefinition
    assumptions: list[InterpretationNote] = Field(default_factory=list)
    warnings: list[InterpretationNote] = Field(default_factory=list)
    source_clauses: list[SourceClause] = Field(default_factory=list)
    coverage: list[ClauseCoverage] = Field(default_factory=list)
class ClarificationRequired(DSLModel):
    status: Literal["clarification_required"]
    partial_strategy: AnyStrategyDefinition | None = None
    questions: list[ClarificationQuestion] = Field(min_length=1)
    ambiguous_terms: list[str] = Field(min_length=1)
    source_clauses: list[SourceClause] = Field(default_factory=list)
    coverage: list[ClauseCoverage] = Field(default_factory=list)
StrategyInterpretationResult = Annotated[ParsedStrategy | ClarificationRequired, Field(discriminator="status")]


class ModelInterpretationEnvelope(DSLModel):
    """Object-root schema sent to providers that reject root discriminated unions.

    The service still parses this provider output as `StrategyInterpretationResult`.
    """

    status: Literal["parsed", "clarification_required"]
    strategy: AnyStrategyDefinition | None = None
    assumptions: list[InterpretationNote] = Field(default_factory=list)
    warnings: list[InterpretationNote] = Field(default_factory=list)
    partial_strategy: AnyStrategyDefinition | None = None
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    ambiguous_terms: list[str] = Field(default_factory=list)
    coverage: list[ClauseCoverage] = Field(default_factory=list)
