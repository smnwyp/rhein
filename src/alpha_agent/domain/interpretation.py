"""The interpreter's typed request, policy, and discriminated results."""
from typing import Annotated, Literal
from pydantic import Field, PositiveInt
from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.strategy import StrategyDefinition
class InterpretationPolicy(DSLModel): default_rsi_window: PositiveInt | None = None
class StrategyInterpretationRequest(DSLModel):
    strategy_text: str = Field(min_length=1)
    symbol: str | None = Field(default=None, min_length=1)
    policy: InterpretationPolicy = Field(default_factory=InterpretationPolicy)
class InterpretationNote(DSLModel): code: str = Field(min_length=1); message: str = Field(min_length=1)
class ParsedStrategy(DSLModel):
    status: Literal["parsed"]; strategy: StrategyDefinition
    assumptions: list[InterpretationNote] = Field(default_factory=list)
    warnings: list[InterpretationNote] = Field(default_factory=list)
class ClarificationRequired(DSLModel):
    status: Literal["clarification_required"]
    partial_strategy: StrategyDefinition | None = None
    questions: list[ClarificationQuestion] = Field(min_length=1)
    ambiguous_terms: list[str] = Field(min_length=1)
StrategyInterpretationResult = Annotated[ParsedStrategy | ClarificationRequired, Field(discriminator="status")]


class ModelInterpretationEnvelope(DSLModel):
    """Object-root schema sent to providers that reject root discriminated unions.

    The service still parses this provider output as `StrategyInterpretationResult`.
    """

    status: Literal["parsed", "clarification_required"]
    strategy: StrategyDefinition | None = None
    assumptions: list[InterpretationNote] = Field(default_factory=list)
    warnings: list[InterpretationNote] = Field(default_factory=list)
    partial_strategy: StrategyDefinition | None = None
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    ambiguous_terms: list[str] = Field(default_factory=list)
