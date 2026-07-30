"""Discriminated interpretation outcomes."""

from typing import Annotated, Literal

from pydantic import Field

from alpha_agent.domain.clarification import ClarificationQuestion
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.strategy import StrategyDefinition


class ParserNote(DSLModel):
    code: str
    message: str


class ParsedStrategy(DSLModel):
    status: Literal["parsed"]
    strategy: StrategyDefinition
    assumptions: list[ParserNote] = Field(default_factory=list)
    warnings: list[ParserNote] = Field(default_factory=list)


class ClarificationRequired(DSLModel):
    status: Literal["clarification_required"]
    partially_parsed_strategy: StrategyDefinition | None = None
    questions: list[ClarificationQuestion] = Field(min_length=1)
    ambiguous_terms: list[str] = Field(min_length=1)


ParserResult = Annotated[ParsedStrategy | ClarificationRequired, Field(discriminator="status")]
