"""Structured clarification requests."""

from typing import Literal

from pydantic import Field, model_validator

from alpha_agent.domain.indicators import DSLModel


class ClarificationQuestion(DSLModel):
    id: str = Field(min_length=1)
    ambiguous_term: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer_type: Literal["number", "percentage", "window", "choice", "free_text"]
    choices: list[str] | None = None

    @model_validator(mode="after")
    def choices_match_type(self) -> "ClarificationQuestion":
        if self.expected_answer_type == "choice" and not self.choices:
            raise ValueError("choice questions require choices")
        if self.expected_answer_type != "choice" and self.choices is not None:
            raise ValueError("choices are only valid for choice questions")
        return self
