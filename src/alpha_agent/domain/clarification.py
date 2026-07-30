"""Precise unresolved inputs returned by the interpreter agent."""
from typing import Literal
from pydantic import Field, model_validator
from alpha_agent.domain.indicators import DSLModel
class ClarificationQuestion(DSLModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target_path: str = Field(min_length=1)
    suggested_answers: list[str] = Field(default_factory=list)
    answer_kind: Literal["number", "percentage", "window", "choice", "free_text"] = "free_text"
    @model_validator(mode="after")
    def validate_choices(self) -> "ClarificationQuestion":
        if self.answer_kind == "choice" and not self.suggested_answers: raise ValueError("choice questions require suggested_answers")
        return self
