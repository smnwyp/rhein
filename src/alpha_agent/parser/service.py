"""Strategy Interpreter Agent orchestration."""

from pydantic import TypeAdapter, ValidationError

from alpha_agent.domain.results import ParsedStrategy, ParserResult
from alpha_agent.errors import ModelResponseError
from alpha_agent.model.client import ModelClient
from alpha_agent.parser.prompts import SYSTEM_PROMPT, user_prompt
from alpha_agent.parser.validation import validate_strategy

_RESULT_ADAPTER = TypeAdapter(ParserResult)


class StrategyInterpreter:
    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def interpret(self, text: str, *, asset: str | None = None) -> ParserResult:
        raw = self._client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt(text, asset),
        )
        try:
            result = _RESULT_ADAPTER.validate_python(raw)
        except ValidationError as error:
            raise ModelResponseError("model response does not match the parser contract") from error
        if isinstance(result, ParsedStrategy):
            validate_strategy(result.strategy)
        return result
