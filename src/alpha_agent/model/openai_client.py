"""Optional OpenAI Responses API adapter; the domain remains provider-independent."""
import json
from collections.abc import Mapping
from typing import Any
from alpha_agent.domain.interpretation import ModelInterpretationEnvelope, StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.parser.prompts import SYSTEM_PROMPT, user_prompt

class OpenAIStrategyModelClient:
    def __init__(self, *, api_key: str | None = None, model: str = "gpt-5.6-sol", client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ModelClientFailure("OpenAI client is not installed; install project dependencies") from error
            client = OpenAI(api_key=api_key)
        self._client, self._model = client, model

    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        # The Responses API requires an object root and rejects root oneOf/anyOf.
        # The envelope is provider-facing only; service.py still validates the result
        # against the narrower discriminated union used by the domain.
        schema = ModelInterpretationEnvelope.model_json_schema()
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt(request),
                text={"format": {"type": "json_schema", "name": "strategy_interpretation", "schema": schema, "strict": False}},
                store=False,
            )
            return json.loads(response.output_text)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelClientFailure("OpenAI returned invalid structured output", details={"exception_type": type(error).__name__}) from error
        except Exception as error:
            raise ModelClientFailure(
                "OpenAI strategy interpretation request failed",
                details={
                    "exception_type": type(error).__name__,
                    "http_status": getattr(error, "status_code", None),
                    "provider_message": str(error),
                    "model": self._model,
                },
            ) from error
