"""Amazon Bedrock Converse adapter for the provider-independent interpreter client."""
import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import requests

from alpha_agent.domain.interpretation import ModelInterpretationEnvelope, StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.parser.prompts import SYSTEM_PROMPT, user_prompt


class BedrockStrategyModelClient:
    def __init__(self, *, model: str = "us.anthropic.claude-sonnet-4-6", region: str = "us-east-1", api_key: str | None = None, session: requests.Session | None = None) -> None:
        self._model, self._region = model, region
        self._api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self._session = session or requests.Session()

    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        if not self._api_key:
            raise ModelClientFailure("Bedrock API key is missing", details={"required_env": "AWS_BEARER_TOKEN_BEDROCK"})
        url = f"https://bedrock-runtime.{self._region}.amazonaws.com/model/{quote(self._model, safe='.:_-')}/converse"
        payload = {
            "system": [{"text": self._system_instruction()}],
            "messages": [{"role": "user", "content": [{"text": user_prompt(request)}]}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0},
            "toolConfig": self._tool_config(),
        }
        try:
            response = self._session.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}, timeout=60)
            response.raise_for_status()
            content = response.json()["output"]["message"]["content"]
            for item in content:
                if "toolUse" in item and item["toolUse"].get("name") == "return_strategy_interpretation":
                    tool_input = item["toolUse"]["input"]
                    serialized_result = tool_input["interpretation_json"]
                    if not isinstance(serialized_result, str):
                        raise TypeError("interpretation_json must be a JSON string")
                    return self._parse_json_object(serialized_result)
            output = "".join(item["text"] for item in content if "text" in item)
            raise ModelClientFailure("Bedrock did not return the required structured tool result", details={"model": self._model, "output_preview": output[:500]})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelClientFailure("Bedrock returned invalid structured output", details={"exception_type": type(error).__name__, "model": self._model}) from error
        except requests.RequestException as error:
            provider_message = error.response.text if error.response is not None else str(error)
            raise ModelClientFailure("Bedrock strategy interpretation request failed", details={"exception_type": type(error).__name__, "http_status": getattr(error.response, "status_code", None), "provider_message": provider_message, "model": self._model, "region": self._region}) from error

    @staticmethod
    def _system_instruction() -> str:
        schema = json.dumps(ModelInterpretationEnvelope.model_json_schema(), ensure_ascii=False)
        return f"{SYSTEM_PROMPT}\n\nRequired output JSON Schema (follow it exactly):\n{schema}"

    @staticmethod
    def _tool_config() -> dict[str, Any]:
        # Bedrock custom-tool schemas do not support recursive JSON Schema references.
        # The DSL condition tree is deliberately recursive, so transport it as a JSON
        # string and validate the decoded object locally with the complete Pydantic DSL.
        return {
            "tools": [{
                "toolSpec": {
                    "name": "return_strategy_interpretation",
                    "description": "Return the complete strategy interpretation as a single valid JSON object serialized in interpretation_json.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "interpretation_json": {
                                    "type": "string",
                                    "description": "A JSON-serialized StrategyInterpretationResult matching the supplied DSL schema.",
                                }
                            },
                            "required": ["interpretation_json"],
                            "additionalProperties": False,
                        }
                    },
                    "strict": True,
                }
            }],
            "toolChoice": {"tool": {"name": "return_strategy_interpretation"}},
        }

    @staticmethod
    def _parse_json_object(output: str) -> Mapping[str, Any]:
        """Accept a JSON object even when a model wraps it in Markdown prose/fences."""
        cleaned = output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise
            return json.loads(cleaned[start:end + 1])
