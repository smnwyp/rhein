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
    def __init__(self, *, model: str = "us.anthropic.claude-sonnet-4-6", region: str = "us-east-1", api_key: str | None = None, session: requests.Session | None = None, max_tokens: int = 8192) -> None:
        self._model, self._region = model, region
        self._api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self._session = session or requests.Session()
        self._max_tokens = max_tokens

    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        if not self._api_key:
            raise ModelClientFailure("Bedrock API key is missing", details={"required_env": "AWS_BEARER_TOKEN_BEDROCK"})
        url = f"https://bedrock-runtime.{self._region}.amazonaws.com/model/{quote(self._model, safe='.:_-')}/converse"
        payload = {
            "system": [{"text": self._system_instruction()}],
            "messages": [{"role": "user", "content": [{"text": user_prompt(request)}]}],
            # A timed DSL plus one source-coverage item per original clause can
            # exceed 4k tokens.  A truncation is worse than a larger upper bound:
            # it costs a full failed request and cannot be safely recovered.
            "inferenceConfig": {"maxTokens": self._max_tokens, "temperature": 0},
            "toolConfig": self._tool_config(),
        }
        try:
            response = self._session.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}, timeout=60)
            response.raise_for_status()
            return self._extract_interpretation(response.json())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelClientFailure("Bedrock returned invalid structured output", details={"exception_type": type(error).__name__, "model": self._model}) from error
        except requests.RequestException as error:
            provider_message = error.response.text if error.response is not None else str(error)
            raise ModelClientFailure("Bedrock strategy interpretation request failed", details={"exception_type": type(error).__name__, "http_status": getattr(error.response, "status_code", None), "provider_message": provider_message, "model": self._model, "region": self._region}) from error

    def _extract_interpretation(self, response_payload: object) -> Mapping[str, Any]:
        """Accept Bedrock tool use and the text-JSON fallback without KeyErrors.

        Bedrock model families occasionally emit a text block despite a forced
        custom tool. The fallback remains safe because the service performs the
        full strict Pydantic + semantic validation after this adapter returns.
        """
        if not isinstance(response_payload, Mapping):
            raise ModelClientFailure("Bedrock returned an unexpected response envelope", details={"model": self._model, "response_type": type(response_payload).__name__})
        output = response_payload.get("output")
        message = output.get("message") if isinstance(output, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            raise ModelClientFailure("Bedrock returned an unexpected response envelope", details=self._response_shape(response_payload))
        text_blocks: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            tool_use = item.get("toolUse")
            if isinstance(tool_use, Mapping) and tool_use.get("name") == "return_strategy_interpretation":
                tool_input = tool_use.get("input")
                if isinstance(tool_input, Mapping):
                    serialized = tool_input.get("interpretation_json")
                    if isinstance(serialized, str):
                        return self._parse_json_object(serialized)
                    # Tolerate a provider/model emitting the interpretation object
                    # directly instead of the requested string wrapper.
                    if "status" in tool_input:
                        return dict(tool_input)
            block_text = item.get("text")
            if isinstance(block_text, str):
                text_blocks.append(block_text)
        if text_blocks:
            try:
                return self._parse_json_object("\n".join(text_blocks))
            except json.JSONDecodeError:
                pass
        if response_payload.get("stopReason") == "max_tokens":
            raise ModelClientFailure(
                "Bedrock truncated the strategy interpretation before it completed",
                details={**self._response_shape(response_payload, content), "configured_max_tokens": self._max_tokens},
            )
        raise ModelClientFailure("Bedrock did not return a usable strategy interpretation", details=self._response_shape(response_payload, content))

    def _response_shape(self, payload: Mapping[str, Any], content: list[object] | None = None) -> dict[str, object]:
        """Diagnostic metadata only; never return a raw model response to the UI."""
        blocks = content
        if blocks is None:
            output = payload.get("output")
            message = output.get("message") if isinstance(output, Mapping) else None
            candidate = message.get("content") if isinstance(message, Mapping) else None
            blocks = candidate if isinstance(candidate, list) else []
        return {
            "model": self._model,
            "response_keys": sorted(str(key) for key in payload),
            "content_block_kinds": [sorted(str(key) for key in item) if isinstance(item, Mapping) else type(item).__name__ for item in blocks],
            "stop_reason": payload.get("stopReason"),
        }

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
