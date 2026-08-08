"""Amazon Bedrock Converse adapter for the provider-independent interpreter client."""
import json
import os
from collections.abc import Mapping
from time import sleep as default_sleep
from typing import Any, Callable
from urllib.parse import quote

import requests

from alpha_agent.domain.interpretation import ModelInterpretationEnvelope, StrategyInterpretationRequest
from alpha_agent.domain.semantic_inventory import SemanticInventory
from alpha_agent.errors import ModelClientFailure
from alpha_agent.parser.prompts import SYSTEM_PROMPT, inventory_system_instruction, inventory_user_prompt, user_prompt


class BedrockStrategyModelClient:
    def __init__(self, *, model: str = "us.anthropic.claude-sonnet-4-6", region: str = "us-east-1", api_key: str | None = None, session: requests.Session | None = None, max_tokens: int = 8192, timeout_seconds: int = 180, transient_retry_attempts: int = 1, retry_backoff_seconds: float = 0.75, sleep: Callable[[float], None] = default_sleep) -> None:
        if transient_retry_attempts < 0:
            raise ValueError("transient_retry_attempts must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self._model, self._region = model, region
        self._api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self._session = session or requests.Session()
        self._max_tokens = max_tokens
        # Inventory responses are deliberately compact by contract, but a
        # complex strategy can still need more than 4k output tokens.  Use the
        # same bounded ceiling as the final response; providers charge actual
        # generated tokens, not this unused headroom.
        self._inventory_max_tokens = min(max_tokens, 8192)
        self._timeout_seconds = timeout_seconds
        self._transient_retry_attempts = transient_retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep

    def interpret_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        if not self._api_key:
            raise ModelClientFailure("Bedrock API key is missing", details={"required_env": "AWS_BEARER_TOKEN_BEDROCK"})
        for format_attempt in range(2):
            attempt_request = request if format_attempt == 0 else self._format_retry_request(request)
            payload = {
                "system": [{"text": self._system_instruction()}],
                "messages": [{"role": "user", "content": [{"text": user_prompt(attempt_request)}]}],
                # A timed DSL plus one source-coverage item per original clause can
                # exceed 4k tokens.  A truncation is worse than a larger upper bound:
                # it costs a full failed request and cannot be safely recovered.
                "inferenceConfig": {"maxTokens": self._max_tokens, "temperature": 0},
            }
            try:
                return self._extract_interpretation(self._post_payload(payload))
            except ModelClientFailure as error:
                if format_attempt == 0 and self._is_format_failure(error):
                    continue
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                # A decoder failure is a format failure, not a strategy
                # failure. Give the provider one bounded, format-only retry.
                if format_attempt == 0:
                    continue
                raise ModelClientFailure(
                    "Bedrock returned malformed JSON for the strategy interpretation",
                    details={"exception_type": type(error).__name__, "model": self._model, "attempts": format_attempt + 1},
                ) from error
        raise ModelClientFailure(
            "Bedrock interpretation completed without a usable result",
            details={"model": self._model, "phase": "interpretation_format_retry", "attempts": 2},
        )

    def inspect_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        """Run the compact semantic-inventory phase before recursive DSL lowering."""
        if not self._api_key:
            raise ModelClientFailure("Bedrock API key is missing", details={"required_env": "AWS_BEARER_TOKEN_BEDROCK"})
        for format_attempt in range(2):
            attempt_request = request if format_attempt == 0 else self._format_retry_request(request)
            payload = {
                "system": [{"text": inventory_system_instruction()}],
                "messages": [{"role": "user", "content": [{"text": inventory_user_prompt(attempt_request)}]}],
                "inferenceConfig": {"maxTokens": self._inventory_max_tokens, "temperature": 0},
                "outputConfig": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
                    "name": "semantic_inventory",
                    "description": "A SemanticInventory conforming to the supplied schema.",
                    # SemanticInventory has no recursive condition tree, so it
                    # can use Bedrock's native schema directly. This avoids the
                    # fragile JSON-inside-a-JSON-string transport.
                    "schema": json.dumps(_bedrock_inventory_schema(), ensure_ascii=False, separators=(",", ":")),
                }}}},
            }
            try:
                return self._extract_inventory(self._post_payload(payload))
            except ModelClientFailure as error:
                if format_attempt == 0 and self._is_format_failure(error):
                    continue
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                if format_attempt == 0:
                    continue
                raise ModelClientFailure(
                    "Bedrock returned malformed JSON for the semantic inventory",
                    details={"exception_type": type(error).__name__, "model": self._model, "attempts": format_attempt + 1},
                ) from error
        raise ModelClientFailure(
            "Bedrock semantic inventory completed without a usable result",
            details={"model": self._model, "phase": "inventory_format_retry", "attempts": 2},
        )

    @staticmethod
    def _format_retry_request(request: StrategyInterpretationRequest) -> StrategyInterpretationRequest:
        instruction = (
            "FORMAT-ONLY RETRY: your prior response could not be parsed. Preserve the original strategy meaning, "
            "but return exactly the requested JSON object with double-quoted keys and values. "
            "Do not use Markdown fences, explanation, or any text before or after the JSON."
        )
        combined = f"{request.repair_instruction}\n{instruction}" if request.repair_instruction else instruction
        return request.model_copy(update={"repair_instruction": combined})

    @staticmethod
    def _is_format_failure(error: ModelClientFailure) -> bool:
        return error.message in {
            "Bedrock returned malformed JSON for the strategy interpretation",
            "Bedrock did not return a usable strategy interpretation",
            "Bedrock returned an invalid semantic inventory envelope",
            "Bedrock returned malformed JSON for the semantic inventory",
            "Bedrock did not return a usable semantic inventory",
        }

    def _post_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"https://bedrock-runtime.{self._region}.amazonaws.com/model/{quote(self._model, safe='.:_-')}/converse"
        for attempt in range(self._transient_retry_attempts + 1):
            try:
                response = self._session.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}, timeout=self._timeout_seconds)
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, Mapping):
                    raise ModelClientFailure("Bedrock returned an unexpected response envelope", details={"model": self._model, "response_type": type(response_payload).__name__})
                return response_payload
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt < self._transient_retry_attempts:
                    self._sleep(self._retry_backoff_seconds * (2 ** attempt))
                    continue
                provider_message = error.response.text if error.response is not None else str(error)
                raise ModelClientFailure(
                    "Bedrock transient connection failed after bounded retry",
                    details={
                        "exception_type": type(error).__name__, "http_status": getattr(error.response, "status_code", None),
                        "provider_message": provider_message, "model": self._model, "region": self._region,
                        "timeout_seconds": self._timeout_seconds, "attempts": attempt + 1, "retryable": True,
                    },
                ) from error
            except requests.RequestException as error:
                provider_message = error.response.text if error.response is not None else str(error)
                raise ModelClientFailure("Bedrock strategy interpretation request failed", details={"exception_type": type(error).__name__, "http_status": getattr(error.response, "status_code", None), "provider_message": provider_message, "model": self._model, "region": self._region, "timeout_seconds": self._timeout_seconds}) from error

    def _extract_interpretation(self, response_payload: object) -> Mapping[str, Any]:
        """Read raw JSON text, while retaining compatibility with old tool output.

        Bedrock custom-tool schemas cannot express the recursive DSL condition
        tree: they reject both recursive references and open object properties.
        Sending the payload through a string tool field required double JSON
        escaping and was fragile for long strategies. The current transport is
        plain model text; local Pydantic, semantic, and coverage checks remain
        the authoritative structured-output contract.
        """
        if not isinstance(response_payload, Mapping):
            raise ModelClientFailure("Bedrock returned an unexpected response envelope", details={"model": self._model, "response_type": type(response_payload).__name__})
        output = response_payload.get("output")
        message = output.get("message") if isinstance(output, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            raise ModelClientFailure("Bedrock returned an unexpected response envelope", details=self._response_shape(response_payload))
        text_blocks: list[str] = []
        parse_error: json.JSONDecodeError | None = None
        for item in content:
            if not isinstance(item, Mapping):
                continue
            tool_use = item.get("toolUse")
            if isinstance(tool_use, Mapping) and tool_use.get("name") == "return_strategy_interpretation":
                tool_input = tool_use.get("input")
                if isinstance(tool_input, Mapping):
                    serialized = tool_input.get("interpretation_json")
                    if isinstance(serialized, str):
                        try:
                            return self._parse_json_object(serialized)
                        except json.JSONDecodeError as error:
                            # Backwards-compatible support for the former
                            # string wrapper.  Do not mask a truncation as an
                            # opaque decoder error.
                            parse_error = error
                    # Tolerate a provider/model emitting the interpretation object
                    # directly instead of the requested string wrapper.
                    if "status" in tool_input:
                        return dict(tool_input)
            block_text = item.get("text")
            if isinstance(block_text, str):
                text_blocks.append(block_text)
        if text_blocks:
            try:
                parsed_text = self._parse_json_object("\n".join(text_blocks))
                serialized = parsed_text.get("interpretation_json") if isinstance(parsed_text, Mapping) else None
                return self._parse_json_object(serialized) if isinstance(serialized, str) else parsed_text
            except json.JSONDecodeError as error:
                parse_error = error
        if response_payload.get("stopReason") == "max_tokens":
            raise ModelClientFailure(
                "Bedrock truncated the strategy interpretation before it completed",
                details={**self._response_shape(response_payload, content), "configured_max_tokens": self._max_tokens},
            )
        if parse_error is not None:
            raise ModelClientFailure(
                "Bedrock returned malformed JSON for the strategy interpretation",
                details={**self._response_shape(response_payload, content), "exception_type": type(parse_error).__name__},
            ) from parse_error
        raise ModelClientFailure("Bedrock did not return a usable strategy interpretation", details=self._response_shape(response_payload, content))

    def _extract_string_envelope(self, response_payload: Mapping[str, Any], field_name: str, purpose: str) -> Mapping[str, Any]:
        """Extract a compact structured response whose inner object is a JSON string."""
        output = response_payload.get("output")
        message = output.get("message") if isinstance(output, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            raise ModelClientFailure(f"Bedrock returned an unexpected {purpose} envelope", details=self._response_shape(response_payload))
        text = "\n".join(item["text"] for item in content if isinstance(item, Mapping) and isinstance(item.get("text"), str))
        if response_payload.get("stopReason") == "max_tokens":
            raise ModelClientFailure(f"Bedrock truncated the {purpose}", details={**self._response_shape(response_payload, content), "configured_max_tokens": self._inventory_max_tokens})
        if not text:
            raise ModelClientFailure(f"Bedrock did not return a usable {purpose}", details=self._response_shape(response_payload, content))
        envelope = self._parse_json_object(text)
        inner = envelope.get(field_name)
        if not isinstance(inner, str):
            raise ModelClientFailure(f"Bedrock returned an invalid {purpose} envelope", details=self._response_shape(response_payload, content))
        return self._parse_json_object(inner)

    def _extract_inventory(self, response_payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Read native Inventory JSON, accepting the former string wrapper."""
        output = response_payload.get("output")
        message = output.get("message") if isinstance(output, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            raise ModelClientFailure("Bedrock returned an unexpected semantic inventory envelope", details=self._response_shape(response_payload))
        if response_payload.get("stopReason") == "max_tokens":
            raise ModelClientFailure("Bedrock truncated the semantic inventory", details={**self._response_shape(response_payload, content), "configured_max_tokens": self._inventory_max_tokens})
        text = "\n".join(item["text"] for item in content if isinstance(item, Mapping) and isinstance(item.get("text"), str))
        if not text:
            raise ModelClientFailure("Bedrock did not return a usable semantic inventory", details=self._response_shape(response_payload, content))
        try:
            parsed = self._parse_json_object(text)
        except json.JSONDecodeError as error:
            raise ModelClientFailure(
                "Bedrock returned malformed JSON for the semantic inventory",
                details={**self._response_shape(response_payload, content), "exception_type": type(error).__name__},
            ) from error
        wrapped = parsed.get("inventory_json")
        if not isinstance(wrapped, str):
            return parsed
        try:
            return self._parse_json_object(wrapped)
        except json.JSONDecodeError as error:
            raise ModelClientFailure(
                "Bedrock returned malformed JSON for the semantic inventory",
                details={**self._response_shape(response_payload, content), "exception_type": type(error).__name__},
            ) from error

    def _response_shape(self, payload: Mapping[str, Any], content: list[object] | None = None) -> dict[str, object]:
        """Diagnostic metadata only; never return a raw model response to the UI."""
        blocks = content
        if blocks is None:
            output = payload.get("output")
            message = output.get("message") if isinstance(output, Mapping) else None
            candidate = message.get("content") if isinstance(message, Mapping) else None
            blocks = candidate if isinstance(candidate, list) else []
        text_blocks = [item.get("text") for item in blocks if isinstance(item, Mapping) and isinstance(item.get("text"), str)]
        return {
            "model": self._model,
            "response_keys": sorted(str(key) for key in payload),
            "content_block_kinds": [sorted(str(key) for key in item) if isinstance(item, Mapping) else type(item).__name__ for item in blocks],
            "stop_reason": payload.get("stopReason"),
            # Diagnostics intentionally describe structure only: provider text
            # may contain private strategy prose and must not be echoed into UI.
            "text_block_character_counts": [len(text) for text in text_blocks],
            "text_blocks_start_with_json": [text.lstrip().startswith(("{", "[")) for text in text_blocks],
        }

    @staticmethod
    def _system_instruction() -> str:
        schema = json.dumps(ModelInterpretationEnvelope.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Return the complete StrategyInterpretationResult JSON object directly. "
            "Do not wrap it in `interpretation_json`, do not encode it as a JSON string, "
            "and do not add Markdown or prose. The following JSON Schema describes the "
            f"object that local validation will enforce exactly:\n{schema}"
        )

    @staticmethod
    def _parse_json_object(output: str) -> Mapping[str, Any]:
        """Accept a JSON object even when a model wraps it in Markdown prose/fences."""
        cleaned = output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].strip()
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, Mapping):
                raise TypeError("expected JSON object")
            return parsed
        except json.JSONDecodeError as original_error:
            # Providers occasionally add one prose sentence before/after an
            # otherwise complete JSON object.  ``raw_decode`` locates a full
            # object without the fragile first-{ / last-} slicing approach.
            decoder = json.JSONDecoder()
            for offset, character in enumerate(cleaned):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(cleaned[offset:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping):
                    return parsed
            raise original_error


def _bedrock_transport_schema() -> dict[str, Any]:
    """Small non-recursive schema that Bedrock can compile reliably.

    The embedded value is parsed immediately against the full recursive
    Pydantic contract, so this transport envelope never weakens DSL guards.
    """
    return _bedrock_string_transport_schema("interpretation_json")


def _bedrock_inventory_schema() -> dict[str, Any]:
    """Compile Pydantic's inventory schema to Bedrock's supported subset.

    Bedrock rejects array/string cardinality constraints such as ``maxItems``.
    Those constraints remain authoritative in local Pydantic validation after
    transport; the provider schema only constrains JSON shape and closed object
    fields. Keeping this compiler explicit prevents provider-specific 400s
    from being mistaken for strategy interpretation failures.
    """
    unsupported_keywords = {"maxItems", "minItems", "maxLength", "minLength", "pattern"}

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(child) for key, child in value.items() if key not in unsupported_keywords}
        if isinstance(value, list):
            return [strip(child) for child in value]
        return value

    schema = strip(SemanticInventory.model_json_schema())
    if not isinstance(schema, dict):
        # Defensive only: never leak an AssertionError from a provider-boundary
        # helper into the user-facing interpreter.
        raise ModelClientFailure(
            "could not compile the semantic-inventory transport schema",
            details={"schema_type": type(schema).__name__},
        )
    return schema


def _bedrock_string_transport_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field_name: {"type": "string"}},
        "required": [field_name],
        "additionalProperties": False,
    }
