"""Amazon Bedrock Converse adapter for the provider-independent interpreter client."""
import json
import os
import struct
import zlib
from collections.abc import Mapping
from time import sleep as default_sleep
from typing import Any, Callable
from urllib.parse import quote

import requests

from alpha_agent.domain.interpretation import ModelInterpretationEnvelope, StrategyInterpretationRequest
from alpha_agent.domain.semantic_inventory import SemanticInventory
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.errors import ModelClientFailure
from alpha_agent.parser.prompts import SYSTEM_PROMPT, inventory_system_instruction, inventory_user_prompt, user_prompt


class BedrockStrategyModelClient:
    def __init__(self, *, model: str = "us.anthropic.claude-sonnet-4-6", region: str = "us-east-1", api_key: str | None = None, session: requests.Session | None = None, max_tokens: int = 16384, timeout_seconds: int = 180, transient_retry_attempts: int = 1, retry_backoff_seconds: float = 0.75, sleep: Callable[[float], None] = default_sleep) -> None:
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
                # The final recursive DSL can be substantially larger than the
                # inventory.  Keep its connection alive by consuming Bedrock's
                # event stream, while retaining the exact same local JSON and
                # semantic validation contract below this adapter.
                return self._extract_interpretation(self._post_interpretation_stream(payload))
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

    def interpret_compact_strategy(self, request: StrategyInterpretationRequest) -> Mapping[str, Any]:
        """Compile a large strategy as bare DSL, omitting verbose audit coverage."""
        if not self._api_key:
            raise ModelClientFailure("Bedrock API key is missing", details={"required_env": "AWS_BEARER_TOKEN_BEDROCK"})
        payload = {
            "system": [{"text": self._compact_system_instruction()}],
            "messages": [{"role": "user", "content": [{"text": user_prompt(request)}]}],
            "inferenceConfig": {"maxTokens": self._max_tokens, "temperature": 0},
        }
        return self._extract_interpretation(self._post_interpretation_stream(payload))

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

    def _post_interpretation_stream(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a ConverseStream response reconstructed from text deltas.

        Bedrock streams are AWS EventStream frames, not newline-delimited JSON.
        A connection may close after the complete JSON object but before the
        final ``messageStop`` event; that result is safe to retain.  Otherwise
        retry the whole idempotent compilation request a bounded number of
        times.  We intentionally do not attempt to stitch two partial model
        generations together.
        """
        url = f"https://bedrock-runtime.{self._region}.amazonaws.com/model/{quote(self._model, safe='.:_-')}/converse-stream"
        last_error: Exception | None = None
        last_text = ""
        for attempt in range(self._transient_retry_attempts + 1):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/vnd.amazon.eventstream",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    timeout=self._timeout_seconds,
                    stream=True,
                )
                response.raise_for_status()
                text, stop_reason = self._read_converse_stream(response)
                return self._stream_response_envelope(text, stop_reason)
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error, last_text = error, getattr(error, "streamed_text", "") or last_text
                if last_text and self._is_complete_json_object(last_text):
                    return self._stream_response_envelope(last_text, None)
                if attempt < self._transient_retry_attempts:
                    self._sleep(self._retry_backoff_seconds * (2 ** attempt))
                    continue
                provider_message = error.response.text if error.response is not None else str(error)
                raise ModelClientFailure(
                    "Bedrock transient connection failed while streaming after bounded retry",
                    details={
                        "exception_type": type(error).__name__,
                        "http_status": getattr(error.response, "status_code", None),
                        "provider_message": provider_message,
                        "model": self._model,
                        "region": self._region,
                        "timeout_seconds": self._timeout_seconds,
                        "attempts": attempt + 1,
                        "retryable": True,
                        "streamed_characters": len(last_text),
                    },
                ) from error
            except ModelClientFailure as error:
                # These arrive as normal EventStream frames (rather than an
                # HTTP status) after Bedrock has accepted the request. They
                # are safe to retry because interpretation is side-effect free.
                event_type = error.details.get("event_type") if isinstance(error.details, Mapping) else None
                retryable_events = {
                    "internalServerException", "modelStreamErrorException",
                    "serviceUnavailableException", "throttlingException",
                }
                if event_type in retryable_events and attempt < self._transient_retry_attempts:
                    self._sleep(self._retry_backoff_seconds * (2 ** attempt))
                    continue
                raise
            except requests.RequestException as error:
                provider_message = error.response.text if error.response is not None else str(error)
                raise ModelClientFailure(
                    "Bedrock strategy interpretation streaming request failed",
                    details={
                        "exception_type": type(error).__name__,
                        "http_status": getattr(error.response, "status_code", None),
                        "provider_message": provider_message,
                        "model": self._model,
                        "region": self._region,
                        "timeout_seconds": self._timeout_seconds,
                    },
                ) from error
        raise ModelClientFailure("Bedrock interpretation stream completed without a usable result", details={"model": self._model, "exception_type": type(last_error).__name__ if last_error else None})

    def _read_converse_stream(self, response: Any) -> tuple[str, str | None]:
        """Decode AWS EventStream frames and collect only text deltas."""
        if not hasattr(response, "iter_content"):
            raise ModelClientFailure(
                "Bedrock streaming response does not support incremental reads",
                details={"model": self._model, "response_type": type(response).__name__},
            )
        decoder = _AwsEventStreamDecoder()
        text_parts: list[str] = []
        event_kinds: list[list[str]] = []
        stop_reason: str | None = None
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                for headers, payload in decoder.feed(chunk):
                    event = _decode_stream_event(headers, payload)
                    event_kinds.append(sorted(str(key) for key in event))
                    delta = event.get("contentBlockDelta")
                    if isinstance(delta, Mapping):
                        content_delta = delta.get("delta")
                        if isinstance(content_delta, Mapping) and isinstance(content_delta.get("text"), str):
                            text_parts.append(content_delta["text"])
                    # The bearer-token Bedrock gateway emits the same fields
                    # in a compact flat event (``delta`` / ``stopReason``)
                    # rather than wrapping them under the public API event
                    # name.  Accept it without depending on undocumented
                    # values carried in its auxiliary ``p`` field.
                    flat_delta = event.get("delta")
                    if isinstance(flat_delta, Mapping) and isinstance(flat_delta.get("text"), str):
                        text_parts.append(flat_delta["text"])
                    message_stop = event.get("messageStop")
                    if isinstance(message_stop, Mapping):
                        reason = message_stop.get("stopReason")
                        stop_reason = reason if isinstance(reason, str) else None
                    flat_stop_reason = event.get("stopReason")
                    if isinstance(flat_stop_reason, str):
                        stop_reason = flat_stop_reason
                    for error_key in ("internalServerException", "modelStreamErrorException", "validationException", "throttlingException", "serviceUnavailableException"):
                        if error_key in event:
                            raise ModelClientFailure(
                                "Bedrock returned an error event while streaming strategy interpretation",
                                details={"model": self._model, "event_type": error_key},
                            )
            decoder.finish()
        except (requests.ConnectionError, requests.Timeout) as error:
            # Preserve the completed prefix for the caller's safe JSON check.
            setattr(error, "streamed_text", "".join(text_parts))
            raise
        text = "".join(text_parts)
        if not text and stop_reason is None:
            raise ModelClientFailure(
                "Bedrock stream completed without a text or stop event",
                details={"model": self._model, "event_kinds": event_kinds},
            )
        return text, stop_reason

    @staticmethod
    def _stream_response_envelope(text: str, stop_reason: str | None) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"output": {"message": {"content": [{"text": text}]}}}
        if stop_reason is not None:
            payload["stopReason"] = stop_reason
        return payload

    def _is_complete_json_object(self, text: str) -> bool:
        try:
            self._parse_json_object(text)
        except (json.JSONDecodeError, TypeError):
            return False
        return True

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
    def _compact_system_instruction() -> str:
        schema = json.dumps(TimedStrategyDefinition.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "COMPACT LARGE-DSL MODE. Return only the complete inner timed strategy DSL JSON object, "
            "with no status envelope, coverage, assumptions, warnings, Markdown, or prose. "
            "Local deterministic validation will create the audit mapping. "
            f"The DSL must conform exactly to this JSON Schema:\n{schema}"
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


class _AwsEventStreamDecoder:
    """Minimal, validated AWS EventStream frame decoder for Bedrock text events."""

    _PRELUDE_SIZE = 12
    _MIN_MESSAGE_SIZE = 16

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[Mapping[str, object], bytes]]:
        self._buffer.extend(chunk)
        payloads: list[tuple[Mapping[str, object], bytes]] = []
        while len(self._buffer) >= self._PRELUDE_SIZE:
            total_length, headers_length, expected_prelude_crc = struct.unpack(
                ">III", self._buffer[:self._PRELUDE_SIZE]
            )
            actual_prelude_crc = zlib.crc32(self._buffer[:8]) & 0xFFFFFFFF
            if actual_prelude_crc != expected_prelude_crc:
                raise ModelClientFailure("Bedrock stream prelude checksum failed", details={"phase": "dsl_compilation"})
            if total_length < self._MIN_MESSAGE_SIZE or headers_length > total_length - self._MIN_MESSAGE_SIZE:
                raise ModelClientFailure("Bedrock stream frame has invalid length", details={"phase": "dsl_compilation"})
            if len(self._buffer) < total_length:
                break
            frame = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            expected_message_crc = struct.unpack(">I", frame[-4:])[0]
            actual_message_crc = zlib.crc32(frame[:-4]) & 0xFFFFFFFF
            if actual_message_crc != expected_message_crc:
                raise ModelClientFailure("Bedrock stream frame checksum failed", details={"phase": "dsl_compilation"})
            payload_start = self._PRELUDE_SIZE + headers_length
            payloads.append((_decode_event_headers(frame[self._PRELUDE_SIZE:payload_start]), frame[payload_start:-4]))
        return payloads

    def finish(self) -> None:
        if self._buffer:
            raise requests.ConnectionError("Bedrock stream ended with an incomplete event frame")


def _decode_event_headers(encoded: bytes) -> Mapping[str, object]:
    """Decode the subset of AWS EventStream headers used by Bedrock."""
    headers: dict[str, object] = {}
    offset = 0
    while offset < len(encoded):
        if offset + 2 > len(encoded):
            raise ModelClientFailure("Bedrock stream header was truncated", details={"phase": "dsl_compilation"})
        name_length = encoded[offset]
        offset += 1
        if offset + name_length + 1 > len(encoded):
            raise ModelClientFailure("Bedrock stream header has invalid length", details={"phase": "dsl_compilation"})
        name = encoded[offset:offset + name_length].decode("utf-8")
        offset += name_length
        value_type = encoded[offset]
        offset += 1
        if value_type == 0:
            headers[name] = True
        elif value_type == 1:
            headers[name] = False
        elif value_type in {2, 3, 4, 5}:
            size = {2: 1, 3: 2, 4: 4, 5: 8}[value_type]
            if offset + size > len(encoded):
                raise ModelClientFailure("Bedrock stream numeric header was truncated", details={"phase": "dsl_compilation"})
            headers[name] = int.from_bytes(encoded[offset:offset + size], "big", signed=True)
            offset += size
        elif value_type in {6, 7}:
            if offset + 2 > len(encoded):
                raise ModelClientFailure("Bedrock stream variable header was truncated", details={"phase": "dsl_compilation"})
            size = int.from_bytes(encoded[offset:offset + 2], "big")
            offset += 2
            if offset + size > len(encoded):
                raise ModelClientFailure("Bedrock stream variable header has invalid length", details={"phase": "dsl_compilation"})
            value = bytes(encoded[offset:offset + size])
            headers[name] = value.decode("utf-8") if value_type == 7 else value
            offset += size
        elif value_type == 8:
            if offset + 8 > len(encoded):
                raise ModelClientFailure("Bedrock stream timestamp header was truncated", details={"phase": "dsl_compilation"})
            headers[name] = int.from_bytes(encoded[offset:offset + 8], "big", signed=True)
            offset += 8
        elif value_type == 9:
            if offset + 16 > len(encoded):
                raise ModelClientFailure("Bedrock stream UUID header was truncated", details={"phase": "dsl_compilation"})
            headers[name] = bytes(encoded[offset:offset + 16])
            offset += 16
        else:
            raise ModelClientFailure("Bedrock stream header has unsupported type", details={"phase": "dsl_compilation", "header_type": value_type})
    return headers


def _decode_stream_event(headers: Mapping[str, object], payload: bytes) -> Mapping[str, Any]:
    """Decode a single Bedrock event payload without exposing provider prose."""
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelClientFailure(
            "Bedrock stream event was not valid JSON",
            details={"exception_type": type(error).__name__, "phase": "dsl_compilation"},
        ) from error
    if not isinstance(decoded, Mapping):
        raise ModelClientFailure(
            "Bedrock stream event had an unexpected shape",
            details={"response_type": type(decoded).__name__, "phase": "dsl_compilation"},
        )
    event_type = headers.get(":event-type")
    compact_payload = decoded.get("p")
    if isinstance(event_type, str) and isinstance(compact_payload, Mapping):
        # The Bedrock bearer-token gateway uses the standard AWS EventStream
        # event type header and a compact ``{"p": ...}`` payload.  Public
        # Bedrock examples often show the expanded object directly; accept
        # both forms so the client also works against the native runtime.
        return {event_type: compact_payload}
    return decoded


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
