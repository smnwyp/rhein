import json
import struct
import zlib
from types import SimpleNamespace

import pytest
import requests
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient, _bedrock_inventory_schema, _bedrock_transport_schema


def _event_frame(event: object) -> bytes:
    payload = json.dumps(event).encode("utf-8")
    total_length = 16 + len(payload)
    prelude_without_crc = struct.pack(">II", total_length, 0)
    prelude = prelude_without_crc + struct.pack(">I", zlib.crc32(prelude_without_crc) & 0xFFFFFFFF)
    body = prelude + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _stream_response(*text_parts: str, stop_reason: str = "end_turn", disconnect_after: bool = False):
    frames = [
        _event_frame({"contentBlockDelta": {"delta": {"text": text}}})
        for text in text_parts
    ]
    if not disconnect_after:
        frames.append(_event_frame({"messageStop": {"stopReason": stop_reason}}))

    def iter_content(*, chunk_size: int):
        yield from frames
        if disconnect_after:
            raise requests.ConnectionError("Remote end closed connection")

    return SimpleNamespace(raise_for_status=lambda: None, iter_content=iter_content)


def _gateway_stream_response(*text_parts: str):
    """Match the compact event shape returned by Bedrock bearer auth."""
    frames = [
        _event_frame({"p": "contentBlockDelta", "contentBlockIndex": 0, "delta": {"text": text}})
        for text in text_parts
    ]
    frames.append(_event_frame({"p": "messageStop", "stopReason": "end_turn"}))
    return SimpleNamespace(raise_for_status=lambda: None, iter_content=lambda *, chunk_size: iter(frames))


class Session:
    def __init__(self): self.call = None
    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        result = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"x","question":"Which window?","target_path":"entry_condition","suggested_answers":[],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
        response = {"output": {"message": {"content": [{"text": json.dumps({"interpretation_json": json.dumps(result)})}]}}}
        if kwargs.get("stream"):
            return _stream_response(response["output"]["message"]["content"][0]["text"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

def test_bedrock_adapter_uses_event_stream_for_final_dsl_and_bearer_token_without_network():
    session = Session()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))
    assert result["status"] == "clarification_required"
    assert session.call[0].endswith("/converse-stream")
    assert session.call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert "StrategyInterpretationResult JSON object directly" in session.call[1]["json"]["system"][0]["text"]
    assert "toolConfig" not in session.call[1]["json"]
    assert "outputConfig" not in session.call[1]["json"]
    assert session.call[1]["timeout"] == 180
    assert session.call[1]["stream"] is True
    assert session.call[1]["headers"]["Accept"] == "application/vnd.amazon.eventstream"


def test_bedrock_adapter_accepts_direct_tool_object_and_text_json_fallback():
    client = BedrockStrategyModelClient(api_key="test-token", session=Session())
    direct = {"output": {"message": {"content": [{"toolUse": {"name": "return_strategy_interpretation", "input": {"status": "clarification_required"}}}]}}}
    assert client._extract_interpretation(direct)["status"] == "clarification_required"
    text = {"output": {"message": {"content": [{"text": '{"status":"clarification_required"}'}]}}}
    assert client._extract_interpretation(text)["status"] == "clarification_required"


def test_bedrock_adapter_recovers_a_json_object_surrounded_by_provider_prose():
    client = BedrockStrategyModelClient(api_key="test-token", session=Session())
    assert client._parse_json_object('Here is the result: {"status":"parsed"}\nThank you.') == {"status": "parsed"}


def test_bedrock_adapter_retries_once_when_the_provider_returns_non_json_text():
    class QueuedSession(Session):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            self.call = (url, kwargs)
            if self.calls == 1:
                return _stream_response("I cannot provide JSON today.")
            else:
                result = {"status": "clarification_required", "partial_strategy": None, "questions": [{"question_id": "x", "question": "请提供均线周期。", "target_path": "entry_condition", "suggested_answers": [], "answer_kind": "window"}], "ambiguous_terms": ["均线周期"]}
                return _stream_response(json.dumps({"interpretation_json": json.dumps(result)}))

    session = QueuedSession()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))
    assert result["status"] == "clarification_required"
    assert session.calls == 2
    assert "FORMAT-ONLY RETRY" in session.call[1]["json"]["messages"][0]["content"][0]["text"]


def test_bedrock_adapter_reports_response_shape_not_a_bare_key_error():
    client = BedrockStrategyModelClient(api_key="test-token", session=Session())
    with pytest.raises(ModelClientFailure) as error:
        client._extract_interpretation({"stopReason": "tool_use"})
    assert error.value.details["response_keys"] == ["stopReason"]


def test_bedrock_adapter_uses_sufficient_output_limit_and_identifies_truncation():
    client = BedrockStrategyModelClient(api_key="test-token", session=Session())
    assert client._max_tokens == 16384
    assert client._timeout_seconds == 180
    truncated = {"output": {"message": {"content": [{"text": '{"status":'}]}}, "stopReason": "max_tokens"}
    with pytest.raises(ModelClientFailure, match="truncated") as error:
        client._extract_interpretation(truncated)
    assert error.value.details["configured_max_tokens"] == 16384


def test_bedrock_structured_output_transport_schema_is_small_and_non_recursive():
    assert _bedrock_transport_schema() == {
        "type": "object",
        "properties": {"interpretation_json": {"type": "string"}},
        "required": ["interpretation_json"],
        "additionalProperties": False,
    }


def test_bedrock_adapter_uses_compact_semantic_inventory_request():
    session = Session()
    client = BedrockStrategyModelClient(api_key="test-token", session=session)
    inventory = {
        "status": "compile_eligible",
        "symbols": [],
        "items": [{"semantic_id": "I-ENTRY", "source_clause_ids": ["C01"], "category": "entry_rule", "pseudo_dsl": "enter", "dependencies": [], "required_capabilities": [], "disposition": "resolved"}],
        "closure": [{"dimension": "temporal", "status": "closed", "explanation": "explicit", "source_clause_ids": ["C01"], "semantic_ids": ["I-ENTRY"]}],
        "questions": [], "ambiguous_terms": [], "unsupported_features": [],
    }
    response = {"output": {"message": {"content": [{"text": json.dumps(inventory)}]}}}
    session.post = lambda url, **kwargs: (setattr(session, "call", (url, kwargs)) or SimpleNamespace(raise_for_status=lambda: None, json=lambda: response))

    result = client.inspect_strategy(StrategyInterpretationRequest(strategy_text="Buy", source_clauses=[{"clause_id": "C01", "text": "Buy"}]))

    assert result["status"] == "compile_eligible"
    assert session.call[1]["json"]["inferenceConfig"]["maxTokens"] == 8192
    schema = json.loads(session.call[1]["json"]["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert schema["type"] == "object"
    assert "status" in schema["properties"]


def test_bedrock_inventory_schema_strips_provider_unsupported_cardinality_keywords():
    serialized = json.dumps(_bedrock_inventory_schema())
    assert "maxItems" not in serialized
    assert "minItems" not in serialized
    assert "additionalProperties" in serialized


def test_bedrock_adapter_retries_one_transient_connection_failure_before_succeeding():
    class FlakySession(Session):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("Remote end closed connection")
            return super().post(url, **kwargs)

    delays: list[float] = []
    session = FlakySession()
    client = BedrockStrategyModelClient(
        api_key="test-token",
        session=session,
        transient_retry_attempts=1,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    )

    result = client.interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))

    assert result["status"] == "clarification_required"
    assert session.calls == 2
    assert delays == [0.25]


def test_bedrock_adapter_stops_after_bounded_transient_connection_retries():
    class FailingSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            raise requests.ConnectionError("Remote end closed connection")

    delays: list[float] = []
    session = FailingSession()
    client = BedrockStrategyModelClient(
        api_key="test-token",
        session=session,
        transient_retry_attempts=1,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    )

    with pytest.raises(ModelClientFailure, match="transient connection") as error:
        client.interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))

    assert session.calls == 2
    assert delays == [0.25]
    assert error.value.details["attempts"] == 2
    assert error.value.details["retryable"] is True


def test_bedrock_adapter_uses_complete_json_received_before_stream_disconnect():
    class CompletedThenDisconnectedSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            result = {"status": "clarification_required", "partial_strategy": None, "questions": [], "ambiguous_terms": []}
            return _stream_response(json.dumps(result), disconnect_after=True)

    session = CompletedThenDisconnectedSession()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(
        StrategyInterpretationRequest(strategy_text="Buy on MA")
    )

    assert result["status"] == "clarification_required"
    assert session.calls == 1


def test_bedrock_adapter_retries_incomplete_stream_before_succeeding():
    class IncompleteThenWorkingSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _stream_response('{"status":"clarification_', disconnect_after=True)
            result = {"status": "clarification_required", "partial_strategy": None, "questions": [], "ambiguous_terms": []}
            return _stream_response(json.dumps(result))

    delays: list[float] = []
    session = IncompleteThenWorkingSession()
    result = BedrockStrategyModelClient(
        api_key="test-token", session=session, retry_backoff_seconds=0.25, sleep=delays.append,
    ).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))

    assert result["status"] == "clarification_required"
    assert session.calls == 2
    assert delays == [0.25]


def test_bedrock_adapter_accepts_bearer_gateway_flat_stream_events():
    class GatewaySession:
        def post(self, url, **kwargs):
            result = {"status": "clarification_required", "partial_strategy": None, "questions": [], "ambiguous_terms": []}
            return _gateway_stream_response(json.dumps(result))

    result = BedrockStrategyModelClient(api_key="test-token", session=GatewaySession()).interpret_strategy(
        StrategyInterpretationRequest(strategy_text="Buy on MA")
    )

    assert result["status"] == "clarification_required"


def test_bedrock_adapter_retries_retryable_server_event_from_stream():
    class ServerErrorThenWorkingSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    raise_for_status=lambda: None,
                    iter_content=lambda *, chunk_size: iter([_event_frame({"serviceUnavailableException": {}})]),
                )
            result = {"status": "clarification_required", "partial_strategy": None, "questions": [], "ambiguous_terms": []}
            return _gateway_stream_response(json.dumps(result))

    delays: list[float] = []
    session = ServerErrorThenWorkingSession()
    result = BedrockStrategyModelClient(
        api_key="test-token", session=session, retry_backoff_seconds=0.25, sleep=delays.append,
    ).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))

    assert result["status"] == "clarification_required"
    assert session.calls == 2
    assert delays == [0.25]
