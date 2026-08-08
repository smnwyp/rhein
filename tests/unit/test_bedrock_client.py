import json
from types import SimpleNamespace

import pytest
import requests
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient, _bedrock_inventory_schema, _bedrock_transport_schema

class Session:
    def __init__(self): self.call = None
    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        result = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"x","question":"Which window?","target_path":"entry_condition","suggested_answers":[],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
        response = {"output": {"message": {"content": [{"text": json.dumps({"interpretation_json": json.dumps(result)})}]}}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

def test_bedrock_adapter_uses_plain_json_converse_and_bearer_token_without_network():
    session = Session()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))
    assert result["status"] == "clarification_required"
    assert session.call[0].endswith("/converse")
    assert session.call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert "StrategyInterpretationResult JSON object directly" in session.call[1]["json"]["system"][0]["text"]
    assert "toolConfig" not in session.call[1]["json"]
    assert "outputConfig" not in session.call[1]["json"]
    assert session.call[1]["timeout"] == 180


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
                response = {"output": {"message": {"content": [{"text": "I cannot provide JSON today."}]}}, "stopReason": "end_turn"}
            else:
                result = {"status": "clarification_required", "partial_strategy": None, "questions": [{"question_id": "x", "question": "请提供均线周期。", "target_path": "entry_condition", "suggested_answers": [], "answer_kind": "window"}], "ambiguous_terms": ["均线周期"]}
                response = {"output": {"message": {"content": [{"text": json.dumps({"interpretation_json": json.dumps(result)})}]}}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

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
    assert client._max_tokens == 8192
    assert client._timeout_seconds == 180
    truncated = {"output": {"message": {"content": [{"text": '{"status":'}]}}, "stopReason": "max_tokens"}
    with pytest.raises(ModelClientFailure, match="truncated") as error:
        client._extract_interpretation(truncated)
    assert error.value.details["configured_max_tokens"] == 8192


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
