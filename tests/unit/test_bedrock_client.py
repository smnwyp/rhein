import json
from types import SimpleNamespace
import pytest
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient, _bedrock_transport_schema

class Session:
    def __init__(self): self.call = None
    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        result = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"x","question":"Which window?","target_path":"entry_condition","suggested_answers":[],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
        response = {"output": {"message": {"content": [{"text": json.dumps({"interpretation_json": json.dumps(result)})}]}}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

def test_bedrock_adapter_uses_converse_and_bearer_token_without_network():
    session = Session()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))
    assert result["status"] == "clarification_required"
    assert session.call[0].endswith("/converse")
    assert session.call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert "inner strategy-interpretation JSON object" in session.call[1]["json"]["system"][0]["text"]
    assert "toolConfig" not in session.call[1]["json"]
    transport_schema = json.loads(session.call[1]["json"]["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert transport_schema == _bedrock_transport_schema()
    assert session.call[1]["timeout"] == 180


def test_bedrock_adapter_accepts_direct_tool_object_and_text_json_fallback():
    client = BedrockStrategyModelClient(api_key="test-token", session=Session())
    direct = {"output": {"message": {"content": [{"toolUse": {"name": "return_strategy_interpretation", "input": {"status": "clarification_required"}}}]}}}
    assert client._extract_interpretation(direct)["status"] == "clarification_required"
    text = {"output": {"message": {"content": [{"text": '{"status":"clarification_required"}'}]}}}
    assert client._extract_interpretation(text)["status"] == "clarification_required"


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
