import json
from types import SimpleNamespace
import pytest
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient

class Session:
    def __init__(self): self.call = None
    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        result = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"x","question":"Which window?","target_path":"entry_condition","suggested_answers":[],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
        response = {"output": {"message": {"content": [{"toolUse": {"name": "return_strategy_interpretation", "input": {"interpretation_json": json.dumps(result)}}}]}}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

def test_bedrock_adapter_uses_converse_and_bearer_token_without_network():
    session = Session()
    result = BedrockStrategyModelClient(api_key="test-token", session=session).interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on MA"))
    assert result["status"] == "clarification_required"
    assert session.call[0].endswith("/converse")
    assert session.call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert '"schema_version"' in session.call[1]["json"]["system"][0]["text"]
    assert session.call[1]["json"]["toolConfig"]["toolChoice"]["tool"]["name"] == "return_strategy_interpretation"
    schema = session.call[1]["json"]["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert schema["properties"]["interpretation_json"]["type"] == "string"


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
    truncated = {"output": {"message": {"content": [{"toolUse": {"name": "return_strategy_interpretation", "input": {}}}]}}, "stopReason": "max_tokens"}
    with pytest.raises(ModelClientFailure, match="truncated") as error:
        client._extract_interpretation(truncated)
    assert error.value.details["configured_max_tokens"] == 8192
