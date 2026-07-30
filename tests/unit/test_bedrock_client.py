import json
from types import SimpleNamespace
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
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
