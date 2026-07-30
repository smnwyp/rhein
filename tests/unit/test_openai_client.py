import json
from types import SimpleNamespace
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.model.openai_client import OpenAIStrategyModelClient

class Responses:
    def __init__(self): self.calls = []
    def create(self, **kwargs): self.calls.append(kwargs); return SimpleNamespace(output_text=json.dumps({"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"x","question":"Which window?","target_path":"entry_condition","suggested_answers":[],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}))
def test_openai_adapter_uses_responses_structured_output_without_network():
    responses = Responses(); client = OpenAIStrategyModelClient(client=SimpleNamespace(responses=responses), model="test-model")
    result = client.interpret_strategy(StrategyInterpretationRequest(strategy_text="Buy on moving average", symbol="AAPL"))
    assert result["status"] == "clarification_required"
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert responses.calls[0]["text"]["format"]["schema"]["type"] == "object"
    assert not {"oneOf", "anyOf", "allOf"}.intersection(responses.calls[0]["text"]["format"]["schema"])
