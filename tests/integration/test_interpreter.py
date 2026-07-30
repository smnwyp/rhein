import pytest
from alpha_agent.domain.interpretation import ClarificationRequired, ParsedStrategy, StrategyInterpretationRequest
from alpha_agent.errors import ModelClientFailure, ModelResponseParsingFailure, SemanticStrategyValidationFailure
from alpha_agent.model.fake_client import FakeModelClient
from alpha_agent.monitoring import InMemoryInterpreterMonitor, InterpretationEvaluation
from alpha_agent.parser.service import StrategyInterpreterService

def f(name): return {"kind":"market_field","field":name}
def i(kind, window): return {"kind":"indicator","indicator":{"indicator":kind,"field":"volume" if kind == "rolling_mean" else "close","window":window}}
def s(value): return {"kind":"scalar","value":value}
def cmp(left,right,op="greater_than"): return {"node_type":"comparison","operator":op,"left":left,"right":right}
def cross(left,right,op): return {"node_type":"cross","operator":op,"left":left,"right":right}
def group(op,*items): return {"node_type":"group","operator":op,"conditions":list(items)}
def parsed(entry, exit): return {"status":"parsed","strategy":{"schema_version":"0.1","symbol":"AAPL","frequency":"1d","direction":"long_only","position_mode":"fully_invested_or_flat","entry_condition":entry,"exit_condition":exit},"assumptions":[],"warnings":[]}
@pytest.mark.parametrize("response", [
    parsed(cross(f("close"),i("sma",20),"cross_above"),cross(f("close"),i("sma",20),"cross_below")),
    parsed(group("and",cross(i("sma",10),i("sma",50),"cross_above"),cmp(f("volume"),i("rolling_mean",20))),cross(i("sma",10),i("sma",50),"cross_below")),
    parsed(cmp(i("rsi",14),s(30),"less_than"),cmp(i("rsi",14),s(70))),
    parsed(group("and",cmp(i("rolling_return",20),s(.05)),cmp(f("volume"),{"kind":"scaled_operand","operand":i("rolling_mean",20),"multiplier":1.5})),cmp(i("rolling_return",20),s(0),"less_than")),
])
def test_required_successful_examples(response):
    result = StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test",symbol="AAPL"))
    assert isinstance(result, ParsedStrategy)
def test_clarification_and_partial_strategy_are_validated():
    response = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"decline","question":"What period and percent?","target_path":"entry_condition","suggested_answers":["5-day return below -5%"],"answer_kind":"choice"}],"ambiguous_terms":["fallen significantly","volume increasing"]}
    assert isinstance(StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="ambiguous")), ClarificationRequired)
    invalid = {**response,"partial_strategy":parsed(cmp(i("rsi",14),s(101)),cmp(f("close"),i("sma",20)))["strategy"]}
    with pytest.raises(SemanticStrategyValidationFailure): StrategyInterpreterService(FakeModelClient([invalid])).interpret(StrategyInterpretationRequest(strategy_text="ambiguous"))


def test_empty_provider_transport_fields_are_normalized_but_populated_ones_are_rejected():
    response = parsed(cmp(i("rsi",14),s(30),"less_than"),cmp(i("rsi",14),s(70)))
    response.update({"partial_strategy":None,"questions":[],"ambiguous_terms":[]})
    assert isinstance(StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test")), ParsedStrategy)
    response["questions"] = [{"question_id":"unexpected","question":"Unexpected","target_path":"entry_condition","suggested_answers":[],"answer_kind":"choice"}]
    with pytest.raises(ModelResponseParsingFailure): StrategyInterpreterService(FakeModelClient([response])).interpret(StrategyInterpretationRequest(strategy_text="test"))
def test_malformed_and_client_failures_are_typed_and_monitored():
    monitor = InMemoryInterpreterMonitor(); request = StrategyInterpretationRequest(strategy_text="x")
    with pytest.raises(ModelResponseParsingFailure): StrategyInterpreterService(FakeModelClient([{"status":"parsed"}]), monitor=monitor).interpret(request)
    with pytest.raises(ModelClientFailure): StrategyInterpreterService(FakeModelClient([RuntimeError("network")]), monitor=monitor).interpret(request)
    assert monitor.summary()["failure_rate"] == 1.0
    assert all(event.input_hash != "x" for event in monitor.events)
    monitor.record_evaluation(InterpretationEvaluation(request_id=monitor.events[0].request_id, is_correct=True, category="safety"))
    assert monitor.summary()["evaluation_accuracy"] == 1.0
