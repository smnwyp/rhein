import json
import pytest
from pydantic import TypeAdapter, ValidationError
from alpha_agent.domain.interpretation import StrategyInterpretationResult
from alpha_agent.domain.strategy import StrategyDefinition
from alpha_agent.errors import SemanticStrategyValidationFailure
from alpha_agent.parser.validation import validate_strategy

def field(name): return {"kind": "market_field", "field": name}
def scalar(value): return {"kind": "scalar", "value": value}
def indicator(kind, window, field_name="close"): return {"kind": "indicator", "indicator": {"indicator": kind, "field": "volume" if kind == "rolling_mean" else field_name, "window": window}}
def comparison(left, right, op="greater_than"): return {"node_type": "comparison", "operator": op, "left": left, "right": right}
def strategy(entry=None, exit=None):
    base = comparison(field("close"), indicator("sma", 20))
    return {"schema_version":"0.1", "symbol":"AAPL", "frequency":"1d", "direction":"long_only", "position_mode":"fully_invested_or_flat", "entry_condition":entry or base, "exit_condition":exit or base}

@pytest.mark.parametrize("value", [0, -1])
def test_invalid_indicator_window_is_rejected(value):
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), indicator("sma", value))))
def test_unsupported_indicator_and_cross_scalar_are_rejected():
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), indicator("macd", 12))))
    cross = {"node_type":"cross", "operator":"cross_above", "left":field("close"), "right":scalar(1)}
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(cross))
@pytest.mark.parametrize("threshold", [-1, 101])
def test_rsi_threshold_ranges_are_semantic_errors(threshold):
    model = StrategyDefinition.model_validate(strategy(comparison(indicator("rsi", 14), scalar(threshold))))
    with pytest.raises(SemanticStrategyValidationFailure, match="semantic"): validate_strategy(model)
def test_empty_and_singleton_groups_and_invalid_volume_indicator_are_rejected():
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy({"node_type":"group","operator":"and","conditions":[]}))
    one = StrategyDefinition.model_validate(strategy({"node_type":"group","operator":"and","conditions":[comparison(field("close"), indicator("sma",20))]}))
    with pytest.raises(SemanticStrategyValidationFailure): validate_strategy(one)
    bad = {"kind":"indicator","indicator":{"indicator":"rolling_mean","field":"close","window":20}}
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(strategy(comparison(field("close"), bad)))
def test_nested_and_or_and_comparison_operators_validate():
    leaf = comparison(field("close"), indicator("sma", 20), "greater_than_or_equal")
    tree = {"node_type":"group","operator":"and","conditions":[leaf,{"node_type":"group","operator":"or","conditions":[leaf,comparison(field("close"), indicator("ema", 10), "less_than_or_equal")]}]}
    validate_strategy(StrategyDefinition.model_validate(strategy(tree)))
def test_required_fixed_fields_and_schema_versions_are_rejected():
    for key, value in (("frequency","daily"),("direction","short"),("schema_version","0.2")):
        payload = strategy(); payload[key] = value
        with pytest.raises(ValidationError): StrategyDefinition.model_validate(payload)
    payload = strategy(); del payload["schema_version"]
    with pytest.raises(ValidationError): StrategyDefinition.model_validate(payload)
def test_json_round_trips_preserve_both_result_variants():
    adapter = TypeAdapter(StrategyInterpretationResult)
    parsed = {"status":"parsed","strategy":strategy(),"assumptions":[],"warnings":[]}
    clarification = {"status":"clarification_required","partial_strategy":None,"questions":[{"question_id":"window","question":"Which window?","target_path":"entry_condition","suggested_answers":["20"],"answer_kind":"window"}],"ambiguous_terms":["moving average"]}
    for payload in (parsed, clarification):
        restored = adapter.validate_json(adapter.dump_json(adapter.validate_json(json.dumps(payload))))
        assert restored.status == payload["status"]
    assert adapter.validate_python(parsed).strategy.schema_version == "0.1"
