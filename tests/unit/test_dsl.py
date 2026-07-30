import json

import pytest
from pydantic import TypeAdapter, ValidationError

from alpha_agent.domain.conditions import ConditionGroup, Cross, IndicatorValue, MarketField, Scalar
from alpha_agent.domain.indicators import RSI, SMA
from alpha_agent.domain.results import ParserResult
from alpha_agent.domain.strategy import StrategyDefinition
from alpha_agent.errors import StrategyValidationError
from alpha_agent.parser.validation import validate_strategy


def field(name: str) -> dict[str, object]:
    return {"kind": "market_field", "field": name}


def indicator(kind: str, window: int) -> dict[str, object]:
    data: dict[str, object] = {"type": kind, "window": window}
    if kind != "rolling_mean_volume":
        data["source"] = "close"
    return {"kind": "indicator", "indicator": data}


def comparison(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    return {"kind": "comparison", "operator": "greater_than", "left": left, "right": right}


def strategy(entry: dict[str, object] | None = None) -> dict[str, object]:
    condition = entry or comparison(field("close"), indicator("sma", 20))
    return {
        "schema_version": "0.1", "asset": "AAPL", "frequency": "daily",
        "direction": "long_only", "position_sizing": "fully_invested_or_flat",
        "entry": condition, "exit": comparison(indicator("sma", 20), field("close")),
    }


def test_invalid_window_and_unsupported_indicator_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SMA(type="sma", source="close", window=0)
    with pytest.raises(ValidationError):
        StrategyDefinition.model_validate(strategy(comparison(field("close"), indicator("macd", 12))))


def test_invalid_cross_operands_are_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        Cross.model_validate({"kind": "cross", "operator": "cross_above", "left": field("close"), "right": {"kind": "scalar", "value": 10.0}})


def test_invalid_rsi_threshold_is_rejected_semantically() -> None:
    model = StrategyDefinition.model_validate(strategy(comparison(indicator("rsi", 14), {"kind": "scalar", "value": 101.0})))
    with pytest.raises(StrategyValidationError, match="RSI thresholds"):
        validate_strategy(model)


def test_malformed_and_empty_condition_trees_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StrategyDefinition.model_validate(strategy({"kind": "group", "operator": "and", "conditions": []}))
    one = StrategyDefinition.model_validate(strategy({"kind": "group", "operator": "and", "conditions": [comparison(field("close"), indicator("sma", 20))]}))
    with pytest.raises(StrategyValidationError, match="at least two"):
        validate_strategy(one)


def test_nested_and_or_conditions_validate() -> None:
    leaf = comparison(field("close"), indicator("sma", 20))
    tree = {"kind": "group", "operator": "and", "conditions": [leaf, {"kind": "group", "operator": "or", "conditions": [leaf, leaf]}]}
    model = StrategyDefinition.model_validate(strategy(tree))
    validate_strategy(model)
    assert isinstance(model.entry, ConditionGroup)


def test_frequency_and_unknown_fields_are_rejected() -> None:
    payload = strategy()
    payload["frequency"] = "hourly"
    with pytest.raises(ValidationError):
        StrategyDefinition.model_validate(payload)
    payload = strategy()
    payload["arbitrary_python"] = "print('unsafe')"
    with pytest.raises(ValidationError):
        StrategyDefinition.model_validate(payload)


def test_json_round_trip_preserves_schema_version() -> None:
    result = {"status": "parsed", "strategy": strategy(), "assumptions": [], "warnings": []}
    adapter = TypeAdapter(ParserResult)
    parsed = adapter.validate_json(json.dumps(result))
    restored = adapter.validate_json(adapter.dump_json(parsed))
    assert restored.strategy.schema_version == "0.1"  # type: ignore[union-attr]
