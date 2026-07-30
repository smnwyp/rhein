import pytest

from alpha_agent.domain.results import ClarificationRequired, ParsedStrategy
from alpha_agent.errors import ModelResponseError
from alpha_agent.model.fake_client import FakeModelClient
from alpha_agent.parser.service import StrategyInterpreter


def market(name: str) -> dict[str, object]: return {"kind": "market_field", "field": name}
def scalar(value: float) -> dict[str, object]: return {"kind": "scalar", "value": value}
def ind(kind: str, window: int) -> dict[str, object]:
    definition: dict[str, object] = {"type": kind, "window": window}
    if kind != "rolling_mean_volume": definition["source"] = "close"
    return {"kind": "indicator", "indicator": definition}
def compare(op: str, left: dict[str, object], right: dict[str, object]) -> dict[str, object]: return {"kind": "comparison", "operator": op, "left": left, "right": right}
def cross(op: str, left: dict[str, object], right: dict[str, object]) -> dict[str, object]: return {"kind": "cross", "operator": op, "left": left, "right": right}
def group(*conditions: dict[str, object]) -> dict[str, object]: return {"kind": "group", "operator": "and", "conditions": list(conditions)}
def parsed(entry: dict[str, object], exit: dict[str, object], assumptions: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {"status": "parsed", "strategy": {"schema_version": "0.1", "asset": "AAPL", "frequency": "daily", "direction": "long_only", "position_sizing": "fully_invested_or_flat", "entry": entry, "exit": exit}, "assumptions": assumptions or [], "warnings": []}


@pytest.mark.parametrize(("text", "response"), [
    ("Buy when close crosses above its 20-day moving average; sell below.", parsed(cross("cross_above", market("close"), ind("sma", 20)), cross("cross_below", market("close"), ind("sma", 20)), [{"code": "ma_is_sma", "message": "Moving average interpreted as SMA."}])),
    ("Buy when SMA10 crosses above SMA50 and volume exceeds its average.", parsed(group(cross("cross_above", ind("sma", 10), ind("sma", 50)), compare("greater_than", market("volume"), ind("rolling_mean_volume", 20))), cross("cross_below", ind("sma", 10), ind("sma", 50)))),
    ("Buy below RSI 30, sell above RSI 70.", parsed(compare("less_than", ind("rsi", 14), scalar(30)), compare("greater_than", ind("rsi", 14), scalar(70)), [{"code": "default_rsi_window", "message": "Used RSI window 14."}])),
    ("Buy when 20-day return exceeds 5% and volume exceeds 1.5 times average.", parsed(group(compare("greater_than", ind("rolling_return", 20), scalar(0.05)), compare("greater_than", market("volume"), {"kind": "scaled_series", "series": ind("rolling_mean_volume", 20), "factor": 1.5})), compare("less_than", ind("rolling_return", 20), scalar(0.0)))),
])
def test_four_unambiguous_examples(text: str, response: dict[str, object]) -> None:
    client = FakeModelClient([response])
    result = StrategyInterpreter(client).interpret(text, asset="AAPL")
    assert isinstance(result, ParsedStrategy)
    assert result.strategy.asset == "AAPL"
    assert text in client.requests[0][1]


def test_ambiguous_example_returns_questions_without_inventing_rules() -> None:
    response = {"status": "clarification_required", "partially_parsed_strategy": None, "ambiguous_terms": ["fallen significantly", "volume is increasing"], "questions": [
        {"id": "fall", "ambiguous_term": "fallen significantly", "question": "Over what window and by what percentage?", "expected_answer_type": "free_text", "choices": None},
        {"id": "volume", "ambiguous_term": "volume is increasing", "question": "Which definition should be used?", "expected_answer_type": "choice", "choices": ["Above rolling average", "Above prior day"]},
    ]}
    result = StrategyInterpreter(FakeModelClient([response])).interpret("Buy when the stock has fallen significantly and volume is increasing.", asset="AAPL")
    assert isinstance(result, ClarificationRequired)
    assert result.partially_parsed_strategy is None
    assert result.ambiguous_terms == ["fallen significantly", "volume is increasing"]


def test_malformed_model_output_has_explicit_error() -> None:
    with pytest.raises(ModelResponseError):
        StrategyInterpreter(FakeModelClient([{"status": "parsed"}])).interpret("anything")
