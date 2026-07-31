from alpha_agent.domain.strategy import StrategyDefinition
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.strategy_library import JsonStrategyLibrary, SavedStrategy

def strategy() -> StrategyDefinition:
    return StrategyDefinition.model_validate({"schema_version":"0.1","symbol":"AAPL","frequency":"1d","direction":"long_only","position_mode":"fully_invested_or_flat","entry_condition":{"node_type":"comparison","operator":"greater_than","left":{"kind":"market_field","field":"close"},"right":{"kind":"indicator","indicator":{"indicator":"sma","field":"close","window":20}}},"exit_condition":{"node_type":"comparison","operator":"less_than","left":{"kind":"market_field","field":"close"},"right":{"kind":"indicator","indicator":{"indicator":"sma","field":"close","window":20}}}})

def test_library_persists_original_language_and_dsl_across_instances(tmp_path):
    path = tmp_path / "saved_strategies.json"
    saved = JsonStrategyLibrary(path).save(SavedStrategy(strategy_name="均线策略", original_language="收盘价上穿20日均线买入", strategy=strategy()))
    restored = JsonStrategyLibrary(path).list()
    assert len(restored) == 1
    assert restored[0] == saved
    assert restored[0].original_language == "收盘价上穿20日均线买入"
    assert restored[0].strategy.schema_version == "0.1"

def test_library_can_save_original_language_before_a_dsl_exists(tmp_path):
    stored = JsonStrategyLibrary(tmp_path / "saved_strategies.json").save(
        SavedStrategy(strategy_name="待解释策略", original_language="t0 涨幅后 t1 尾盘买入")
    )
    assert stored.strategy is None


def test_library_deletes_a_draft_without_affecting_validated_strategies(tmp_path):
    library = JsonStrategyLibrary(tmp_path / "saved_strategies.json")
    draft = library.save(SavedStrategy(strategy_name="草稿", original_language="t0 后买入"))
    validated = library.save(SavedStrategy(strategy_name="已验证", original_language="均线策略", strategy=strategy()))
    library.delete(draft.strategy_id)
    assert library.list() == [validated]


def test_library_persists_a_v03_stateful_strategy_and_its_audit_records(tmp_path):
    v03 = TimedStrategyDefinition.model_validate({
        "schema_version": "0.3", "symbol": "AAPL", "frequency": "1d", "direction": "long_only", "position_mode": "fully_invested_or_flat", "data_requirement": "daily_ohlcv",
        "anchor": {"name": "t0", "condition": {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}}}},
        "entry": {"active_day": {"start_offset_days": 0, "end_offset_days": 0}, "condition": {"node_type": "comparison", "operator": "greater_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}}}, "execution": "close"},
        "exit_rules": [{"rule_id": "exit", "priority": 1, "active_days": {"start_offset_days": 1}, "kind": "close_condition", "condition": {"node_type": "comparison", "operator": "less_than", "left": {"kind": "market_field", "field": "close"}, "right": {"kind": "indicator", "indicator": {"indicator": "sma", "field": "close", "window": 20}}}, "execution": "close"}],
        "lifecycle_policy": {"sample_end_open_position": "force_close", "allow_reentry_after_exit": False},
    })
    library = JsonStrategyLibrary(tmp_path / "saved_strategies.json")
    library.save(SavedStrategy(strategy_name="C 点策略", original_language="C 点入场", strategy=v03))
    restored = library.list()
    assert restored[0].strategy == v03
    assert restored[0].strategy.schema_version == "0.3"
