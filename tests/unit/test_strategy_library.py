from alpha_agent.domain.strategy import StrategyDefinition
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
