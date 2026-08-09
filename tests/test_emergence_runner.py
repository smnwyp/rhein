from __future__ import annotations

from emergence_runner import load_research_config


def test_research_config_is_preregistered_and_has_no_trading_parameters() -> None:
    config = load_research_config()

    assert config.study_config.horizons == (20, 40, 60)
    assert config.study_config.quantiles == 5
    assert config.study_config.minimum_sectors == 5
    assert config.minimum_same_sign_years == 5
    assert config.mapping_path.name == "industry_group_map.csv"
    assert "非 GICS" in config.sector_definition
    text = open("config/emergence_study.toml", encoding="utf-8").read().lower()
    assert not {"threshold", "transaction_cost", "drawdown", "annualized", "grid"} & set(text.split())
