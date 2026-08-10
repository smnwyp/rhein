from __future__ import annotations

from steady_supplement_runner import SteadyHorizonVerdict, _verdict, load_steady_supplement_config


def _horizon(
    horizon: int,
    *,
    quantile: bool = True,
    q5: bool = True,
    top_bottom: bool = True,
    yearly: bool = True,
    early_negative: bool = False,
    nonoverlap: bool = True,
) -> SteadyHorizonVerdict:
    return SteadyHorizonVerdict(
        horizon=horizon,
        quantiles_strictly_upward=quantile,
        q5_positive=q5,
        top_bottom_positive_and_significant=top_bottom,
        positive_spread_years=6 if yearly else 4,
        yearly_spread_stable=yearly,
        early_years_all_negative=early_negative,
        non_overlapping_consistent=nonoverlap,
    )


def test_steady_supplement_config_has_no_feature_or_weight_tuning_fields() -> None:
    config = load_steady_supplement_config()

    assert config.score_component == "steady"
    assert config.required_monotonic_horizons == (20, 60)
    assert config.required_all_horizons == (20, 40, 60)
    assert config.minimum_positive_spread_years == 6
    text = open("config/steady_supplement.toml", encoding="utf-8").read().lower()
    assert "weight" not in text
    assert "window" not in text
    assert "grid" not in text


def test_nonmonotonic_or_nonpositive_q5_rejects_steady_direction() -> None:
    config = load_steady_supplement_config()
    verdict = _verdict({20: _horizon(20, quantile=False), 40: _horizon(40), 60: _horizon(60, q5=False)}, supplement_config=config)

    assert not verdict.quantile_requirement_passed
    assert not verdict.accepted_for_oos_design
    assert verdict.outcome.startswith("否决")


def test_early_years_all_negative_is_regime_dependent_not_oos_candidate() -> None:
    config = load_steady_supplement_config()
    verdict = _verdict({20: _horizon(20, yearly=False, early_negative=True), 40: _horizon(40), 60: _horizon(60)}, supplement_config=config)

    assert not verdict.yearly_requirement_passed
    assert not verdict.accepted_for_oos_design
    assert verdict.outcome.startswith("regime 依赖")
