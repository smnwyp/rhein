from __future__ import annotations

from rhein.paths import GROUP_ROOT
from rhein.domain import conditions
from rhein.strategy import CONDITION_IDS, active_condition_ids, parse_ints, parse_percent_ranges
from rhein.ui.strategy_text import params_to_text, strategy_narrative


def test_strategy_parsers_and_condition_ids() -> None:
    assert parse_ints("3, 4") == (3, 4)
    assert parse_percent_ranges("2-2.5") == [(0.02, 0.025)]
    assert active_condition_ids({"use_signal_band": True, "use_forced_exit": False}) == [
        "T0-01", "T0-02", "T0-03", "T0-04", "T0-05", "T0-06", "T0-07", "T0-08", "EN-01",
        "EX-01", "EX-02", "EX-03", "EX-05",
    ]


def test_legacy_strategy_import_is_a_domain_compatibility_facade() -> None:
    """Staged migration must not break callers using ``rhein.strategy``."""
    assert CONDITION_IDS is conditions.CONDITION_IDS
    assert active_condition_ids is conditions.active_condition_ids
    assert parse_ints is conditions.parse_ints


def test_current_settings_text_keeps_entry_confirmation_wording() -> None:
    params = {
        "band_lo": 0.02, "band_hi": 0.025, "baseline_lookback": 15,
        "baseline_max_rise": 0.20, "baseline_rsi_period": 14,
        "baseline_rsi_max": 90, "entry_trend_fast_sma": 5,
        "entry_trend_slow_sma": 10, "entry_volume_fast_window": 5,
        "entry_volume_slow_window": 20, "hard_stop_days": (3, 4),
        "stop_pct": 0.02, "sma_n": 5, "cost_bps": 0.0,
    }
    assert "tN收盘≥t0" in params_to_text(params)


def test_current_settings_describes_the_actual_early_stop_execution_price() -> None:
    params = {
        "band_lo": .02, "band_hi": .025, "baseline_lookback": 15,
        "baseline_max_rise": .20, "baseline_rsi_period": 14,
        "baseline_rsi_max": 90, "entry_trend_fast_sma": 5,
        "entry_trend_slow_sma": 10, "entry_volume_fast_window": 5,
        "entry_volume_slow_window": 20, "entry_lag": 2, "hard_stop_days": (3, 4),
        "stop_pct": .02, "sma_n": 5, "cost_bps": 0, "stop_intraday": True,
    }
    assert "盘中任意即时价格" in strategy_narrative(params)
    params["stop_intraday"] = False
    assert "收盘价触及" in strategy_narrative(params)


def test_nine_mature_group_directories_exist() -> None:
    groups = [path for path in GROUP_ROOT.glob("[0-9][0-9]_*") if path.is_dir() and not path.name.startswith("10_")]
    assert len(groups) == 9


def test_streamlit_entrypoint_renders() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)

    assert not app.exception
    assert len(app.tabs) == 5
    scope = next(widget for widget in app.selectbox if widget.label == "数据范围")
    assert len([item for item in scope.options if "流动性" in str(item)]) == 9


def test_group_selection_loads_a_preset_and_runs_current_combo() -> None:
    """Regression path for the historical group-selection white-screen bug."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    scope = next(widget for widget in app.selectbox if widget.label == "数据范围")
    group = next(str(option) for option in scope.options if str(option).startswith("03 "))
    scope.select(group).run(timeout=45)
    assert not app.exception
    next(button for button in app.button if button.label == "运行当前参数").click().run(timeout=90)
    assert not app.exception
    assert any(widget.label == "参数预设版本" for widget in app.selectbox)
    assert len(app.dataframe) >= 1


def test_condition_toggle_reruns_without_breaking_controls() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    toggle = next(widget for widget in app.get("toggle") if "【T0-01】" in widget.label)
    toggle.set_value(False).run(timeout=45)
    assert not app.exception
    assert not next(widget for widget in app.get("toggle") if "【T0-01】" in widget.label).value
