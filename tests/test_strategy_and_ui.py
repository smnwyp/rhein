from __future__ import annotations

from pathlib import Path

import pandas as pd

from rhein.paths import GROUP_ROOT
from rhein.domain import conditions
from rhein.strategy import CONDITION_IDS, active_condition_ids, parse_ints, parse_percent_ranges
from rhein.ui.strategy_text import params_to_text, strategy_narrative
from rhein.ui.trade_chart import event_text_layer, selected_event_id, trade_selector_options
from rhein.engine.eligibility import baseline_is_eligible


def test_strategy_parsers_and_condition_ids() -> None:
    assert parse_ints("3, 4") == (3, 4)
    assert parse_percent_ranges("2-2.5") == [(0.02, 0.025)]
    assert active_condition_ids({"use_signal_band": True, "use_forced_exit": False}) == [
        "T0-01", "T0-02", "T0-03", "T0-04", "T0-05", "T0-06", "T0-07", "T0-08", "T0-09", "T0-10", "EN-01",
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


def test_t0_bullish_candle_condition_rejects_bearish_and_doji_candles() -> None:
    params = {key: False for key in conditions.ATOMIC_TOGGLE_KEYS}
    params.update({"use_baseline_bullish_candle": True, "baseline_lookback": 1,
                   "band_lo": .02, "band_hi": .025, "baseline_rsi_max": 90,
                   "baseline_max_rise": .20})
    common = dict(ret1=[0, 0], rsi=[0, 0], entry_fast_sma=[0, 0], entry_slow_sma=[0, 0],
                  baseline_sma20=[0, 0], vol_fast_sma=[0, 0], vol_slow_sma=[0, 0], params=params)
    assert baseline_is_eligible(1, close=[1, 2], open_=[1, 1], **common)
    assert not baseline_is_eligible(1, close=[1, 1], open_=[1, 1], **common)
    assert not baseline_is_eligible(1, close=[1, 1], open_=[1, 2], **common)


def test_t0_ma20_must_be_at_least_point_zero_one_percent_above_two_days_earlier() -> None:
    params = {key: False for key in conditions.ATOMIC_TOGGLE_KEYS}
    params.update({"use_baseline_sma20_rising": True, "baseline_lookback": 1,
                   "band_lo": .02, "band_hi": .025, "baseline_rsi_max": 90,
                   "baseline_max_rise": .20})
    common = dict(ret1=[0, 0, 0], rsi=[0, 0, 0], entry_fast_sma=[0, 0, 0],
                  entry_slow_sma=[0, 0, 0], vol_fast_sma=[0, 0, 0],
                  vol_slow_sma=[0, 0, 0], params=params)
    assert baseline_is_eligible(2, close=[1, 1, 1], open_=[1, 1, 1],
                                baseline_sma20=[10, 10, 10.001], **common)
    assert not baseline_is_eligible(2, close=[1, 1, 1], open_=[1, 1, 1],
                                    baseline_sma20=[10, 10, 10.0009], **common)


def test_trade_chart_selector_uses_trade_identity_not_just_row_number() -> None:
    trades = pd.DataFrame([
        {"signal": "2022-12-02", "entry": "2022-12-02", "exit": "2022-12-12", "reason": "到期", "ret_pct": 1.16},
        {"signal": "2022-12-15", "entry": "2022-12-15", "exit": "2022-12-21", "reason": "止损", "ret_pct": -2.00},
    ])
    option_ids, labels = trade_selector_options(trades)
    assert option_ids[0] != option_ids[1]
    assert "t0 2022-12-02" in labels[option_ids[0]]
    assert "出场 2022-12-21" in labels[option_ids[1]]


def test_trade_chart_event_selection_accepts_vega_list_and_mapping_payloads() -> None:
    assert selected_event_id({"selection": {"trade_event": [{"EventId": "entry"}]}}) == "entry"
    assert selected_event_id({"selection": {"trade_event": {"EventId": ["exit"]}}}) == "exit"
    assert selected_event_id({"selection": {"trade_event": [{"EventId": "anchor"}]}}) is None


def test_trade_chart_finds_annotation_layer_after_clickable_marker_is_added() -> None:
    text_layer = {"mark": {"type": "text"}, "encoding": {"text": {"field": "Label"}}}
    layers = [{"mark": {"type": "point"}}, text_layer, {"mark": {"type": "point"}}]

    assert event_text_layer(layers) is text_layer


def test_current_settings_updates_when_t0_09_is_toggled() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    assert "阳线（收盘>开盘）" in app.info[0].value
    toggle = next(widget for widget in app.get("toggle") if "【T0-09】" in widget.label)
    toggle.set_value(False).run(timeout=45)
    assert not app.exception
    assert "阳线（收盘>开盘）" not in app.info[0].value


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
    entry_day = next(widget for widget in app.selectbox if widget.label == "入场日（相对 t0）")
    assert entry_day.options[0] == "t0"
    entry_day.select(0).run(timeout=45)
    assert not app.exception


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
    assert "夏普比率（高→低）" in next(
        widget.options for widget in app.selectbox if widget.label == "排序指标"
    )
    assert len(app.dataframe) >= 1


def test_condition_toggle_reruns_without_breaking_controls() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    toggle = next(widget for widget in app.get("toggle") if "【T0-01】" in widget.label)
    toggle.set_value(False).run(timeout=45)
    assert not app.exception
    assert not next(widget for widget in app.get("toggle") if "【T0-01】" in widget.label).value


def test_old_cached_kpis_remain_displayable_after_risk_metrics_are_added() -> None:
    """A session created before Sharpe/annualized fields must not white-screen."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    app.session_state["single_kpis"] = pd.DataFrame([{
        "标的": "TEST", "n_trades": 1, "gross_profit": 1.0, "gross_loss": 1.0,
        "cumulative_return_pct": 1.0, "max_drawdown_pct": -1.0, "win_rate_pct": 50.0,
        "profit_factor": 1.0, "payoff_ratio": 1.0, "avg_return_pct": 1.0,
        "avg_days_held": 1.0,
    }])
    app.session_state["single_mode"] = "固定仓位"
    app.run(timeout=45)

    assert not app.exception
    assert any("更新前的缓存" in info.value for info in app.info)


def test_interpreter_hides_stale_saved_run_when_the_strategy_dsl_changed() -> None:
    """A historical result must not masquerade as evidence for a new DSL."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("pages/1_Strategy_Interpreter.py")
    app.run(timeout=45)
    scope = next(widget for widget in app.selectbox if widget.label == "标的分组")
    scope.select(next(option for option in scope.options if str(option).startswith("02 "))).run(timeout=45)
    app.radio[0].set_value("已保存策略").run(timeout=45)
    saved_strategy = next(widget for widget in app.selectbox if widget.label == "选择已保存策略")
    saved_strategy.select("周四策略 · 已验证").run(timeout=45)
    assert not app.exception
    assert not any(widget.label == "载入这次已保存回测" for widget in app.button)
    assert any("基于旧 DSL 版本" in widget.value for widget in app.info)


def test_trade_chart_keeps_price_macd_and_dmi_legends_independent() -> None:
    """Regression: Vega-Lite must not merge unrelated colour legends."""
    page = Path("pages/1_Strategy_Interpreter.py").read_text()

    assert '"resolve": {"scale": {"color": "independent"}}' in page


def test_trade_chart_marker_click_opens_an_in_chart_condition_overlay() -> None:
    """The chart must expose entry/exit clicks and draw the card after exit."""
    page = Path("pages/1_Strategy_Interpreter.py").read_text()

    assert '"name": "trade_event"' in page
    assert '"fields": ["EventId"]' in page
    assert 'on_select="rerun"' in page
    assert 'selection_mode="trade_event"' in page
    assert '"fillOpacity": 0.94' in page
    assert 'exit_index + 3' in page
    assert 'decisive_entry_audit_rows(event_rows)' in page
    assert '"ClickLabel": "入场 ⓘ"' in page
    assert '点击图中带蓝色“入场 ⓘ”或红色“出场 ⓘ”' in page


def test_compact_audit_overlay_lines_keep_evidence_and_limit_card_height() -> None:
    from rhein.ui.trade_chart import compact_audit_overlay_lines

    rows = [
        {"左侧": "SMA(60)", "左侧数值": 12.1, "比较": ">", "右侧": "SMA(60)[-1]", "右侧数值": 12.0, "结果": "通过"},
        {"左侧": "DIF", "左侧数值": 2.0, "比较": "≥", "右侧": "2", "右侧数值": 2, "结果": "通过"},
        {"左侧": "MA20", "左侧数值": 10, "比较": "上穿", "右侧": "MA60", "右侧数值": 9, "结果": "未通过"},
    ]

    assert compact_audit_overlay_lines(rows, maximum=2) == [
        "✓ SMA(60) > SMA(60)[-1]  [12.1 / 12.0]",
        "✓ DIF ≥ 2  [2.0 / 2]",
    ]


def test_entry_overlay_shows_only_the_or_branch_that_admitted_the_trade() -> None:
    from rhein.ui.trade_chart import decisive_entry_audit_rows

    rows = [
        {"DSL 路径": "anchor.condition.conditions[0].conditions[0]", "比较": ">", "结果": "未通过"},
        {"DSL 路径": "anchor.condition.conditions[0]", "比较": "group_result", "结果": "未通过"},
        {"DSL 路径": "anchor.condition.conditions[2].conditions[0]", "比较": ">", "结果": "通过"},
        {"DSL 路径": "anchor.condition.conditions[2]", "比较": "group_result", "结果": "通过"},
        {"DSL 路径": "anchor.condition", "比较": "group_result", "结果": "通过"},
    ]

    label, selected = decisive_entry_audit_rows(rows)

    assert label == "命中买入分支：第 3 组"
    assert [row["DSL 路径"] for row in selected] == [
        "anchor.condition.conditions[2].conditions[0]",
        "anchor.condition.conditions[2]",
    ]
