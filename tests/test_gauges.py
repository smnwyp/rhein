from __future__ import annotations

from rhein.ui.gauges import kpi_gauge_html


def test_gauges_render_values_and_their_screening_labels() -> None:
    sharpe = kpi_gauge_html("sharpe", 1.2, max_drawdown_limit=15)
    annualized = kpi_gauge_html("annualized_return", 12, max_drawdown_limit=15)
    drawdown = kpi_gauge_html("drawdown", -12, max_drawdown_limit=15)

    assert "夏普 ≥ 1" in sharpe
    assert 'left:28.57%;transform:translateX(-50%);">0<' in sharpe
    assert "年化 ≥ 10%" in annualized
    assert 'left:50.00%;transform:translateX(-50%);">10%<' in annualized
    assert "在 15.0% 阈值内" in drawdown
    assert 'left:50.00%;transform:translateX(-50%);">15.0%<' in drawdown


def test_gauges_handle_missing_values_without_rendering_nan() -> None:
    rendered = kpi_gauge_html("sharpe", float("nan"), max_drawdown_limit=15)

    assert "不适用" in rendered
    assert "nan" not in rendered.lower()
