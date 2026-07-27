"""Small, dependency-free KPI gauges used by the Streamlit overview."""
from __future__ import annotations

from math import isfinite


def _format(value: float | None, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None and isfinite(value) else "不适用"


def _clamp_percent(value: float, low: float, high: float) -> float:
    return max(0.0, min(100.0, (value - low) / (high - low) * 100))


def _tick_html(position: float, label: str) -> str:
    """Anchor a tick at its actual position rather than approximating with spaces."""
    if position <= 0:
        alignment = "left:0;transform:none;"
    elif position >= 100:
        alignment = "left:100%;transform:translateX(-100%);"
    else:
        alignment = f"left:{position:.2f}%;transform:translateX(-50%);"
    return f'<span style="position:absolute;top:0;white-space:nowrap;{alignment}">{label}</span>'


def kpi_gauge_html(kind: str, value: float | None, *, max_drawdown_limit: float) -> str:
    """Render a labelled horizontal gauge for one of the three core KPIs."""
    try:
        value = float(value) if value is not None and isfinite(float(value)) else None
    except (TypeError, ValueError):
        value = None
    if kind == "sharpe":
        label, display = "中位标的夏普比率", _format(value)
        position = _clamp_percent(value, -1.0, 2.5) if value is not None else 0.0
        gradient = "#dc3545 0 28.6%, #f0ad4e 28.6% 57.1%, #198754 57.1% 100%"
        if value is None:
            note = "需至少有一笔有效交易与非零日收益波动"
        elif value < 0:
            note = "负风险调整收益"
        elif value < 1:
            note = "偏弱：继续验证"
        elif value < 1.5:
            note = "可作为候选：夏普 ≥ 1"
        else:
            note = "较强：夏普 ≥ 1.5"
        ticks = ((0, "−1"), (28.57, "0"), (57.14, "1"), (100, "2.5+"))
    elif kind == "annualized_return":
        label, display = "中位标的年化收益", _format(value, "%")
        position = _clamp_percent(value, -10.0, 30.0) if value is not None else 0.0
        gradient = "#dc3545 0 25%, #f0ad4e 25% 50%, #198754 50% 100%"
        if value is None:
            note = "需重新运行当前参数"
        elif value <= 0:
            note = "未产生正年化收益"
        elif value < 10:
            note = "偏低：低于 10% 筛选线"
        elif value < 20:
            note = "候选：年化 ≥ 10%"
        else:
            note = "较强：年化 ≥ 20%"
        ticks = ((0, "−10%"), (25, "0%"), (50, "10%"), (100, "30%+"))
    elif kind == "drawdown":
        label, display = "中位标的最大回撤", _format(value, "%")
        drawdown = abs(value) if value is not None else 0.0
        upper = max(25.0, max_drawdown_limit * 2)
        position = _clamp_percent(drawdown, 0.0, upper) if value is not None else 0.0
        half_limit = max_drawdown_limit / 2 / upper * 100
        limit_position = max_drawdown_limit / upper * 100
        gradient = f"#198754 0 {half_limit:.1f}%, #f0ad4e {half_limit:.1f}% {limit_position:.1f}%, #dc3545 {limit_position:.1f}% 100%"
        if value is None:
            note = "需至少有一笔完成交易"
        elif drawdown <= max_drawdown_limit / 2:
            note = "较低回撤"
        elif drawdown <= max_drawdown_limit:
            note = f"在 {max_drawdown_limit:.1f}% 阈值内"
        else:
            note = f"超出 {max_drawdown_limit:.1f}% 阈值"
        ticks = ((0, "0%"), (half_limit, f"{max_drawdown_limit / 2:.1f}%"),
                 (limit_position, f"{max_drawdown_limit:.1f}%"), (100, f"{upper:.0f}%+"))
    else:
        raise ValueError(f"Unknown KPI gauge: {kind}")

    marker = "display: none;" if value is None else f"left: calc({position:.2f}% - 6px);"
    tick_labels = "".join(_tick_html(tick_position, tick_label) for tick_position, tick_label in ticks)
    return f"""
    <div style="border:1px solid #e6e9ef;border-radius:10px;padding:14px 14px 10px;background:#fff;min-height:126px;box-sizing:border-box;">
      <div style="font-size:0.9rem;color:#4a5568;font-weight:600;">{label}</div>
      <div style="font-size:1.75rem;font-weight:700;line-height:1.35;color:#1f2937;">{display}</div>
      <div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,{gradient});margin:11px 2px 7px;">
        <span style="position:absolute;top:-4px;width:18px;height:18px;border-radius:50%;background:#1f2937;border:3px solid white;box-shadow:0 1px 3px #667085;{marker}"></span>
      </div>
      <div style="position:relative;height:15px;font-size:0.72rem;color:#718096;margin:0 2px;">{tick_labels}</div>
      <div style="font-size:0.78rem;color:#4a5568;margin-top:7px;">{note}</div>
    </div>
    """
