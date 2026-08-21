"""Pure Matplotlib rendering for reproducible per-trade visual evidence."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from rhein.ui.macd import add_display_mmacd

from .models import TradeEvidence


@dataclass(frozen=True)
class RenderedTradeChart:
    trade_id: str
    png_bytes: bytes
    checksum: str


def _candle_axis(axis, frame: pd.DataFrame, title: str, event_dates: dict[str, str]) -> None:
    x = np.arange(len(frame))
    for position, row in enumerate(frame.itertuples(index=False)):
        open_, high, low, close = (float(row.Open), float(row.High), float(row.Low), float(row.Close))
        color = "#d9534f" if close < open_ else "#198754"
        axis.vlines(position, low, high, color=color, linewidth=0.8)
        axis.add_patch(Rectangle((position - .32, min(open_, close)), .64, max(abs(close - open_), 0.0001), color=color, alpha=.86))
    for window, color in ((5, "#2563eb"), (10, "#f59e0b"), (20, "#7c3aed")):
        axis.plot(x, frame["Close"].rolling(window).mean(), color=color, linewidth=.85, label=f"MA{window}")
    for label, date in event_dates.items():
        positions = np.flatnonzero(pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d") == date)
        if len(positions):
            position = int(positions[0])
            price = float(frame.iloc[position]["Close"])
            axis.scatter([position], [price], s=28, marker={"t0": "o", "entry": "^", "exit": "v"}[label], color={"t0": "#2563eb", "entry": "#0ea5e9", "exit": "#dc2626"}[label], zorder=5)
            axis.annotate(label, (position, price), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axis.set_title(title, fontsize=9, loc="left")
    axis.grid(alpha=.18, linewidth=.5)
    axis.legend(loc="upper left", ncol=3, fontsize=6, frameon=False)
    axis.set_xlim(-1, max(len(frame), 1))


def _window(frame: pd.DataFrame, focus_date: str, before: int = 18, after: int = 18) -> pd.DataFrame:
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    positions = np.flatnonzero(dates == pd.Timestamp(focus_date).normalize())
    if not len(positions):
        return frame.iloc[:0].copy()
    position = int(positions[0])
    return frame.iloc[max(0, position - before): min(len(frame), position + after + 1)].copy()


def render_trade_chart(trade: TradeEvidence, ohlc: pd.DataFrame) -> RenderedTradeChart:
    """Render source OHLC plus *display-only* MMACD/DD context.

    The renderer never reads or modifies strategy parameters.  Its fixed
    MMACD(10,24,8) panel is labelled as display evidence, not DSL evidence.
    """
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(ohlc.columns):
        raise ValueError("source OHLC is missing chart columns")
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    frame = add_display_mmacd(ohlc.loc[:, columns].sort_values("Date").reset_index(drop=True))
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    if any(not (dates == pd.Timestamp(value).normalize()).any() for value in (trade.signal_date, trade.entry_date, trade.exit_date)):
        raise ValueError("trade dates are absent from source OHLC")
    event_dates = {"t0": trade.signal_date, "entry": trade.entry_date, "exit": trade.exit_date}
    context = _window(frame, trade.entry_date, before=35, after=max(35, trade.days_held + 14))
    entry = _window(frame, trade.entry_date)
    exit_ = _window(frame, trade.exit_date)
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=(2.0, 1.45, .7, .9))
    context_axis = figure.add_subplot(grid[0, :]); entry_axis = figure.add_subplot(grid[1, 0]); exit_axis = figure.add_subplot(grid[1, 1])
    volume_axis = figure.add_subplot(grid[2, :]); indicator_axis = figure.add_subplot(grid[3, :])
    _candle_axis(context_axis, context, f"{trade.symbol} · 价格上下文 · {trade.trade_id}", event_dates)
    _candle_axis(entry_axis, entry, "入场放大", event_dates)
    _candle_axis(exit_axis, exit_, "出场放大", event_dates)
    x = np.arange(len(context))
    volume_axis.bar(x, context["Volume"].to_numpy(dtype=float), color="#64748b", width=.65)
    volume_axis.set_title("成交量", fontsize=9, loc="left"); volume_axis.grid(alpha=.18, linewidth=.5)
    indicator_axis.axhline(0, color="#64748b", linewidth=.7)
    indicator_axis.bar(x, context["MMACD_HIST"], color=np.where(context["MMACD_HIST"] >= 0, "#198754", "#d9534f"), width=.65)
    indicator_axis.plot(x, context["MMACD_DD"], color="#2563eb", linewidth=1, label="DD")
    indicator_axis.plot(x, context["MMACD_DED"], color="#f59e0b", linewidth=1, label="DED")
    indicator_axis.set_title("展示指标：MMACD/DD (10,24,8)，不等同于策略规则", fontsize=9, loc="left")
    indicator_axis.legend(loc="upper left", ncol=2, fontsize=7, frameon=False); indicator_axis.grid(alpha=.18, linewidth=.5)
    entry_summary = trade.entry_audit.get("summary", "审计不可用") if isinstance(trade.entry_audit, dict) else "审计不可用"
    exit_summary = trade.exit_audit.get("summary", "审计不可用") if isinstance(trade.exit_audit, dict) else "审计不可用"
    audit_summary = f"入场审计：{str(entry_summary)[:120]}\n出场审计：{str(exit_summary)[:120]}"
    figure.suptitle(
        f"收益 {trade.return_pct:+.3f}% · 持有 {trade.days_held} 日 · 出场 {trade.exit_reason}\n{audit_summary}",
        fontsize=9, x=.01, ha="left",
    )
    output = BytesIO()
    figure.savefig(output, format="png", dpi=110, metadata={"Software": "Rhein Research Skill"})
    plt.close(figure)
    contents = output.getvalue()
    return RenderedTradeChart(trade_id=trade.trade_id, png_bytes=contents, checksum=sha256(contents).hexdigest())
