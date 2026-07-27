"""Progress-aware execution of a parameter set across a collection of symbols."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def collect_results(paths: list[Path], params: dict, capital: float, compound: bool,
                    *, load_ohlc, run_backtest, progress_label: str = "回测中") -> tuple[pd.DataFrame, pd.DataFrame]:
    progress = st.progress(0, text=progress_label)
    status = st.empty()
    kpis, trades, errors = [], [], []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        try:
            ohlc = load_ohlc(path)
            trade_df, stats = run_backtest(ohlc, capital=capital, compound=compound, **params)
            calendar_days = max(int((ohlc["Date"].iloc[-1] - ohlc["Date"].iloc[0]).days), 1)
            equity_multiple = stats["final_equity"] / stats["initial_capital"]
            annualized_return_pct = (equity_multiple ** (365.25 / calendar_days) - 1) * 100
            kpis.append({
                "标的": path.stem.upper(), "源文件": str(path),
                "数据覆盖天数": calendar_days, "annualized_return_pct": annualized_return_pct,
                **stats,
            })
            if not trade_df.empty:
                item = trade_df.copy()
                item.insert(0, "symbol", path.stem.upper())
                trades.append(item)
        except Exception as exc:
            errors.append({"source": str(path), "error": str(exc)})
        if index == total or index % max(1, total // 100) == 0:
            progress.progress(index / total, text=f"{progress_label}：{index}/{total}")
            status.caption(f"已处理 {index}/{total} 个标的")
    progress.empty()
    status.empty()
    if errors:
        st.warning(f"{len(errors)} 个文件未能运行。", icon="⚠️")
        st.dataframe(pd.DataFrame(errors), hide_index=True)
    kpi_frame = pd.DataFrame(kpis)
    if not kpi_frame.empty:
        kpi_frame["cumulative_return_pct"] = (kpi_frame["final_equity"] / kpi_frame["initial_capital"] - 1) * 100
    return kpi_frame, (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())
