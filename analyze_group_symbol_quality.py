"""按分组最佳策略分析高/低胜率与盈亏标的的可观测特征。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import input_files, load_ohlc, run_backtest

GROUP_ROOT = Path("data/nasdaq_10y/groups")
OUT = Path("reports")
MIN_TRADES = 3


def features(df: pd.DataFrame) -> dict:
    tail = df.tail(252).copy()
    close, volume = tail["Close"], tail["Volume"]
    annual_vol = close.pct_change().std() * np.sqrt(252) * 100
    return {
        "近252日年化波动率_%": annual_vol,
        "近252日日均成交额": (close * volume).median(),
        "近20日收益_%": (close.iloc[-1] / close.iloc[max(0, len(close) - 20)] - 1) * 100,
        "近60日收益_%": (close.iloc[-1] / close.iloc[max(0, len(close) - 60)] - 1) * 100,
        "收盘相对SMA50_%": (close.iloc[-1] / close.rolling(50).mean().iloc[-1] - 1) * 100,
        "历史交易日数": len(df),
    }


def summarize(group: str, frame: pd.DataFrame) -> list[dict]:
    eligible = frame[frame["交易数"] >= MIN_TRADES].copy()
    output = []
    for label, mask in {
        "高胜率（≥50%）": eligible["胜率 (%)"] >= 50,
        "低胜率（≤20%）": eligible["胜率 (%)"] <= 20,
        "盈利标的": eligible["累计收益率 (%)"] > 0,
        "亏损标的": eligible["累计收益率 (%)"] < 0,
    }.items():
        x = eligible[mask]
        if x.empty:
            continue
        output.append({
            "策略分组": group, "比较组": label, "标的数": len(x), "交易数": x["交易数"].sum(),
            "平均标的胜率 (%)": x["胜率 (%)"].mean(), "平均标的累计收益 (%)": x["累计收益率 (%)"].mean(),
            "平均最大回撤 (%)": x["最大回撤 (%)"].mean(),
            "平均年化波动率 (%)": x["近252日年化波动率_%"].mean(),
            "成交额中位数": x["近252日日均成交额"].median(),
            "平均近20日收益 (%)": x["近20日收益_%"].mean(),
            "平均近60日收益 (%)": x["近60日收益_%"].mean(),
            "平均相对SMA50 (%)": x["收盘相对SMA50_%"].mean(),
            "平均历史交易日数": x["历史交易日数"].mean(),
        })
    return output


def main() -> None:
    OUT.mkdir(exist_ok=True)
    symbol_rows, cohort_rows = [], []
    for folder in sorted(GROUP_ROOT.glob("[0-9][0-9]_*") ):
        if not folder.is_dir() or folder.name.startswith("10_"):
            continue
        meta = json.loads((folder / "search_metadata.json").read_text())
        params = dict(meta["best_by_profit_factor"]["parameters"])
        params["hard_stop_days"] = tuple(params["hard_stop_days"])
        rows = []
        for path in input_files(folder):
            df = load_ohlc(path)
            _, stats = run_backtest(df, capital=10_000, compound=False, **params)
            row = {"策略分组": meta["group"], "标的": path.stem.upper(), **features(df),
                   "交易数": stats["n_trades"], "胜率 (%)": stats.get("win_rate_pct"),
                   "盈利因子": stats.get("profit_factor"), "最大回撤 (%)": stats.get("max_drawdown_pct"),
                   "累计收益率 (%)": (stats["final_equity"] / 10_000 - 1) * 100}
            rows.append(row)
        frame = pd.DataFrame(rows)
        symbol_rows.extend(rows)
        cohort_rows.extend(summarize(meta["group"], frame))
    symbols = pd.DataFrame(symbol_rows)
    cohorts = pd.DataFrame(cohort_rows)
    symbols.to_csv(OUT / "group_symbol_quality_analysis.csv", index=False)
    cohorts.to_csv(OUT / "group_quality_cohort_summary.csv", index=False)
    (OUT / "group_quality_analysis.md").write_text(
        "# 分组标的质量分析\n\n"
        "仅统计至少 3 笔交易的标的。特征使用数据末端近 252 日，故为描述性相关，不可直接用于历史筛选或永久黑名单。"
        "候选过滤规则必须进一步做滚动样本外验证。\n\n" + cohorts.to_markdown(index=False, floatfmt=".2f") + "\n",
        encoding="utf-8",
    )
    print(f"逐标的：{OUT / 'group_symbol_quality_analysis.csv'}")
    print(f"分组汇总：{OUT / 'group_quality_cohort_summary.csv'}")


if __name__ == "__main__":
    main()
