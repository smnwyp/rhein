"""按策略相关特征为当前 Nasdaq 股票池分组。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/nasdaq_10y")
REQUIRED = {"Date", "Close", "Volume"}


def tercile(series: pd.Series, labels: list[str]) -> pd.Series:
    """用排序处理重复值，稳定地生成低/中/高三档。"""
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.dropna()
    if len(valid) < 3:
        return result
    result.loc[valid.index] = pd.qcut(valid.rank(method="first"), 3, labels=labels).astype(str)
    return result


def market_cap_group(value: float) -> str:
    if pd.isna(value):
        return "市值未知"
    if value >= 200_000_000_000:
        return "超大盘（≥200B）"
    if value >= 10_000_000_000:
        return "大盘（10B–200B）"
    if value >= 2_000_000_000:
        return "中盘（2B–10B）"
    if value >= 300_000_000:
        return "小盘（0.3B–2B）"
    return "微盘（<0.3B）"


def main() -> None:
    universe_path = DATA_DIR / "universe_current_nasdaq_common_stocks.csv"
    universe = pd.read_csv(universe_path, usecols=["symbol", "name", "marketCap"])
    universe = universe[universe["symbol"].notna() & universe["symbol"].astype(str).str.strip().ne("")].copy()
    universe["market_cap_usd"] = pd.to_numeric(
        universe["marketCap"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    rows, failures = [], []
    for index, item in enumerate(universe.itertuples(index=False), start=1):
        path = DATA_DIR / f"{item.symbol}.csv"
        try:
            frame = pd.read_csv(path, usecols=list(REQUIRED)).dropna(subset=["Close", "Volume"])
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.sort_values("Date")
            recent = frame.tail(252)
            returns = recent["Close"].pct_change().dropna()
            annualized_vol = returns.std(ddof=1) * np.sqrt(252) * 100 if len(returns) >= 20 else np.nan
            median_dollar_volume = (recent["Close"] * recent["Volume"]).median() if len(recent) >= 20 else np.nan
            rows.append({
                "symbol": item.symbol, "name": item.name, "history_days": len(frame),
                "first_date": frame["Date"].iloc[0].date(), "last_date": frame["Date"].iloc[-1].date(),
                "annualized_volatility_pct_252d": annualized_vol,
                "median_daily_dollar_volume_252d": median_dollar_volume,
                "market_cap_usd": item.market_cap_usd,
            })
        except Exception as exc:
            failures.append({"symbol": item.symbol, "error": str(exc)})
        if index % 250 == 0 or index == len(universe):
            print(f"已处理 {index}/{len(universe)}；异常 {len(failures)}")

    groups = pd.DataFrame(rows)
    mature = groups[groups["history_days"] >= 252].copy()
    groups["波动档"] = tercile(mature["annualized_volatility_pct_252d"], ["低波动", "中波动", "高波动"])
    groups["流动性档"] = tercile(mature["median_daily_dollar_volume_252d"], ["低流动性", "中流动性", "高流动性"])
    groups["市值档"] = groups["market_cap_usd"].map(market_cap_group)
    groups["历史状态"] = np.where(groups["history_days"] >= 252, "历史≥1年", "历史不足1年")
    groups["策略特征组"] = np.where(
        groups["history_days"] < 252,
        "历史不足1年（不进入主验证）",
        groups["流动性档"].fillna("未知流动性") + "－" + groups["波动档"].fillna("未知波动"),
    )
    groups = groups.sort_values(["策略特征组", "symbol"]).reset_index(drop=True)
    summary = groups.groupby("策略特征组", dropna=False).agg(
        标的数=("symbol", "size"),
        年化波动率中位数=("annualized_volatility_pct_252d", "median"),
        日均成交额中位数=("median_daily_dollar_volume_252d", "median"),
        历史天数中位数=("history_days", "median"),
    ).reset_index()
    groups.to_csv(DATA_DIR / "nasdaq_feature_groups.csv", index=False)
    summary.to_csv(DATA_DIR / "nasdaq_feature_group_summary.csv", index=False)
    pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(DATA_DIR / "nasdaq_feature_group_failures.csv", index=False)
    (DATA_DIR / "nasdaq_feature_group_metadata.json").write_text(json.dumps({
        "generated_at": date.today().isoformat(),
        "feature_window": "最近 252 个交易日",
        "volatility_formula": "日收益率标准差 × sqrt(252) × 100%",
        "liquidity_formula": "Close × Volume 的 252 日中位数",
        "lookahead_warning": "当前横截面分组；历史回测必须按历史时点重建分组",
        "symbols": len(groups), "failures": len(failures),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("分组完成：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
