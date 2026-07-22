"""输出 Nasdaq 特征组统计、原始策略表现与受约束搜索域建议。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/nasdaq_10y")

VOLATILITY_DOMAINS = {
    "低波动": {"signal_lower_pct": "1.5, 2.0, 2.5", "band_width_pct": "0.5, 1.0", "stop_pct": "1.5, 2.0, 2.5"},
    "中波动": {"signal_lower_pct": "3.0, 4.0, 5.0", "band_width_pct": "1.0, 1.5", "stop_pct": "3.0, 4.0, 5.0"},
    "高波动": {"signal_lower_pct": "5.0, 6.0, 7.0", "band_width_pct": "1.5, 2.5", "stop_pct": "5.0, 7.0, 9.0"},
}
LIQUIDITY_COSTS = {
    "高流动性": "0 bps（当前基准；成本压力测试另行进行）",
    "中流动性": "0 bps（当前基准；成本压力测试另行进行）",
    "低流动性": "0 bps（当前基准；成本压力测试另行进行）",
}


def raw_strategy_by_group(groups: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))["results"]
    rows = []
    for item in raw:
        stat = item["stats"]
        if stat["mode"] == "复利":
            rows.append({"symbol": item["symbol"], **stat})
    stats = pd.DataFrame(rows)
    merged = groups[["symbol", "策略特征组"]].merge(stats, on="symbol", how="left")
    result = []
    for group, frame in merged.groupby("策略特征组", sort=False):
        active = frame[frame["n_trades"].fillna(0) > 0]
        trades = active["n_trades"].sum()
        gross_profit, gross_loss = active["gross_profit"].fillna(0).sum(), active["gross_loss"].fillna(0).sum()
        result.append({
            "策略特征组": group,
            "原始策略_有交易标的": len(active),
            "原始策略_总交易数": int(trades),
            "原始策略_加权胜率_%": (active["win_rate_pct"].fillna(0) * active["n_trades"]).sum() / trades if trades else np.nan,
            "原始策略_盈利因子": gross_profit / gross_loss if gross_loss else np.nan,
            "原始策略_独立账户总盈亏": active["total_pnl"].sum(),
            "原始策略_独立最大回撤中位数_%": active["max_drawdown_pct"].median(),
        })
    return pd.DataFrame(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="原始统一策略的 summary JSON")
    ap.add_argument("--output-dir", default="reports")
    args = ap.parse_args()
    groups = pd.read_csv(DATA_DIR / "nasdaq_feature_groups.csv")
    values = {group: [] for group in groups["策略特征组"].unique()}
    for index, (symbol, group) in enumerate(groups[["symbol", "策略特征组"]].itertuples(index=False, name=None), start=1):
        frame = pd.read_csv(DATA_DIR / f"{symbol}.csv", usecols=["Close"]).tail(252)
        returns = frame["Close"].pct_change().dropna().to_numpy()
        if len(returns):
            values[group].append(returns)
        if index % 500 == 0 or index == len(groups):
            print(f"已读取 {index}/{len(groups)}")

    rows = []
    for group, subset in groups.groupby("策略特征组", sort=False):
        returns = np.concatenate(values[group])
        vol_label = next((name for name in VOLATILITY_DOMAINS if name in group), "历史不足")
        liquidity_label = next((name for name in LIQUIDITY_COSTS if name in group), "不适用")
        domain = VOLATILITY_DOMAINS.get(vol_label, {})
        rows.append({
            "策略特征组": group,
            "标的数": len(subset),
            "历史天数中位数": subset["history_days"].median(),
            "年化波动率_Q25_%": subset["annualized_volatility_pct_252d"].quantile(.25),
            "年化波动率_中位数_%": subset["annualized_volatility_pct_252d"].median(),
            "年化波动率_Q75_%": subset["annualized_volatility_pct_252d"].quantile(.75),
            "日均成交额中位数": subset["median_daily_dollar_volume_252d"].median(),
            "日收益_P90_%": np.quantile(returns, .90) * 100,
            "日收益_P95_%": np.quantile(returns, .95) * 100,
            "建议信号下限候选_%": domain.get("signal_lower_pct", "不进入主搜索"),
            "建议信号带宽候选_%": domain.get("band_width_pct", "不进入主搜索"),
            "建议止损候选_%": domain.get("stop_pct", "不进入主搜索"),
            "建议成本压力测试": LIQUIDITY_COSTS.get(liquidity_label, "不进入主搜索"),
            "入场日候选": "t1, t2, t3" if domain else "不进入主搜索",
            "趋势SMA候选": "3, 5, 8, 10" if domain else "不进入主搜索",
        })
    stats = pd.DataFrame(rows)
    raw_stats = raw_strategy_by_group(groups, Path(args.summary))
    analysis = stats.merge(raw_stats, on="策略特征组", how="left")
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)
    analysis_path = output / "nasdaq_group_statistical_analysis.csv"
    domain_path = output / "nasdaq_group_search_domains.csv"
    analysis.to_csv(analysis_path, index=False)
    analysis[["策略特征组", "标的数", "日收益_P90_%", "日收益_P95_%", "建议信号下限候选_%",
              "建议信号带宽候选_%", "建议止损候选_%", "建议成本压力测试", "入场日候选", "趋势SMA候选"]].to_csv(domain_path, index=False)
    report = ["# Nasdaq 特征组统计与搜索域建议", "",
              "## 说明", "",
              "统计特征使用每只股票最近 252 个交易日；日收益 P90/P95 是组内全部近一年日收益的分位数。",
              "原始策略表现来自统一的 2%–2.5%、t2、t3/t4、2% 止损、SMA5 回测，且独立账户总盈亏不是组合收益。",
              "", "## 分组统计、基准表现与建议域", ""]
    report += [analysis.to_markdown(index=False, floatfmt=".2f"), "", "## 如何使用", "",
               "1. 先在各组使用其建议信号下限、带宽、止损的候选值，并固定入场日为 t1/t2/t3、SMA 为 3/5/8/10。",
               "2. 使用 `SEARCH_DOMAIN_AND_FILTERS.md` 的两阶段筛选，不在全市场混合搜索。",
               "3. 当前基准统一使用 0 bps；成本压力测试应作为独立实验，不能混入基准参数排名。",
               "4. 15% 最大回撤是硬条件，但需先明确组合构建规则；当前表中的回撤为独立标的回撤中位数，仅可作代理诊断。",
               "", "## 前视限制", "",
               "本报告的分组采用当前最近 252 日数据，适合当前横截面研究。历史验证必须在每个历史时点重建特征组，不能将当前分组直接投射回过去。"]
    (output / "nasdaq_group_statistical_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"分析 CSV：{analysis_path}")
    print(f"搜索域 CSV：{domain_path}")
    print(f"报告：{output / 'nasdaq_group_statistical_analysis.md'}")


if __name__ == "__main__":
    main()
