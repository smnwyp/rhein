"""对高流动性－高波动 Nasdaq 组执行受约束参数扫描。"""
from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from rhein.backtest import input_files, load_ohlc, run_backtest


GROUP_DIR = Path("data/nasdaq_10y/groups/03_高流动性_高波动")
BANDS = [(0.05, 0.065), (0.05, 0.075), (0.06, 0.075),
         (0.06, 0.085), (0.07, 0.085), (0.07, 0.095)]
ENTRY_LAGS = [1, 2, 3]
STOP_PCTS = [0.05, 0.07, 0.09]
SMAS = [3, 5, 8, 10]
COST_BPS = 0.0
CAPITAL = 10_000.0
COMPOUND = False


def main() -> None:
    files = input_files(GROUP_DIR)
    datasets = [(path.stem.upper(), load_ohlc(path)) for path in files]
    combinations = list(product(BANDS, ENTRY_LAGS, STOP_PCTS, SMAS))
    rows = []
    for index, ((band_lo, band_hi), entry_lag, stop_pct, sma_n) in enumerate(combinations, start=1):
        totals = {"trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0,
                  "return_sum": 0.0, "win_return_sum": 0.0, "loss_return_sum": 0.0,
                  "losses": 0, "days_held": 0.0}
        drawdowns = []
        profitable_symbols = 0
        all_returns = []
        symbol_returns = []
        for _, df in datasets:
            trades, stats = run_backtest(
                df, band_lo=band_lo, band_hi=band_hi, entry_lag=entry_lag,
                hard_stop_days=(entry_lag + 1, entry_lag + 2), stop_pct=stop_pct,
                sma_n=sma_n, capital=CAPITAL, compound=COMPOUND, stop_intraday=True,
                cost_bps=COST_BPS,
            )
            totals["trades"] += len(trades)
            if not trades.empty:
                returns = trades["ret_pct"]
                winners = returns > 0
                totals["wins"] += int(winners.sum())
                totals["losses"] += int((~winners).sum())
                totals["return_sum"] += returns.sum()
                totals["win_return_sum"] += returns[winners].sum()
                totals["loss_return_sum"] += returns[~winners].sum()
                totals["days_held"] += trades["days_held"].sum()
                all_returns.extend(returns.tolist())
            totals["gross_profit"] += stats.get("gross_profit", 0) or 0
            totals["gross_loss"] += stats.get("gross_loss", 0) or 0
            symbol_returns.append((stats["final_equity"] / CAPITAL - 1) * 100)
            if stats["n_trades"]:
                drawdowns.append(stats["max_drawdown_pct"])
                profitable_symbols += stats["total_pnl"] > 0
        dd_q25 = pd.Series(drawdowns).quantile(.25)
        avg_win = totals["win_return_sum"] / totals["wins"] if totals["wins"] else None
        avg_loss = totals["loss_return_sum"] / totals["losses"] if totals["losses"] else None
        rows.append({
            "signal_lo_pct": band_lo * 100, "signal_hi_pct": band_hi * 100,
            "entry_day": f"t{entry_lag}", "early_stop_days": f"t{entry_lag + 1},t{entry_lag + 2}",
            "stop_pct": stop_pct * 100, "sma_n": sma_n, "cost_bps": COST_BPS,
            "symbols": len(datasets), "trades": totals["trades"],
            "win_rate_pct": totals["wins"] / totals["trades"] * 100 if totals["trades"] else None,
            "avg_return_pct": totals["return_sum"] / totals["trades"] if totals["trades"] else None,
            "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
            "payoff_ratio": avg_win / abs(avg_loss) if avg_win is not None and avg_loss not in (None, 0) else None,
            "profit_factor": totals["gross_profit"] / totals["gross_loss"] if totals["gross_loss"] else None,
            "median_symbol_cumulative_return_pct": pd.Series(symbol_returns).median(),
            "mean_symbol_cumulative_return_pct": pd.Series(symbol_returns).mean(),
            "profitable_symbols": profitable_symbols,
            "avg_days_held": totals["days_held"] / totals["trades"] if totals["trades"] else None,
            "max_trade_win_pct": max(all_returns) if all_returns else None,
            "max_trade_loss_pct": min(all_returns) if all_returns else None,
            "q25_individual_max_drawdown_pct": dd_q25,
            "passes_15pct_drawdown_proxy": dd_q25 >= -15,
        })
        print(f"已完成 {index}/{len(combinations)}")
    results = pd.DataFrame(rows).sort_values(["profit_factor", "median_symbol_cumulative_return_pct"], ascending=False)
    output = Path("reports")
    output.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output / f"high_liquidity_high_volatility_scan_{stamp}.csv"
    report_path = output / f"high_liquidity_high_volatility_scan_{stamp}.md"
    results.to_csv(csv_path, index=False)
    report = ["# 高流动性－高波动组：受约束参数扫描", "",
              f"标的数：{len(datasets)}；组合数：{len(combinations)}；资金模式：{'复利' if COMPOUND else '固定仓位'}；单边成本：{COST_BPS} bps。",
              "", "## 搜索域", "",
              "- 信号带：5.0–6.5、5.0–7.5、6.0–7.5、6.0–8.5、7.0–8.5、7.0–9.5%。",
              "- 入场：t1、t2、t3；早期止损日自动对应 t(N+1)、t(N+2)。",
              "- 止损：5%、7%、9%；趋势出场 SMA：3、5、8、10。",
              "", "## 全部组合（按盈利因子排序）", "", results.to_markdown(index=False, floatfmt=".3f"),
              "", "## 口径", "",
              "- 累计收益率是每个标的独立账户的 `(最终权益 ÷ 初始资金 − 1) × 100%`；表中使用其中位数/平均数，不相加为虚假的组合金额。",
              "- 平均单笔收益、平均盈利、平均亏损与盈亏比按所有逐笔净收益率汇总计算；盈亏比 = 平均盈利 ÷ |平均亏损|。",
              "- `q25_individual_max_drawdown_pct` 是独立标的最大回撤的 25 分位数；其不低于 -15% 才通过临时的保守代理筛选。",
              "- 最终的 15% 回撤硬条件仍需在明确等权/持仓上限等组合规则后，基于组合权益曲线复核。"]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"CSV：{csv_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
