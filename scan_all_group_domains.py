"""按特征组的受约束域执行两阶段参数扫描，并写入各组 metadata。"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from backtest import input_files, load_ohlc, run_backtest


GROUP_ROOT = Path("data/nasdaq_10y/groups")
CAPITAL = 10_000.0
COMPOUND = False  # 项目默认：固定仓位
DOMAINS = {
    "低波动": {"lowers": [1.5, 2.0, 2.5], "widths": [0.5, 1.0], "stops": [1.5, 2.0, 2.5]},
    "中波动": {"lowers": [3.0, 4.0, 5.0], "widths": [1.0, 1.5], "stops": [3.0, 4.0, 5.0]},
    "高波动": {"lowers": [5.0, 6.0, 7.0], "widths": [1.5, 2.5], "stops": [5.0, 7.0, 9.0]},
}
# 当前基准统一按零费用计算；成本压力测试应作为独立实验运行。
COSTS = {"高流动性": 0.0, "中流动性": 0.0, "低流动性": 0.0}
ENTRY_LAGS, SMAS = [1, 2, 3], [3, 5, 8, 10]
# 本轮统一使用用户定义的入场趋势过滤默认线；UI 可另行调整并单组回测。
ENTRY_TREND_FAST_SMA, ENTRY_TREND_SLOW_SMA = 5, 10
ENTRY_VOLUME_FAST_WINDOW, ENTRY_VOLUME_SLOW_WINDOW = 5, 20
BASELINE_LOOKBACK, BASELINE_MAX_RISE, BASELINE_RSI_PERIOD, BASELINE_RSI_MAX = 15, .20, 14, 90
STRATEGY_VERSION = "v2_core_breakout_only"
STRATEGY_LABEL = "v2 核心突破：T0-01、EN-01、EX-01～EX-03"
ACTIVE_CONDITION_IDS = ("T0-01", "EN-01", "EX-01", "EX-02", "EX-03")
ATOMIC_FLAGS = {
    "use_signal_band": True,
    "use_baseline_prior_low": False,
    "use_baseline_max_rise": False,
    "use_baseline_rsi": False,
    "use_entry_close_vs_t0": True,
    "use_entry_close_above_fast_sma": False,
    "use_entry_fast_above_slow_sma": False,
    "use_entry_volume_sma": False,
    "use_early_stop": True,
    "use_exit_below_entry": True,
    "use_exit_below_sma": True,
}


def group_labels(folder: Path) -> tuple[str, str, str]:
    manifest = pd.read_csv(folder / "group_manifest.csv", usecols=["策略特征组"])
    group = manifest["策略特征组"].iloc[0]
    vol = next(label for label in DOMAINS if label in group)
    liquidity = next(label for label in COSTS if label in group)
    return group, vol, liquidity


def evaluate(datasets: list[tuple[str, pd.DataFrame]], params: dict) -> dict:
    totals = {"trades": 0, "wins": 0, "losses": 0, "return_sum": 0.0, "win_return_sum": 0.0,
              "loss_return_sum": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "days": 0.0}
    returns_by_symbol, drawdowns, payoff_ratios, win_rates_by_symbol = [], [], [], []
    profitable_symbols = 0
    for _, df in datasets:
        trades, stats = run_backtest(df, capital=CAPITAL, compound=COMPOUND, **params)
        # 与 UI 的“中位标的累计收益”一致：只统计实际产生交易的标的，
        # 不让零交易标的的 0% 收益扭曲收益分布。
        if stats["n_trades"] > 0:
            returns_by_symbol.append((stats["final_equity"] / CAPITAL - 1) * 100)
            win_rates_by_symbol.append(stats["win_rate_pct"])
        payoff_ratio = stats.get("payoff_ratio")
        if payoff_ratio is not None and pd.notna(payoff_ratio) and pd.api.types.is_number(payoff_ratio):
            payoff_ratios.append(float(payoff_ratio))
        if stats["total_pnl"] > 0:
            profitable_symbols += 1
        if not trades.empty:
            returns = trades["ret_pct"]
            winners = returns > 0
            totals["trades"] += len(trades)
            totals["wins"] += int(winners.sum())
            totals["losses"] += int((~winners).sum())
            totals["return_sum"] += returns.sum()
            totals["win_return_sum"] += returns[winners].sum()
            totals["loss_return_sum"] += returns[~winners].sum()
            totals["days"] += trades["days_held"].sum()
            totals["gross_profit"] += stats.get("gross_profit", 0) or 0
            totals["gross_loss"] += stats.get("gross_loss", 0) or 0
            drawdowns.append(stats["max_drawdown_pct"])
    avg_win = totals["win_return_sum"] / totals["wins"] if totals["wins"] else None
    avg_loss = totals["loss_return_sum"] / totals["losses"] if totals["losses"] else None
    dd = pd.Series(drawdowns, dtype=float)
    symbol_payoff = pd.Series(payoff_ratios, dtype=float)
    symbol_win_rates = pd.Series(win_rates_by_symbol, dtype=float)
    return {
        "trades": totals["trades"],
        "win_rate_pct": totals["wins"] / totals["trades"] * 100 if totals["trades"] else None,
        "mean_symbol_win_rate_pct": symbol_win_rates.mean() if len(symbol_win_rates) else None,
        "median_symbol_win_rate_pct": symbol_win_rates.median() if len(symbol_win_rates) else None,
        "avg_return_pct": totals["return_sum"] / totals["trades"] if totals["trades"] else None,
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "payoff_ratio": avg_win / abs(avg_loss) if avg_win is not None and avg_loss not in (None, 0) else None,
        "mean_symbol_payoff_ratio": symbol_payoff.mean() if len(symbol_payoff) else None,
        "median_symbol_payoff_ratio": symbol_payoff.median() if len(symbol_payoff) else None,
        "max_symbol_payoff_ratio": symbol_payoff.max() if len(symbol_payoff) else None,
        "symbols_with_payoff_ratio": len(symbol_payoff),
        "profit_factor": totals["gross_profit"] / totals["gross_loss"] if totals["gross_loss"] else None,
        "median_symbol_cumulative_return_pct": pd.Series(returns_by_symbol).median(),
        "mean_symbol_cumulative_return_pct": pd.Series(returns_by_symbol).mean(),
        "profitable_symbols": profitable_symbols,
        "avg_days_held": totals["days"] / totals["trades"] if totals["trades"] else None,
        "q25_individual_max_drawdown_pct": dd.quantile(.25),
        "passes_15pct_drawdown_proxy": bool(dd.quantile(.25) >= -15) if len(dd) else False,
    }


def with_params(params: dict, metrics: dict) -> dict:
    return {
        "strategy_version": STRATEGY_VERSION,
        "active_condition_ids": ", ".join(ACTIVE_CONDITION_IDS),
        "signal_lo_pct": params["band_lo"] * 100, "signal_hi_pct": params["band_hi"] * 100,
        "entry_day": f"t{params['entry_lag']}",
        "entry_trend_filter": (f"tN-1/tN 任一天收盘>SMA{params['entry_trend_fast_sma']}"
                               f">SMA{params['entry_trend_slow_sma']}且5日均量>20日均量")
        if params.get("entry_trend_filter", True) else "无",
        "entry_trend_fast_sma": params["entry_trend_fast_sma"],
        "entry_trend_slow_sma": params["entry_trend_slow_sma"],
        "entry_volume_fast_window": params["entry_volume_fast_window"],
        "entry_volume_slow_window": params["entry_volume_slow_window"],
        "baseline_filter": f"t0-{params['baseline_lookback']}~t0；涨幅≤{params['baseline_max_rise']:.0%}；RSI{params['baseline_rsi_period']}≤{params['baseline_rsi_max']}",
        "early_stop_days": ",".join(f"t{day}" for day in params["hard_stop_days"]),
        "stop_pct": params["stop_pct"] * 100, "sma_n": params["sma_n"],
        "cost_bps": params["cost_bps"], **metrics,
    }


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["profit_factor", "median_symbol_cumulative_return_pct"], ascending=False)


def scan_group(folder_text: str) -> dict:
    folder = Path(folder_text)
    group, vol_label, liquidity_label = group_labels(folder)
    domain, cost_bps = DOMAINS[vol_label], COSTS[liquidity_label]
    datasets = [(path.stem.upper(), load_ohlc(path)) for path in input_files(folder)]
    bands = [(lower, lower + width) for lower, width in product(domain["lowers"], domain["widths"])]
    baseline_stop = domain["stops"][1]
    stage1_rows = []
    for band, entry in product(bands, ENTRY_LAGS):
        params = {"band_lo": band[0] / 100, "band_hi": band[1] / 100, "entry_lag": entry,
                  "hard_stop_days": (entry + 1, entry + 2), "stop_pct": baseline_stop / 100,
                  "sma_n": 5, "entry_trend_filter": True, **ATOMIC_FLAGS,
                  "entry_trend_fast_sma": ENTRY_TREND_FAST_SMA,
                  "entry_trend_slow_sma": ENTRY_TREND_SLOW_SMA,
                  "entry_volume_fast_window": ENTRY_VOLUME_FAST_WINDOW,
                  "entry_volume_slow_window": ENTRY_VOLUME_SLOW_WINDOW,
                  "baseline_lookback": BASELINE_LOOKBACK, "baseline_max_rise": BASELINE_MAX_RISE,
                  "baseline_rsi_period": BASELINE_RSI_PERIOD, "baseline_rsi_max": BASELINE_RSI_MAX,
                  "stop_intraday": True, "cost_bps": cost_bps}
        stage1_rows.append(with_params(params, evaluate(datasets, params)))
    stage1 = sort_results(pd.DataFrame(stage1_rows))
    candidates = stage1.head(3)
    stage2_rows = []
    for candidate in candidates.itertuples(index=False):
        entry = int(str(candidate.entry_day).replace("t", ""))
        for stop, sma in product(domain["stops"], SMAS):
            params = {"band_lo": candidate.signal_lo_pct / 100, "band_hi": candidate.signal_hi_pct / 100,
                      "entry_lag": entry, "hard_stop_days": (entry + 1, entry + 2),
                      "stop_pct": stop / 100, "sma_n": sma, "entry_trend_filter": True, **ATOMIC_FLAGS,
                      "entry_trend_fast_sma": ENTRY_TREND_FAST_SMA,
                      "entry_trend_slow_sma": ENTRY_TREND_SLOW_SMA,
                      "entry_volume_fast_window": ENTRY_VOLUME_FAST_WINDOW,
                      "entry_volume_slow_window": ENTRY_VOLUME_SLOW_WINDOW,
                      "baseline_lookback": BASELINE_LOOKBACK, "baseline_max_rise": BASELINE_MAX_RISE,
                      "baseline_rsi_period": BASELINE_RSI_PERIOD, "baseline_rsi_max": BASELINE_RSI_MAX,
                      "stop_intraday": True, "cost_bps": cost_bps}
            stage2_rows.append(with_params(params, evaluate(datasets, params)))
    stage2 = sort_results(pd.DataFrame(stage2_rows))
    best = stage2.iloc[0].to_dict()
    best_params = {
        "band_lo": best["signal_lo_pct"] / 100, "band_hi": best["signal_hi_pct"] / 100,
        "entry_lag": int(str(best["entry_day"]).replace("t", "")),
        "hard_stop_days": [int(day.replace("t", "")) for day in str(best["early_stop_days"]).split(",")],
        "stop_pct": best["stop_pct"] / 100, "sma_n": int(best["sma_n"]),
        "entry_trend_filter": True, **ATOMIC_FLAGS,
        "entry_trend_fast_sma": int(best["entry_trend_fast_sma"]),
        "entry_trend_slow_sma": int(best["entry_trend_slow_sma"]),
        "entry_volume_fast_window": int(best["entry_volume_fast_window"]),
        "entry_volume_slow_window": int(best["entry_volume_slow_window"]),
        "baseline_lookback": BASELINE_LOOKBACK, "baseline_max_rise": BASELINE_MAX_RISE,
        "baseline_rsi_period": BASELINE_RSI_PERIOD, "baseline_rsi_max": BASELINE_RSI_MAX,
        "stop_intraday": True, "cost_bps": best["cost_bps"],
    }
    stage1_file = f"search_{STRATEGY_VERSION}_stage1_results.csv"
    stage2_file = f"search_{STRATEGY_VERSION}_stage2_results.csv"
    stage1.to_csv(folder / stage1_file, index=False)
    stage2.to_csv(folder / stage2_file, index=False)
    metadata_path = folder / "search_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    except (OSError, ValueError):
        metadata = {}
    versions = metadata.setdefault("strategy_versions", {})
    # Migrate the previously flat record once, preserving its reproducible best combo.
    if "best_by_profit_factor" in metadata and "v1_full_conditions" not in versions:
        versions["v1_full_conditions"] = {
            "label": "v1 全条件过滤（历史结果）",
            "generated_at": metadata.get("generated_at"),
            "active_condition_ids": ["T0-01", "T0-02", "T0-03", "T0-04", "EN-01", "EN-02", "EN-03", "EN-04", "EX-01", "EX-02", "EX-03"],
            "best_by_profit_factor": metadata["best_by_profit_factor"],
            "result_files": ["search_stage1_results.csv", "search_stage2_results.csv"],
        }
    version_record = {
        "label": STRATEGY_LABEL,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "固定仓位", "search_method": "两阶段受约束搜索",
        "active_condition_ids": list(ACTIVE_CONDITION_IDS),
        "condition_flags": ATOMIC_FLAGS,
        "cost_bps": cost_bps, "stage1_combinations": len(stage1), "stage2_combinations": len(stage2),
        "best_by_profit_factor": {"parameters": best_params, "metrics": best},
        "result_files": [stage1_file, stage2_file],
        "risk_note": "最高盈利因子不代表已满足 15% 组合最大回撤硬条件；该字段仅包含独立标的回撤代理。",
    }
    versions[STRATEGY_VERSION] = version_record
    metadata.update({"generated_at": version_record["generated_at"], "group": group, "symbols": len(datasets),
                     "mode": "固定仓位", "search_method": "版本化两阶段受约束搜索",
                     "best_by_profit_factor": version_record["best_by_profit_factor"],
                     "active_strategy_version": STRATEGY_VERSION,
                     "risk_note": version_record["risk_note"]})
    (folder / "search_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"group_folder": folder.name, "group": group, "symbols": len(datasets), **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    folders = [path for path in sorted(GROUP_ROOT.iterdir()) if path.is_dir() and not path.name.startswith("10_")]
    summaries = []
    if args.workers == 1:
        # 方便受限运行环境或调试环境：避免 ProcessPool 启动失败时无输出地挂起。
        for folder in folders:
            summary = scan_group(str(folder))
            summaries.append(summary)
            print(f"完成：{folder.name}；PF={summary['profit_factor']:.3f}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(scan_group, str(folder)): folder.name for folder in folders}
            for future in as_completed(futures):
                name = futures[future]
                summary = future.result()
                summaries.append(summary)
                print(f"完成：{name}；PF={summary['profit_factor']:.3f}")
    overview = sort_results(pd.DataFrame(summaries))
    output = Path("reports")
    output.mkdir(exist_ok=True)
    overview.to_csv(output / f"group_domain_search_overview_{STRATEGY_VERSION}.csv", index=False)
    (output / f"group_domain_search_overview_{STRATEGY_VERSION}.md").write_text(
        "# 各特征组受约束搜索结果\n\n" + overview.to_markdown(index=False, floatfmt=".3f") + "\n",
        encoding="utf-8",
    )
    print(f"总览：reports/group_domain_search_overview_{STRATEGY_VERSION}.csv")


if __name__ == "__main__":
    main()
