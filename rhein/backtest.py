"""动量突破策略回测，并生成中文 KPI 报告。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CONFIG_ROOT, REPORTS_ROOT
from .data.discovery import input_files
from .data.ohlc import load_ohlc
from .engine.kpis import calculate_kpis, cross_asset_summary
from .engine.indicators import compute_indicators
from .engine.eligibility import baseline_is_eligible, validate_run_parameters


DEFAULT_STRATEGY = {
    "band_lo": 0.02,
    "band_hi": 0.025,
    "stop_pct": 0.02,
    "sma_n": 5,
    "entry_trend_fast_sma": 5,
    "entry_trend_slow_sma": 10,
    "entry_volume_fast_window": 5,
    "entry_volume_slow_window": 20,
    "baseline_lookback": 15,
    "baseline_max_rise": 0.20,
    "baseline_rsi_period": 14,
    "baseline_rsi_max": 90,
    "entry_lag": 2,
    "hard_stop_days": (3, 4),
    # Every rule is independently switchable.  Keep all enabled so v1.3 is reproducible.
    "use_signal_band": True,
    "use_baseline_prior_low": True,
    "use_baseline_max_rise": True,
    "use_baseline_rsi": True,
    "use_baseline_close_above_fast_sma": True,
    "use_baseline_fast_above_slow_sma": True,
    "use_baseline_close_above_sma20": True,
    "use_baseline_volume_sma": True,
    "use_entry_close_vs_t0": True,
    "use_early_stop": True,
    "use_exit_below_entry": True,
    "use_exit_below_sma": True,
    "use_forced_exit": False,
    "forced_exit_day": 5,
    "use_forced_exit_intraday_protection": True,
    "forced_exit_intraday_stop_pct": 0.01,
    # Kept solely for old saved profiles; new callers should use the three atomic entry toggles.
    "entry_trend_filter": True,
    "stop_intraday": True,
    "cost_bps": 0.0,
}
PROFILE_KEYS = set(DEFAULT_STRATEGY)


def run_backtest(df: pd.DataFrame, band_lo=0.02, band_hi=0.025,
                 stop_pct=0.02, sma_n=5, entry_lag=2, hard_stop_days=(3, 4),
                 capital=10_000.0, compound=True, stop_intraday=True,
                 cost_bps=0.0, vol_scaled=False, vol_window=20,
                 entry_trend_filter=True, entry_trend_fast_sma=5,
                 entry_trend_slow_sma=10, baseline_lookback=15,
                 baseline_max_rise=0.20, baseline_rsi_period=14, baseline_rsi_max=90,
                 entry_volume_fast_window=5, entry_volume_slow_window=20,
                 use_signal_band=True, use_baseline_prior_low=True,
                 use_baseline_max_rise=True, use_baseline_rsi=True,
                 use_baseline_close_above_fast_sma=True,
                 use_baseline_fast_above_slow_sma=True,
                 use_baseline_close_above_sma20=True,
                 use_baseline_volume_sma=True, use_entry_close_vs_t0=True,
                 use_entry_close_above_fast_sma=None, use_entry_fast_above_slow_sma=None,
                 use_entry_volume_sma=None,
                 use_early_stop=True, use_exit_below_entry=True,
                 use_exit_below_sma=True, use_forced_exit=False,
                 forced_exit_day=5, forced_exit_days_after_entry=None,
                 use_forced_exit_intraday_protection=True,
                 forced_exit_intraday_stop_pct=0.01, signal_start=None):
    """执行动量突破策略。

    所有筛选与出场规则均可独立启停。入场确认窗口只使用 tN-1、tN，
    不使用 tN+1，避免未来函数。signal_start 可限制只接受该日期（含）
    之后的 t0；此前行情仍保留，用于测试集的指标预热。
    """
    # Deprecated EN-02/EN-03 keyword names are kept for saved profiles and
    # command-line callers. Their semantics now belong to the t0 baseline.
    if use_entry_close_above_fast_sma is not None:
        use_baseline_close_above_fast_sma = use_entry_close_above_fast_sma
    if use_entry_fast_above_slow_sma is not None:
        use_baseline_fast_above_slow_sma = use_entry_fast_above_slow_sma
    if use_entry_volume_sma is not None:
        use_baseline_volume_sma = use_entry_volume_sma
    # Accept the old relative-to-entry argument when loading historical saved combos.
    if forced_exit_days_after_entry is not None:
        forced_exit_day = entry_lag + forced_exit_days_after_entry
    validation_params = locals()
    validate_run_parameters(validation_params)
    c, o, lo, v = (df[name].to_numpy() for name in ("Close", "Open", "Low", "Volume"))
    signal_start = pd.Timestamp(signal_start) if signal_start is not None else None
    n = len(df)
    indicators = compute_indicators(
        df, sma_n=sma_n, entry_trend_fast_sma=entry_trend_fast_sma,
        entry_trend_slow_sma=entry_trend_slow_sma,
        entry_volume_fast_window=entry_volume_fast_window,
        entry_volume_slow_window=entry_volume_slow_window,
        baseline_rsi_period=baseline_rsi_period, vol_window=vol_window,
    )
    ret1, sma, entry_fast_sma, entry_slow_sma = (
        indicators[key] for key in ("ret1", "sma", "entry_fast_sma", "entry_slow_sma")
    )
    baseline_sma20, vol_fast_sma, vol_slow_sma, rsi, sig = (
        indicators[key] for key in ("baseline_sma20", "vol_fast_sma", "vol_slow_sma", "rsi", "sig")
    )

    trades, equity = [], capital
    i = 1
    while i < n - entry_lag:
        # 测试集保留训练期历史以计算指标，但不允许训练期 t0 产生任何交易。
        if signal_start is not None and df["Date"].iloc[i] < signal_start:
            i += 1
            continue
        if vol_scaled:
            if np.isnan(sig[i]) or sig[i] == 0:
                i += 1
                continue
            blo, bhi = band_lo * sig[i] / 0.01, band_hi * sig[i] / 0.01
        else:
            blo, bhi = band_lo, band_hi
        eligibility_params = validation_params | {"band_lo": blo, "band_hi": bhi}
        if not baseline_is_eligible(
            i, close=c, ret1=ret1, rsi=rsi, entry_fast_sma=entry_fast_sma,
            entry_slow_sma=entry_slow_sma, baseline_sma20=baseline_sma20,
            vol_fast_sma=vol_fast_sma, vol_slow_sma=vol_slow_sma, params=eligibility_params,
        ):
            i += 1
            continue

        t0, confirmation_idx = i, i + entry_lag
        if use_entry_close_vs_t0 and c[confirmation_idx] < c[t0]:
            i = t0 + 1
            continue
        entry_idx = confirmation_idx
        entry_px = c[entry_idx]
        effective_stop = (stop_pct * sig[entry_idx] / 0.01
                          if vol_scaled and not np.isnan(sig[entry_idx]) else stop_pct)
        stop_level = (1 - effective_stop) * entry_px
        exit_px = exit_idx = reason = None
        for j in range(entry_idx + 1, n):
            day_from_signal = j - t0
            if use_early_stop and day_from_signal in hard_stop_days:
                if stop_intraday and lo[j] <= stop_level:
                    exit_px = min(o[j], stop_level) if o[j] < stop_level else stop_level
                    exit_idx, reason = j, f"t{day_from_signal} 日内止损"
                    break
                if not stop_intraday and c[j] <= stop_level:
                    exit_px, exit_idx, reason = c[j], j, f"t{day_from_signal} 收盘止损"
                    break
            # 强制平仓为持有上限；若同日早期止损已触发，早期止损优先。
            if use_forced_exit and day_from_signal == forced_exit_day:
                forced_protection_level = entry_px * (1 - forced_exit_intraday_stop_pct)
                if use_forced_exit_intraday_protection and lo[j] <= forced_protection_level:
                    exit_px = min(o[j], forced_protection_level) if o[j] < forced_protection_level else forced_protection_level
                    exit_idx, reason = j, f"t{forced_exit_day} 强制日日内保护"
                    break
                exit_px, exit_idx, reason = c[j], j, f"t{forced_exit_day} 强制平仓"
                break
            if day_from_signal >= (max(hard_stop_days) + 1 if use_early_stop else entry_lag + 1):
                if use_exit_below_entry and c[j] < entry_px:
                    exit_px, exit_idx, reason = c[j], j, "收盘价低于入场价"
                    break
                if use_exit_below_sma and not np.isnan(sma[j]) and c[j] < sma[j]:
                    exit_px, exit_idx, reason = c[j], j, f"收盘价低于 SMA{sma_n}"
                    break
        if exit_px is None:
            exit_px, exit_idx, reason = c[-1], n - 1, "数据结束强制平仓"

        net_return = exit_px / entry_px - 1 - 2 * cost_bps / 10_000
        stake = equity if compound else capital
        pnl = stake * net_return
        equity += pnl
        trades.append({
            "signal": str(df["Date"].iloc[t0].date()),
            "entry": str(df["Date"].iloc[entry_idx].date()),
            "exit": str(df["Date"].iloc[exit_idx].date()),
            "days_held": exit_idx - entry_idx,
            "entry_px": round(float(entry_px), 4),
            "exit_px": round(float(exit_px), 4),
            "ret_pct": round(float(net_return * 100), 3),
            "pnl_eur": round(float(pnl), 2),
            "reason": reason,
        })
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    return trades_df, calculate_kpis(trades_df, capital, compound)


def display(value, suffix="") -> str:
    if value is None:
        return "不适用"
    return f"{value}{suffix}"


def markdown_report(results: list[dict], params: dict, generated_at: str) -> str:
    lines = ["# 动量突破策略回测报告", "", f"生成时间：{generated_at}", "",
             "## 本次参数", "", "| 参数 | 数值 |", "|---|---:|"]
    for key, value in params.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 各标的 KPI", ""]
    for result in results:
        stat = result["stats"]
        lines += [f"### {result['symbol']}（{stat['mode']}）", "",
                  "实际策略参数：" + "；".join([
                      f"信号区间 {result['strategy']['band_lo']:.2%}–{result['strategy']['band_hi']:.2%}",
                      f"确认日 t{result['strategy']['entry_lag']}",
                      (f"基准点：收盘>SMA{result['strategy'].get('entry_trend_fast_sma', 5)}；"
                       f"SMA{result['strategy'].get('entry_trend_fast_sma', 5)}>SMA{result['strategy'].get('entry_trend_slow_sma', 10)}；"
                       "收盘>SMA20；入场窗口 tN-1/tN：5日均量>20日均量"),
                      "早期止损日 " + ", ".join(f"t{d}" for d in result['strategy']['hard_stop_days']),
                      f"止损 {result['strategy']['stop_pct']:.2%}",
                      f"SMA{result['strategy']['sma_n']}",
                      "盘中止损" if result['strategy']['stop_intraday'] else "收盘止损",
                      f"单边手续费 {result['strategy']['cost_bps']} bps",
                  ]), "",
                  "| 指标 | 数值 |", "|---|---:|"]
        rows = [
            ("交易次数", stat["n_trades"]), ("初始资金", stat["initial_capital"]),
            ("最终权益", stat["final_equity"]), ("总盈亏", stat["total_pnl"]),
        ]
        if stat["n_trades"]:
            rows += [("胜率", display(stat["win_rate_pct"], "%")),
                     ("平均单笔收益", display(stat["avg_return_pct"], "%")),
                     ("平均盈利", display(stat["avg_win_pct"], "%")),
                     ("平均亏损", display(stat["avg_loss_pct"], "%")),
                     ("盈亏比", display(stat["payoff_ratio"])),
                     ("盈利因子 (Profit Factor)", display(stat["profit_factor"])),
                     ("最大回撤", display(stat["max_drawdown_pct"], "%")),
                     ("平均持仓天数", stat["avg_days_held"]),
                     ("最大单笔盈利", display(stat["max_win_pct"], "%")),
                     ("最大单笔亏损", display(stat["max_loss_pct"], "%"))]
        lines.extend(f"| {name} | {value} |" for name, value in rows)
        if stat["n_trades"]:
            reasons = result["trades"]["reason"].value_counts().to_dict()
            lines += ["", "退出原因：" + "；".join(f"{k} {v} 笔" for k, v in reasons.items()), ""]

    lines += ["", "## 跨标的合并统计（用于比较，不是组合回测）", "",
              "| 模式 | 交易次数 | 胜率 | 初始资金合计 | 最终权益合计 | 总盈亏合计 | 盈亏比 | 盈利因子 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in dict.fromkeys(r["stats"]["mode"] for r in results):
        summary = cross_asset_summary(results, mode)
        lines.append("| {mode} | {n_trades} | {win_rate} | {initial} | {final} | {pnl} | {payoff} | {pf} |".format(
            mode=mode, n_trades=summary["n_trades"],
            win_rate=display(summary["win_rate"], "%"), initial=summary["initial"],
            final=summary["final"], pnl=summary["pnl"],
            payoff=display(summary["payoff"]), pf=display(summary["pf"])))

    lines += ["", "## 指标计算与口径", "",
              "- **胜率** = 盈利交易数 ÷ 总交易数 × 100%。净收益率大于 0 才记为盈利，持平记为非盈利。",
              "- **最终权益**：复利模式为 `初始资金 × ∏(1 + 每笔净收益率)`；固定仓位模式为 `初始资金 + 各笔盈亏金额之和`。",
              "- **总盈亏** = 最终权益 − 初始资金。每笔净收益率已扣除设置的双边手续费。",
              "- **盈亏比** = 平均盈利收益率 ÷ 平均亏损收益率的绝对值，衡量单笔赚赔的幅度；没有亏损时显示“不适用”。",
              "- **盈利因子** = 所有盈利金额之和 ÷ 所有亏损金额绝对值之和，衡量总赚金额相对于总亏金额；没有亏损时显示“不适用”。",
              "- **最大回撤** = 权益曲线相对之前历史最高点的最大跌幅。数值越接近 0，回撤越小。",
              "", "## 解释提醒", "",
              "每个 CSV 均以独立初始资金运行。因此“跨标的合并统计”将独立账户的资金和逐笔交易汇总，只用于观察总体交易分布；它没有按交易日期模拟资金分配或持仓重叠，不能视作一个同时持仓的组合回测。"]
    return "\n".join(lines) + "\n"


def parse_days(value: str) -> tuple[int, ...]:
    try:
        days = tuple(int(day.strip()) for day in value.split(",") if day.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("止损日必须是逗号分隔的整数，例如 3,4") from exc
    if not days or any(day < 1 for day in days):
        raise argparse.ArgumentTypeError("止损日必须是正整数，例如 3,4")
    return days


def load_profiles(path: str | None) -> dict:
    if path is None:
        return {}
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ValueError(f"找不到参数档案文件：{profile_path}")
    with profile_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("参数档案必须是一个 JSON 对象")
    profiles = {}
    for symbol, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"{symbol} 的参数档案必须是 JSON 对象")
        unknown = set(values) - PROFILE_KEYS
        if unknown:
            raise ValueError(f"{symbol} 的未知参数：{', '.join(sorted(unknown))}")
        profile = dict(values)
        if "hard_stop_days" in profile:
            profile["hard_stop_days"] = tuple(profile["hard_stop_days"])
        profiles[symbol.upper()] = profile
    return profiles


def resolve_strategy(symbol: str, profiles: dict, args) -> dict:
    """档案优先于基础默认值，命令行显式参数优先于档案。"""
    strategy = DEFAULT_STRATEGY | profiles.get("DEFAULT", {}) | profiles.get(symbol.upper(), {})
    overrides = {
        "band_lo": args.band_lo, "band_hi": args.band_hi, "stop_pct": args.stop_pct,
        "sma_n": args.sma, "entry_lag": args.entry_lag,
        "entry_trend_fast_sma": args.entry_trend_fast_sma,
        "entry_trend_slow_sma": args.entry_trend_slow_sma,
        "hard_stop_days": args.hard_stop_days, "cost_bps": args.cost_bps,
    }
    strategy.update({key: value for key, value in overrides.items() if value is not None})
    if args.close_stop:
        strategy["stop_intraday"] = False
    if strategy["band_lo"] <= 0 or strategy["band_hi"] <= strategy["band_lo"]:
        raise ValueError(f"{symbol}：涨幅区间必须满足 0 < 下限 < 上限")
    if strategy["entry_lag"] < 1:
        raise ValueError(f"{symbol}：确认日 entry_lag 至少为 1")
    if strategy["entry_trend_fast_sma"] < 2 or strategy["entry_trend_slow_sma"] <= strategy["entry_trend_fast_sma"]:
        raise ValueError(f"{symbol}：入场趋势 SMA 必须满足 2 ≤ 快线周期 < 慢线周期")
    if any(day <= strategy["entry_lag"] for day in strategy["hard_stop_days"]):
        raise ValueError(f"{symbol}：早期止损日必须晚于确认日 t{strategy['entry_lag']}")
    return strategy


def load_sweep_grid(path: str) -> dict:
    """读取参数扫描范围；每个标的可有不同的候选值。"""
    grid_path = Path(path)
    if not grid_path.is_file():
        raise ValueError(f"找不到参数扫描文件：{grid_path}")
    with grid_path.open(encoding="utf-8") as fh:
        grid = json.load(fh)
    required = {"band_ranges", "entry_lags", "stop_pcts"}
    for symbol, values in grid.items():
        missing = required - set(values)
        if missing:
            raise ValueError(f"{symbol} 的扫描范围缺少：{', '.join(sorted(missing))}")
    return {symbol.upper(): values for symbol, values in grid.items()}


def sweep_report(rows: list[dict], generated_at: str, mode: str, grid_path: str) -> str:
    lines = ["# 动量突破策略参数扫描报告", "", f"生成时间：{generated_at}",
             f"资金模式：{mode}（每个组合以 10,000 初始资金独立运行）", f"扫描范围：`{grid_path}`", "",
             "## 全部参数组合（按标的、盈利因子由高到低）", "",
             "| 标的 | 信号区间 | 确认日 | 早期止损日 | 止损 | SMA | 交易数 | 胜率 | 平均收益 | 总盈亏 | 最终权益 | 盈亏比 | 盈利因子 | 最大回撤 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        s = row["stats"]
        lines.append(
            f"| {row['symbol']} | {row['band_lo']:.2%}–{row['band_hi']:.2%} | "
            f"t{row['entry_lag']} | {', '.join('t'+str(d) for d in row['hard_stop_days'])} | "
            f"{row['stop_pct']:.2%} | {row['sma_n']} | {s['n_trades']} | "
            f"{display(s.get('win_rate_pct'), '%')} | {display(s.get('avg_return_pct'), '%')} | "
            f"{s['total_pnl']:.2f} | {s['final_equity']:.2f} | {display(s.get('payoff_ratio'))} | "
            f"{display(s.get('profit_factor'))} | {display(s.get('max_drawdown_pct'), '%')} |")
    lines += ["", "## 如何解读", "",
              "- 每一行只改变信号涨幅区间、确认日和早期止损幅度；确认日为 tN 时，早期止损自动放在 t(N+1)、t(N+2)，因此策略结构保持一致。",
              "- 优先同时观察**盈利因子、最大回撤、交易数和不同时间段的稳定性**，而不是仅按最终权益或胜率挑第一名。",
              "- 这是使用同一份历史数据的样本内扫描。表中最佳组合只是假设候选，必须在未参与调参的时间段或未来数据中验证，才能避免过拟合。"]
    return "\n".join(lines) + "\n"


def run_sweep(files: list[Path], profiles: dict, args) -> None:
    grid = load_sweep_grid(args.grid_config)
    compound = not args.no_compound
    rows = []
    for file_path in files:
        symbol = file_path.stem.upper()
        if symbol not in grid:
            print(f"[扫描跳过] {symbol}：在 {args.grid_config} 中没有定义候选范围")
            continue
        df, base, candidates = load_ohlc(file_path), resolve_strategy(symbol, profiles, args), grid[symbol]
        for band_lo, band_hi in candidates["band_ranges"]:
            for entry_lag in candidates["entry_lags"]:
                for stop_pct in candidates["stop_pcts"]:
                    for sma_n in candidates.get("sma_ns", [base["sma_n"]]):
                        strategy = base | {"band_lo": band_lo, "band_hi": band_hi,
                                           "entry_lag": entry_lag,
                                           "hard_stop_days": (entry_lag + 1, entry_lag + 2),
                                           "stop_pct": stop_pct, "sma_n": sma_n}
                        _, stats = run_backtest(df, capital=args.capital, compound=compound, **strategy)
                        rows.append({"symbol": symbol, **strategy, "stats": stats})
    if not rows:
        raise ValueError("没有可扫描的参数组合")
    rows.sort(key=lambda r: (r["symbol"], -(r["stats"].get("profit_factor") or -1),
                             -r["stats"].get("avg_return_pct", -999)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"sweep_report_{stamp}.md"
    csv_path = output_dir / f"sweep_results_{stamp}.csv"
    json_path = output_dir / f"sweep_results_{stamp}.json"
    mode = "复利" if compound else "固定仓位"
    report_path.write_text(sweep_report(rows, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        mode, args.grid_config), encoding="utf-8")
    flat_rows = [{key: value for key, value in row.items() if key not in {"stats", "hard_stop_days"}} |
                 {"hard_stop_days": ",".join(map(str, row["hard_stop_days"]))} | row["stats"] for row in rows]
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    print(f"参数扫描完成：{len(rows)} 个组合（{mode}）")
    for symbol in dict.fromkeys(row["symbol"] for row in rows):
        best = next(row for row in rows if row["symbol"] == symbol)
        s = best["stats"]
        print(f"- {symbol} 当前排序第一：{best['band_lo']:.2%}–{best['band_hi']:.2%}，"
              f"t{best['entry_lag']}，止损 {best['stop_pct']:.2%}，"
              f"盈利因子 {display(s.get('profit_factor'))}，最大回撤 {display(s.get('max_drawdown_pct'), '%')}")
    print(f"完整报告：{report_path}")
    print(f"结构化结果：{csv_path} / {json_path}")


def main():
    ap = argparse.ArgumentParser(description="动量突破策略批量回测与中文 KPI 报告")
    ap.add_argument("input", help="单个 CSV 文件或包含 CSV 的目录，例如 data")
    ap.add_argument("--profile-config", default=str(CONFIG_ROOT / "strategy_profiles.json"),
                    help="按股票代码加载 JSON 参数档案；传空字符串禁用")
    ap.add_argument("--band-lo", type=float, help="覆盖所有标的档案的信号涨幅下限")
    ap.add_argument("--band-hi", type=float, help="覆盖所有标的档案的信号涨幅上限")
    ap.add_argument("--stop-pct", type=float, help="覆盖所有标的档案的早期止损比例")
    ap.add_argument("--sma", type=int, help="覆盖所有标的档案的 SMA 周期")
    ap.add_argument("--entry-lag", type=int, help="信号后第几日确认入场：1=t1，2=t2，3=t3")
    ap.add_argument("--entry-trend-fast-sma", type=int, help="入场趋势过滤的快线 SMA 周期")
    ap.add_argument("--entry-trend-slow-sma", type=int, help="入场趋势过滤的慢线 SMA 周期")
    ap.add_argument("--hard-stop-days", type=parse_days,
                    help="确认入场后、以信号日计的早期止损考察日，例如 3,4")
    ap.add_argument("--capital", type=float, default=10_000)
    ap.add_argument("--cost-bps", type=float, help="覆盖所有标的档案的单边手续费，单位为基点")
    ap.add_argument("--compound", action="store_true", help="使用复利模式（默认使用固定仓位）")
    ap.add_argument("--both-modes", action="store_true", help="同时运行复利与固定仓位模式")
    ap.add_argument("--no-compound", action="store_true", help="兼容旧命令；固定仓位本来就是默认值")
    ap.add_argument("--close-stop", action="store_true", help="t3/t4 以收盘价而不是盘中低价触发止损")
    ap.add_argument("--output-dir", default=str(REPORTS_ROOT), help="报告输出目录")
    ap.add_argument("--sweep", action="store_true", help="执行参数组合扫描并生成完整比较报告")
    ap.add_argument("--grid-config", default=str(CONFIG_ROOT / "parameter_grid.json"), help="参数扫描范围 JSON 文件")
    args = ap.parse_args()

    profiles = load_profiles(args.profile_config or None)
    files = input_files(Path(args.input))
    if args.sweep:
        run_sweep(files, profiles, args)
        return
    run_params = {"每标的初始资金": args.capital,
                  "参数档案": args.profile_config or "未使用（基础默认值）",
                  "命令行覆盖": "已使用" if any(value is not None for value in
                      (args.band_lo, args.band_hi, args.stop_pct, args.sma, args.entry_lag,
                       args.entry_trend_fast_sma, args.entry_trend_slow_sma,
                       args.hard_stop_days, args.cost_bps)) else "无"}
    results = []
    modes = [True, False] if args.both_modes else ([True] if args.compound else [False])
    for file_path in files:
        df = load_ohlc(file_path)
        strategy = resolve_strategy(file_path.stem, profiles, args)
        for compound in modes:
            trades, stats = run_backtest(df, capital=args.capital, compound=compound, **strategy)
            results.append({"symbol": file_path.stem.upper(), "source": str(file_path),
                            "stats": stats, "trades": trades, "strategy": strategy})

    # 微秒避免自动化或连续命令在同一秒内覆盖上一份报告。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{stamp}.md"
    trades_path = output_dir / f"trades_{stamp}.csv"
    summary_path = output_dir / f"summary_{stamp}.json"
    stock_kpis_path = output_dir / f"stock_kpis_{stamp}.csv"
    report_path.write_text(markdown_report(results, run_params, datetime.now().strftime("%Y-%m-%d %H:%M:%S")), encoding="utf-8")
    trade_frames = []
    for result in results:
        frame = result["trades"].copy()
        if not frame.empty:
            frame.insert(0, "symbol", result["symbol"])
            frame.insert(1, "mode", result["stats"]["mode"])
            trade_frames.append(frame)
    (pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()).to_csv(trades_path, index=False)
    serializable = [{"symbol": r["symbol"], "source": r["source"], "strategy": r["strategy"],
                     "stats": r["stats"]} for r in results]
    summary_path.write_text(json.dumps({"generated_at": stamp, "parameters": run_params,
                                        "results": serializable}, ensure_ascii=False, indent=2), encoding="utf-8")
    stock_kpis = []
    for result in results:
        strategy = dict(result["strategy"])
        strategy["hard_stop_days"] = ",".join(map(str, strategy["hard_stop_days"]))
        stock_kpis.append({"symbol": result["symbol"], "source": result["source"],
                           **strategy, **result["stats"]})
    pd.DataFrame(stock_kpis).to_csv(stock_kpis_path, index=False)

    print("\n回测完成：")
    for result in results:
        stat = result["stats"]
        print(f"- {result['symbol']} / {stat['mode']}：交易 {stat['n_trades']} 笔，"
              f"胜率 {display(stat.get('win_rate_pct'), '%')}，总盈亏 {stat['total_pnl']:.2f}，"
              f"最终权益 {stat['final_equity']:.2f}，盈亏比 {display(stat.get('payoff_ratio'))}")
    print(f"中文报告：{report_path}")
    print(f"交易明细：{trades_path}")
    print(f"JSON 汇总：{summary_path}")
    print(f"逐标的 KPI：{stock_kpis_path}")


if __name__ == "__main__":
    main()
