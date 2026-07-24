"""在全量标的上分层搜索全部原子条件开关，并把收益率最佳组合写入 metadata。

这不是 2^14 的暴力穷举。先用 48 个确定性、覆盖每个条件 on/off 的候选进行
条件筛选，再对每组前两名条件组合执行完整的两阶段数值参数搜索。
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from rhein.backtest import input_files, load_ohlc
from rhein.strategy import active_condition_ids
from scripts.scan_all_group_domains import (
    BASELINE_LOOKBACK, BASELINE_MAX_RISE, BASELINE_RSI_MAX, BASELINE_RSI_PERIOD,
    CAPITAL, COMPOUND, COSTS, DOMAINS, ENTRY_LAGS, ENTRY_TREND_FAST_SMA,
    ENTRY_TREND_SLOW_SMA, ENTRY_VOLUME_FAST_WINDOW, ENTRY_VOLUME_SLOW_WINDOW,
    GROUP_ROOT, SMAS, evaluate, group_labels,
)

STRATEGY_VERSION = "v7_all_conditions_train70_test30"
STRATEGY_LABEL = "v7 全条件开关搜索（样本 70% / 测试 30%）"
OPTIMIZATION_TARGET = "median_symbol_cumulative_return_pct"
OPTIMIZATION_LABEL = "样本集中位标的累计收益率最高"
TRAIN_RATIO = 0.70
SCREENING_PROFILES = 48
FINALIST_PROFILES = 2
FLAG_KEYS = (
    "use_signal_band", "use_baseline_prior_low", "use_baseline_max_rise",
    "use_baseline_rsi", "use_baseline_close_above_fast_sma",
    "use_baseline_fast_above_slow_sma", "use_baseline_close_above_sma20",
    "use_baseline_volume_sma", "use_baseline_bullish_candle", "use_entry_close_vs_t0", "use_early_stop",
    "use_exit_below_entry", "use_exit_below_sma", "use_forced_exit",
    "use_forced_exit_intraday_protection",
)


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [OPTIMIZATION_TARGET, "profit_factor"], ascending=False, na_position="last"
    )


def base_params(vol_label: str, liquidity_label: str) -> dict:
    """条件筛选阶段的固定数值基线；最终阶段会重新扫描数值域。"""
    domain = DOMAINS[vol_label]
    lower, width = domain["lowers"][1], domain["widths"][0]
    return {
        "band_lo": lower / 100, "band_hi": (lower + width) / 100,
        "entry_lag": 2, "hard_stop_days": (3, 4),
        "stop_pct": domain["stops"][1] / 100, "sma_n": 5,
        "entry_trend_fast_sma": ENTRY_TREND_FAST_SMA,
        "entry_trend_slow_sma": ENTRY_TREND_SLOW_SMA,
        "entry_volume_fast_window": ENTRY_VOLUME_FAST_WINDOW,
        "entry_volume_slow_window": ENTRY_VOLUME_SLOW_WINDOW,
        "baseline_lookback": BASELINE_LOOKBACK,
        "baseline_max_rise": BASELINE_MAX_RISE,
        "baseline_rsi_period": BASELINE_RSI_PERIOD,
        "baseline_rsi_max": BASELINE_RSI_MAX,
        "forced_exit_day": 5,
        "forced_exit_intraday_stop_pct": .01,
        "stop_intraday": True,
        "cost_bps": COSTS[liquidity_label],
        "entry_trend_filter": True,
    }


def normalize_flags(flags: dict[str, bool]) -> dict[str, bool]:
    """EX-05 只有 EX-04 启用时才有含义，避免产生重复的无效组合。"""
    output = {key: bool(flags.get(key, False)) for key in FLAG_KEYS}
    if not output["use_forced_exit"]:
        output["use_forced_exit_intraday_protection"] = False
    return output


def profile_key(flags: dict[str, bool], stop_intraday: bool, forced_exit_day: int) -> tuple:
    return tuple(flags[key] for key in FLAG_KEYS) + (stop_intraday, forced_exit_day)


def condition_profiles(group_name: str) -> list[dict]:
    """包含全开/全关/逐项翻转，并由确定性随机样本补足至固定数量。"""
    current = {
        "use_signal_band": True, "use_baseline_prior_low": True,
        "use_baseline_max_rise": True, "use_baseline_rsi": False,
        "use_baseline_close_above_fast_sma": True,
        "use_baseline_fast_above_slow_sma": True,
        "use_baseline_close_above_sma20": True, "use_baseline_volume_sma": True,
        "use_baseline_bullish_candle": True,
        "use_entry_close_vs_t0": True, "use_early_stop": True,
        "use_exit_below_entry": True, "use_exit_below_sma": True,
        "use_forced_exit": False, "use_forced_exit_intraday_protection": False,
    }
    raw = [dict.fromkeys(FLAG_KEYS, False), dict.fromkeys(FLAG_KEYS, True), current]
    for key in FLAG_KEYS:
        flipped = current.copy()
        flipped[key] = not flipped[key]
        raw.append(flipped)
    rng = random.Random(f"rhein-v6:{group_name}")
    while len(raw) < SCREENING_PROFILES:
        raw.append({key: bool(rng.getrandbits(1)) for key in FLAG_KEYS})
    profiles, seen = [], set()
    for index, flags in enumerate(raw):
        flags = normalize_flags(flags)
        stop_intraday = bool(index % 2)
        forced_exit_day = (4, 5, 6)[index % 3]
        key = profile_key(flags, stop_intraday, forced_exit_day)
        if key not in seen:
            seen.add(key)
            profiles.append({"flags": flags, "stop_intraday": stop_intraday,
                             "forced_exit_day": forced_exit_day})
    return profiles


def record(params: dict, metrics: dict, stage: str, profile_no: int) -> dict:
    return {
        "搜索阶段": stage, "条件候选编号": profile_no,
        "启用条件": "、".join(active_condition_ids(params)) or "无",
        "早期止损触发": "盘中低价" if params["stop_intraday"] else "收盘价",
        "强制平仓日": f"t{params['forced_exit_day']}" if params["use_forced_exit"] else "未启用",
        "信号下限 (%)": params["band_lo"] * 100,
        "信号上限 (%)": params["band_hi"] * 100,
        "入场确认日": f"t{params['entry_lag']}",
        "早期止损日": ",".join(f"t{x}" for x in params["hard_stop_days"]),
        "止损幅度 (%)": params["stop_pct"] * 100, "SMA 周期": params["sma_n"],
        **metrics,
    }


def split_datasets(datasets: list[tuple[str, pd.DataFrame]]) -> tuple[list[tuple[str, pd.DataFrame]], dict[str, pd.Timestamp]]:
    """按每个标的自身的时间轴切分，返回训练数据与测试期首日。

    测试回测会使用完整数据和 signal_start，因此可读取训练期历史计算指标，
    但只接受测试期开始后的 t0，避免任何训练交易进入测试统计。
    """
    train, test_starts = [], {}
    for symbol, df in datasets:
        split_index = max(1, min(len(df) - 1, int(len(df) * TRAIN_RATIO)))
        train.append((symbol, df.iloc[:split_index].copy()))
        test_starts[symbol] = pd.Timestamp(df["Date"].iloc[split_index])
    return train, test_starts


def scan_group(folder_text: str) -> dict:
    folder = Path(folder_text)
    group, vol_label, liquidity_label = group_labels(folder)
    datasets = [(path.stem.upper(), load_ohlc(path)) for path in input_files(folder)]
    train_datasets, test_starts = split_datasets(datasets)
    baseline = base_params(vol_label, liquidity_label)

    screened: list[tuple[dict, dict, int]] = []
    for number, profile in enumerate(condition_profiles(group), start=1):
        params = {**baseline, **profile["flags"], "stop_intraday": profile["stop_intraday"],
                  "forced_exit_day": profile["forced_exit_day"]}
        screened.append((params, evaluate(train_datasets, params), number))
    screen_frame = sort_results(pd.DataFrame([
        record(params, metrics, "条件筛选", number) for params, metrics, number in screened
    ]))
    finalist_numbers = screen_frame.head(FINALIST_PROFILES)["条件候选编号"].tolist()
    finalists = [(params, number) for params, _, number in screened if number in finalist_numbers]

    domain = DOMAINS[vol_label]
    bands = [(lo, lo + width) for lo, width in product(domain["lowers"], domain["widths"])]
    final_rows: list[tuple[dict, dict, int]] = []
    for flags_base, number in finalists:
        stage1: list[tuple[dict, dict]] = []
        for band, entry in product(bands, ENTRY_LAGS):
            params = {**flags_base, "band_lo": band[0] / 100, "band_hi": band[1] / 100,
                      "entry_lag": entry, "hard_stop_days": (entry + 1, entry + 2),
                      "stop_pct": domain["stops"][1] / 100, "sma_n": 5}
            stage1.append((params, evaluate(train_datasets, params)))
            final_rows.append((params, stage1[-1][1], number))
        stage1_frame = sort_results(pd.DataFrame([
            {"i": index, **metrics} for index, (_, metrics) in enumerate(stage1)
        ])).head(3)
        for candidate in stage1_frame.itertuples(index=False):
            source, _ = stage1[int(candidate.i)]
            for stop, sma in product(domain["stops"], SMAS):
                params = {**source, "stop_pct": stop / 100, "sma_n": sma}
                final_rows.append((params, evaluate(train_datasets, params), number))

    final_frame = sort_results(pd.DataFrame([
        record(params, metrics, "完整复验", number) for params, metrics, number in final_rows
    ]))
    # sort_values 保留原始索引，因此可准确取回产生第一名记录的完整参数字典。
    best_params, best_metrics, _ = final_rows[int(final_frame.index[0])]
    test_metrics = evaluate(datasets, best_params, signal_starts=test_starts)

    screen_file = f"search_{STRATEGY_VERSION}_condition_screening.csv"
    final_file = f"search_{STRATEGY_VERSION}_full_results.csv"
    screen_frame.to_csv(folder / screen_file, index=False)
    final_frame.to_csv(folder / final_file, index=False)
    metadata_path = folder / "search_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    versions = metadata.setdefault("strategy_versions", {})
    version_record = {
        "label": STRATEGY_LABEL, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "固定仓位", "search_method": "48 条件候选筛选 + 前 2 名完整参数复验",
        "optimization_target": OPTIMIZATION_TARGET, "optimization_label": OPTIMIZATION_LABEL,
        "validation_split": {"method": "每标的按日期顺序切分", "train_ratio": TRAIN_RATIO,
                             "test_ratio": 1 - TRAIN_RATIO,
                             "test_indicator_history": "保留训练期历史，仅测试期 t0 计入交易"},
        "condition_search": {"all_atomic_conditions": list(FLAG_KEYS), "screening_profiles": len(screen_frame),
                             "finalist_profiles": FINALIST_PROFILES},
        "best_combo": {"parameters": best_params, "metrics": best_metrics,
                       "train_metrics": best_metrics, "test_metrics": test_metrics},
        "result_files": [screen_file, final_file],
        "risk_note": "只用样本集选择组合；测试集未参与搜索。收益率为有交易标的累计收益率中位数；回撤为独立标的回撤 25 分位数代理。",
    }
    versions[STRATEGY_VERSION] = version_record
    metadata.update({"generated_at": version_record["generated_at"], "group": group, "symbols": len(datasets),
                     "active_strategy_version": STRATEGY_VERSION, "best_combo": version_record["best_combo"],
                     "mode": "固定仓位", "search_method": version_record["search_method"],
                     "risk_note": version_record["risk_note"]})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"group": group, **best_metrics,
            "test_median_symbol_cumulative_return_pct": test_metrics["median_symbol_cumulative_return_pct"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    folders = [path for path in sorted(GROUP_ROOT.iterdir()) if path.is_dir() and not path.name.startswith("10_")]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_group, str(folder)): folder.name for folder in folders}
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(f"完成：{summary['group']}；样本中位收益={summary['median_symbol_cumulative_return_pct']:.3f}%；"
                  f"测试中位收益={summary['test_median_symbol_cumulative_return_pct']:.3f}%", flush=True)
    overview = sort_results(pd.DataFrame(summaries))
    report = Path("reports") / f"group_open_condition_search_overview_{STRATEGY_VERSION}.csv"
    overview.to_csv(report, index=False)
    print(f"总览：{report}")


if __name__ == "__main__":
    main()
