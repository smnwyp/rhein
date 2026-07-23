"""Streamlit UI：动量突破策略的参数回测与扫描。"""
from __future__ import annotations

from itertools import product
import importlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rhein.backtest as _backtest
from rhein.paths import DATA_ROOT, GROUP_ROOT
from rhein.strategy import (
    ATOMIC_TOGGLE_KEYS,
    DEFAULT_CONDITION_STATES,
    active_condition_ids,
    parse_ints,
    parse_percent_list,
    parse_percent_ranges,
)
# Streamlit 保留已 import 的模块。策略函数新增参数时，已运行的本地应用会
# 同时拿到新 UI/metadata 与旧函数对象，造成 "unexpected keyword argument"。
# 只在检测到函数签名过旧时 reload；日常 rerun 不 reload，避免不必要的状态扰动。
if "use_baseline_close_above_fast_sma" not in _backtest.run_backtest.__code__.co_varnames:
    _backtest = importlib.reload(_backtest)
input_files, load_ohlc, run_backtest = _backtest.input_files, _backtest.load_ohlc, _backtest.run_backtest


st.set_page_config(page_title="Momentum Breakout 回测", layout="wide")
# Streamlit Cloud 的工作目录不保证等于仓库根目录；数据路径由 rhein.paths
# 统一从仓库根目录解析，以免分组选择器因相对路径失效而静默消失。
DEFAULT_MAX_DRAWDOWN_PCT = 15.0

KPI_LABELS = {
    "标的": "标的", "n_trades": "交易次数", "win_rate_pct": "胜率 (%)",
    "cumulative_return_pct": "累计收益率 (%)", "payoff_ratio": "盈亏比", "profit_factor": "盈利因子",
    "max_drawdown_pct": "最大回撤 (%)", "avg_return_pct": "平均单笔收益 (%)",
    "avg_days_held": "平均持仓天数",
}
KPI_DEFINITIONS = {
    "交易次数": "实际开仓后完成平仓的次数；同一股票持仓期间的新信号不计入。",
    "胜率 (%)": "盈利交易数 ÷ 总交易数 × 100%；净收益率大于 0 才计为盈利。",
    "累计收益率 (%)": "(最终权益 ÷ 初始资金 − 1) × 100%；用于横向比较不同标的，避免相加独立账户金额。",
    "盈亏比": "平均盈利收益率 ÷ |平均亏损收益率|；衡量单笔平均赚赔幅度。",
    "盈利因子": "所有盈利金额之和 ÷ |所有亏损金额之和|；大于 1 表示历史总盈利额高于总亏损额。",
    "最大回撤 (%)": "min(权益_t ÷ 截至 t 的历史最高权益 − 1) × 100%；越接近 0，回撤越小。",
    "平均单笔收益 (%)": "所有单笔净收益率的算术平均值；单笔净收益率 = 出场价 ÷ 入场价 − 1 − 双边手续费。",
    "平均持仓天数": "所有交易从入场日至出场日的交易日间隔的平均值。",
}


def params_to_text(params: dict) -> str:
    on = lambda key: params.get(key, True)
    rules = []
    if on("use_signal_band"):
        rules.append(f"信号 {params['band_lo']:.2%}–{params['band_hi']:.2%}")
    if on("use_baseline_prior_low"):
        rules.append(f"{params['baseline_lookback']}日低点不在t0")
    if on("use_baseline_max_rise"):
        rules.append(f"相对低点延伸≤{params['baseline_max_rise']:.0%}")
    if on("use_baseline_rsi"):
        rules.append(f"RSI{params['baseline_rsi_period']}≤{params['baseline_rsi_max']}")
    if on("use_baseline_close_above_fast_sma"):
        rules.append(f"t0收盘>SMA{params['entry_trend_fast_sma']}")
    if on("use_baseline_fast_above_slow_sma"):
        rules.append(f"t0 SMA{params['entry_trend_fast_sma']}>SMA{params['entry_trend_slow_sma']}")
    if on("use_baseline_close_above_sma20"):
        rules.append("t0收盘>SMA20")
    if on("use_baseline_volume_sma"):
        rules.append(f"t0量SMA{params['entry_volume_fast_window']}>量SMA{params['entry_volume_slow_window']}")
    if on("use_entry_close_vs_t0"):
        rules.append("tN收盘≥t0")
    entry = []
    if entry:
        rules.append("tN-1/tN任一天：" + "且".join(entry))
    if on("use_early_stop"):
        rules.append(f"早期止损t{','.join(map(str, params['hard_stop_days']))} / {params['stop_pct']:.2%}")
    exits = []
    if on("use_exit_below_entry"):
        exits.append("跌破入场价")
    if on("use_exit_below_sma"):
        exits.append(f"跌破SMA{params['sma_n']}")
    if on("use_forced_exit"):
        forced = f"t{params.get('forced_exit_day', 5)}强制平仓"
        if on("use_forced_exit_intraday_protection"):
            forced += f"（日内跌{params.get('forced_exit_intraday_stop_pct', .01):.1%}先卖）"
        exits.append(forced)
    if exits:
        rules.append("后期出场：" + "或".join(exits))
    return "；".join(rules) + f"；单边费用 {params['cost_bps']:.1f} bps"


def strategy_narrative(params: dict) -> str:
    """将当前参数整理为可直接阅读的策略规则。"""
    stop_days = "、".join(f"t{day}" for day in params["hard_stop_days"])
    after_day = max(params["hard_stop_days"]) + 1 if params.get("use_early_stop", True) else params["entry_lag"] + 1
    on = lambda key: params.get(key, True)
    t0 = []
    if on("use_signal_band"): t0.append(f"单日涨幅在 {params['band_lo']:.1%}–{params['band_hi']:.1%}")
    if on("use_baseline_prior_low"): t0.append(f"t0-{params['baseline_lookback']} 至t0低点不在t0")
    if on("use_baseline_max_rise"): t0.append(f"相对该低点涨幅不超过{params['baseline_max_rise']:.0%}")
    if on("use_baseline_rsi"): t0.append(f"RSI({params['baseline_rsi_period']})不高于{params['baseline_rsi_max']}")
    if on("use_baseline_close_above_fast_sma"): t0.append(f"收盘>SMA{params['entry_trend_fast_sma']}")
    if on("use_baseline_fast_above_slow_sma"): t0.append(f"SMA{params['entry_trend_fast_sma']}>SMA{params['entry_trend_slow_sma']}")
    if on("use_baseline_close_above_sma20"): t0.append("收盘>SMA20")
    if on("use_baseline_volume_sma"): t0.append(f"量SMA{params['entry_volume_fast_window']}>量SMA{params['entry_volume_slow_window']}")
    entry = []
    if on("use_entry_close_vs_t0"): entry.append("tN收盘不低于t0")
    paragraphs = [
        f"基准点：t0 需满足“{'、'.join(t0) if t0 else '无基准筛选'}”。",
        f"入场点：在 t{params['entry_lag']}，需满足“{'、'.join(entry) if entry else '无入场确认筛选'}”后按收盘价入场。",
    ]
    if on("use_early_stop"):
        paragraphs.append(f"早期出场：在 {stop_days}，收盘价触及 {params['stop_pct']:.1%} 止损幅度即按收盘价出场。")
    exits = []
    if on("use_exit_below_entry"): exits.append("跌破入场价")
    if on("use_exit_below_sma"): exits.append(f"跌破SMA{params['sma_n']}")
    forced_exit_day = params.get("forced_exit_day", 5)
    if exits:
        if on("use_forced_exit") and forced_exit_day <= after_day:
            paragraphs.append(
                f"后期技术出场：原规则从 t{after_day} 起，收盘{'或'.join(exits)}时出场；"
                f"但 EX-04 会在 t{forced_exit_day} 强制平仓并优先执行，因此本组合下该规则不会实际触发。"
            )
        elif on("use_forced_exit"):
            paragraphs.append(
                f"后期技术出场：从 t{after_day} 至 t{forced_exit_day - 1}，收盘{'或'.join(exits)}时出场。"
            )
        else:
            paragraphs.append(f"后期技术出场：从 t{after_day} 起，收盘{'或'.join(exits)}时出场。")
    else:
        paragraphs.append(f"后期技术出场：从 t{after_day} 起不设后期技术出场。")
    if on("use_forced_exit"):
        forced_text = f"强制平仓：t{forced_exit_day} 当天必须平仓。"
        if on("use_forced_exit_intraday_protection"):
            forced_text = (
                f"强制平仓：t{forced_exit_day} 当天，若盘中任意即时价格触及入场价下"
                f"{params.get('forced_exit_intraday_stop_pct', .01):.1%}，立即按保护规则出场；否则以 t{forced_exit_day} 收盘价强制平仓。"
            )
        paragraphs.append(forced_text)
    return "\n\n".join(paragraphs)


def available_data_scopes() -> dict[str, str]:
    """返回 UI 可选数据范围及其目录；分组不存在时仍可使用基础数据。"""
    scopes = {
        "示例数据（NVDA、TSLA）": str(DATA_ROOT),
        "全部 Nasdaq 当前股票池": str(DATA_ROOT / "nasdaq_10y"),
    }
    if GROUP_ROOT.is_dir():
        for folder in sorted(path for path in GROUP_ROOT.iterdir() if path.is_dir()):
            manifest = folder / "group_manifest.csv"
            try:
                count = len(pd.read_csv(manifest, usecols=["symbol"]))
            except (FileNotFoundError, ValueError):
                count = "?"
            label = f"{folder.name.replace('_', ' ')}（{count} 个标的）"
            scopes[label] = str(folder)
    scopes["自定义路径"] = ""
    return scopes


def load_group_preset(data_path: str) -> dict | None:
    """读取分组搜索完成后写入的最佳组合；非分组目录返回 None。"""
    metadata_path = Path(data_path) / "search_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        import json
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def group_preset_versions(metadata: dict) -> dict[str, dict]:
    """Return versioned presets, with a read-only fallback for old metadata."""
    versions = metadata.get("strategy_versions")
    if isinstance(versions, dict) and versions:
        return versions
    if "best_by_profit_factor" in metadata:
        return {"v1_legacy": {"label": "v1 历史最佳组合", **metadata}}
    return {}


def load_group_best_overview() -> pd.DataFrame:
    """One best-combo row per group, using each group's active versioned metadata record."""
    rows = []
    for folder in sorted(path for path in GROUP_ROOT.glob("[0-9][0-9]_*")
                         if path.is_dir() and not path.name.startswith("10_")):
        metadata = load_group_preset(str(folder)) or {}
        versions = group_preset_versions(metadata)
        version_id = metadata.get("active_strategy_version") or next(iter(versions), None)
        record = versions.get(version_id, {})
        best = record.get("best_by_profit_factor", metadata.get("best_by_profit_factor", {}))
        params, metrics = best.get("parameters", {}), best.get("metrics", {})
        if not params or not metrics:
            continue
        rows.append({
            "策略分组": metadata.get("group", folder.name.replace("_", "－")),
            "策略版本": record.get("label", version_id or "历史版本"),
            "启用条件": "、".join(record.get("active_condition_ids", active_condition_ids(params))),
            "信号区间 (%)": f"{params['band_lo'] * 100:.2f}–{params['band_hi'] * 100:.2f}",
            "入场确认日": f"t{params['entry_lag']}",
            "早期止损幅度 (%)": params["stop_pct"] * 100,
            "趋势 SMA 周期": params["sma_n"],
            "交易数": metrics.get("trades"), "合并胜率 (%)": metrics.get("win_rate_pct"),
            "盈利因子": metrics.get("profit_factor"),
            "中位标的累计收益 (%)": metrics.get("median_symbol_cumulative_return_pct"),
            "回撤25分位数 (%)": metrics.get("q25_individual_max_drawdown_pct"),
        })
    return pd.DataFrame(rows)


def saved_combos_path(data_path: str) -> Path | None:
    """Saved combos are intentionally scoped to a concrete feature-group folder."""
    folder = Path(data_path)
    return folder / "saved_combos.json" if (folder / "group_manifest.csv").is_file() else None


def load_saved_combos(data_path: str) -> list[dict]:
    path = saved_combos_path(data_path)
    if path is None or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        combos = payload.get("combos", [])
        return combos if isinstance(combos, list) else []
    except (OSError, ValueError):
        return []


def normalize_parameters(raw_params: dict) -> dict:
    """补齐旧搜索 metadata 中不存在的策略字段。

    搜索结果会长期保存在各分组目录，而策略条件会持续新增。这里是唯一的
    向后兼容边界：无论预设来自历史 metadata 还是用户保存的组合，进入 UI 前
    都先变成当前版本可运行的完整参数，不能让缺一个新字段导致整个页面停止。
    """
    params = dict(raw_params)
    # EN-02/EN-03 were renamed to T0-05/T0-06 and now evaluate at t0.
    # Historical presets retain their intent while no longer passing obsolete
    # keyword arguments into the engine.
    legacy_baseline_flags = {
        "use_entry_close_above_fast_sma": "use_baseline_close_above_fast_sma",
        "use_entry_fast_above_slow_sma": "use_baseline_fast_above_slow_sma",
        "use_entry_volume_sma": "use_baseline_volume_sma",
    }
    for old_key, new_key in legacy_baseline_flags.items():
        if new_key not in params and old_key in params:
            params[new_key] = params[old_key]
        params.pop(old_key, None)
    defaults = {
        "band_lo": .02, "band_hi": .025, "entry_lag": 2,
        "hard_stop_days": (3, 4), "stop_pct": .02, "sma_n": 5,
        "cost_bps": 0.0, "stop_intraday": True,
        "entry_trend_fast_sma": 5, "entry_trend_slow_sma": 10,
        "entry_volume_fast_window": 5, "entry_volume_slow_window": 20,
        "baseline_lookback": 15, "baseline_max_rise": .20,
        "baseline_rsi_period": 14, "baseline_rsi_max": 90,
        "forced_exit_intraday_stop_pct": .01,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    params["hard_stop_days"] = tuple(params["hard_stop_days"])
    params.setdefault(
        "forced_exit_day",
        params["entry_lag"] + params.get("forced_exit_days_after_entry", 3),
    )
    for toggle_key, default in DEFAULT_CONDITION_STATES.items():
        params.setdefault(toggle_key, default)
    return params


def reset_group_session_state() -> None:
    """切换数据组时清理旧组绑定的结果及可能互相冲突的策略控件。"""
    group_keys = (
        "applied_preset_token", "single_kpis", "single_trades", "single_params",
        "single_mode", "chart_trade_index", "top_symbol_table", "saved_combo_name",
    )
    # 新组应从一套有效默认参数开始。否则上一组预设可能出现例如 t4 入场却
    # 沿用 t3 止损日的矛盾组合，导致参数校验 st.stop()，表面上表现为白屏。
    parameter_keys = (
        "band_range", "entry_lag", "stop_days_text", "stop_pct", "close_stop", "sma_n",
        "cost_bps", "entry_trend_fast_sma", "entry_trend_slow_sma",
        "entry_volume_fast_window", "entry_volume_slow_window", "baseline_lookback",
        "baseline_max_rise_pct", "baseline_rsi_period", "baseline_rsi_max",
        "forced_exit_day", "forced_exit_intraday_stop_pct", *ATOMIC_TOGGLE_KEYS,
    )
    for key in (*group_keys, *parameter_keys):
        st.session_state.pop(key, None)


def apply_parameters_to_controls(params: dict, token: str) -> None:
    """Synchronize a saved/preset parameter dict into sidebar controls before rerun."""
    params = normalize_parameters(params)
    st.session_state.update({
        "band_range": (round(params["band_lo"] * 100, 1), round(params["band_hi"] * 100, 1)),
        "entry_lag": int(params["entry_lag"]),
        "stop_days_text": ",".join(map(str, params["hard_stop_days"])),
        "stop_pct": round(params["stop_pct"] * 100, 1),
        # 历史 metadata 在 EX-01/EX-04 加入前不含这些字段；读取预设时必须补齐，
        # 否则切换分组会在侧边栏初始化阶段因 KeyError 而停止渲染。
        "close_stop": not bool(params.get("stop_intraday", True)), "sma_n": int(params["sma_n"]),
        "entry_trend_fast_sma": int(params.get("entry_trend_fast_sma", 5)),
        "entry_trend_slow_sma": int(params.get("entry_trend_slow_sma", 10)),
        "entry_volume_fast_window": int(params.get("entry_volume_fast_window", 5)),
        "entry_volume_slow_window": int(params.get("entry_volume_slow_window", 20)),
        "baseline_lookback": int(params.get("baseline_lookback", 15)),
        "baseline_max_rise_pct": float(params.get("baseline_max_rise", .20) * 100),
        "baseline_rsi_period": int(params.get("baseline_rsi_period", 14)),
        "baseline_rsi_max": int(params.get("baseline_rsi_max", 90)),
        "cost_bps": float(params["cost_bps"]),
        "forced_exit_day": int(params.get("forced_exit_day", params.get("entry_lag", 2) + params.get("forced_exit_days_after_entry", 3))),
        "forced_exit_intraday_stop_pct": float(params.get("forced_exit_intraday_stop_pct", .01) * 100),
        **{key: bool(params.get(key, DEFAULT_CONDITION_STATES[key])) for key in ATOMIC_TOGGLE_KEYS},
        "applied_preset_token": token,
    })


def load_saved_combo_to_controls(combo: dict, data_path: str) -> None:
    apply_parameters_to_controls(combo["parameters"], f"saved:{data_path}:{combo['id']}")
    st.session_state[f"strategy_version_choice::{Path(data_path).name}"] = "custom"


def save_combo(data_path: str, name: str, params: dict) -> None:
    path = saved_combos_path(data_path)
    if path is None:
        raise ValueError("只有特征组目录可以保存组合。")
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("请为组合填写名称。")
    combos = load_saved_combos(data_path)
    if any(item.get("name", "").casefold() == clean_name.casefold() for item in combos):
        raise ValueError("该组内已有同名组合；请使用不同名称。")
    serializable_params = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
    now = datetime.now().isoformat(timespec="seconds")
    combos.append({"id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}", "name": clean_name,
                   "created_at": now, "active_condition_ids": active_condition_ids(params),
                   "parameters": serializable_params})
    path.write_text(json.dumps({"schema_version": 1, "group_folder": Path(data_path).name,
                                "updated_at": now, "combos": combos}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_group_combinations() -> pd.DataFrame:
    """汇总各成熟分组已经落盘的两阶段搜索结果，不重新执行回测。"""
    column_names = {
        "signal_lo_pct": "信号下限 (%)", "signal_hi_pct": "信号上限 (%)",
        "entry_day": "入场确认日", "entry_trend_filter": "入场趋势过滤",
        "entry_trend_fast_sma": "入场趋势快线 SMA", "entry_trend_slow_sma": "入场趋势慢线 SMA",
        "entry_volume_fast_window": "入场成交量短期均线", "entry_volume_slow_window": "入场成交量长期均线",
        "early_stop_days": "早期止损日", "stop_pct": "止损幅度 (%)",
        "sma_n": "SMA 周期", "cost_bps": "单边费用 (bps)", "trades": "交易数",
        "win_rate_pct": "合并胜率 (%)", "mean_symbol_win_rate_pct": "平均标的胜率 (%)",
        "median_symbol_win_rate_pct": "中位标的胜率 (%)", "avg_return_pct": "平均单笔收益 (%)",
        "avg_win_pct": "平均盈利 (%)", "avg_loss_pct": "平均亏损 (%)",
        "payoff_ratio": "合并盈亏比", "mean_symbol_payoff_ratio": "平均标的盈亏比",
        "median_symbol_payoff_ratio": "中位标的盈亏比", "max_symbol_payoff_ratio": "最高标的盈亏比",
        "symbols_with_payoff_ratio": "有有效盈亏比标的数", "profit_factor": "盈利因子",
        "median_symbol_cumulative_return_pct": "中位标的累计收益 (%)",
        "mean_symbol_cumulative_return_pct": "平均标的累计收益 (%)",
        "profitable_symbols": "盈利标的数", "avg_days_held": "平均持仓天数",
        "q25_individual_max_drawdown_pct": "回撤25分位数 (%)",
        "passes_15pct_drawdown_proxy": "满足原15%回撤代理",
    }
    frames = []
    for folder in sorted(path for path in GROUP_ROOT.glob("[0-9][0-9]_*") if path.is_dir() and not path.name.startswith("10_")):
        metadata = load_group_preset(str(folder)) or {}
        group_name = metadata.get("group", folder.name.replace("_", "－"))
        for filename, stage in (("search_stage1_results.csv", "第一阶段"),
                                ("search_stage2_results.csv", "第二阶段")):
            path = folder / filename
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path)
            except (OSError, ValueError):
                continue
            frame.insert(0, "策略分组", group_name)
            frame.insert(1, "搜索阶段", stage)
            frames.append(frame.rename(columns=column_names))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_group_preset(preset: dict, token: str) -> None:
    """在控件创建前，把分组最佳组合写入左侧面板的 session state。"""
    apply_parameters_to_controls(preset["best_by_profit_factor"]["parameters"], token)


def switch_to_custom_params() -> None:
    """用户手动修改任一策略参数后，避免界面值与实际运行参数脱节。"""
    data_path = st.session_state.get("active_data_path")
    if data_path:
        st.session_state[f"strategy_version_choice::{Path(data_path).name}"] = "custom"
    st.session_state.pop("applied_preset_token", None)


def initialize_parameter_controls() -> None:
    """只在首次打开时提供控件默认值，避免与 session state 的预设值冲突。"""
    defaults = {
        "band_range": (2.0, 2.5), "entry_lag": 2, "stop_days_text": "3,4",
        "stop_pct": 2.0, "close_stop": False, "sma_n": 5, "cost_bps": 0.0,
        "entry_trend_fast_sma": 5, "entry_trend_slow_sma": 10,
        "entry_volume_fast_window": 5, "entry_volume_slow_window": 20,
        "baseline_lookback": 15, "baseline_max_rise_pct": 20.0,
        "baseline_rsi_period": 14, "baseline_rsi_max": 90,
        "forced_exit_day": 5, "forced_exit_intraday_stop_pct": 1.0, **DEFAULT_CONDITION_STATES,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def collect_results(paths: list[Path], params: dict, capital: float, compound: bool,
                    progress_label: str = "回测中") -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个股票独立运行并整理成逐标的 KPI 与逐笔交易表。"""
    progress = st.progress(0, text=progress_label)
    status = st.empty()
    kpis, trades = [], []
    errors = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        try:
            trade_df, stats = run_backtest(load_ohlc(path), capital=capital, compound=compound, **params)
            kpis.append({"标的": path.stem.upper(), "源文件": str(path), **stats})
            if not trade_df.empty:
                item = trade_df.copy()
                item.insert(0, "symbol", path.stem.upper())
                trades.append(item)
        except Exception as exc:  # 个别异常文件不阻断整个股票池
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
        kpi_frame["cumulative_return_pct"] = (
            kpi_frame["final_equity"] / kpi_frame["initial_capital"] - 1
        ) * 100
    return kpi_frame, (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())


def aggregate_scan(kpis: pd.DataFrame, params: dict) -> dict:
    active = kpis[kpis["n_trades"] > 0]
    total_trades = int(kpis["n_trades"].sum())
    weighted_win_rate = ((kpis["win_rate_pct"].fillna(0) * kpis["n_trades"]).sum() / total_trades
                         if total_trades else np.nan)
    gross_profit = kpis["gross_profit"].fillna(0).sum()
    gross_loss = kpis["gross_loss"].fillna(0).sum()
    return {
        "信号下限": params["band_lo"], "信号上限": params["band_hi"],
        "入场确认": f"t{params['entry_lag']}",
        "早期止损日": ",".join(f"t{day}" for day in params["hard_stop_days"]),
        "止损": params["stop_pct"], "SMA": params["sma_n"],
        "交易数": total_trades, "有交易标的": len(active),
        "胜率_%": weighted_win_rate,
        "平均标的胜率_%": active["win_rate_pct"].mean() if len(active) else np.nan,
        "中位标的胜率_%": active["win_rate_pct"].median() if len(active) else np.nan,
        "中位标的累计收益_%": active["cumulative_return_pct"].median() if len(active) else np.nan,
        "平均标的累计收益_%": active["cumulative_return_pct"].mean() if len(active) else np.nan,
        "盈利因子": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "中位最大回撤_%": active["max_drawdown_pct"].median() if len(active) else np.nan,
        "回撤25分位数_%": active["max_drawdown_pct"].quantile(.25) if len(active) else np.nan,
        "平均每标的交易数": kpis["n_trades"].mean(),
    }


def style_by_drawdown(frame: pd.DataFrame, drawdown_column: str,
                      threshold_pct: float, integer_columns: tuple[str, ...] = ()) -> pd.io.formats.style.Styler:
    """按回撤阈值整行着色，并统一数值显示格式。"""
    def style_row(row: pd.Series) -> list[str]:
        drawdown = row[drawdown_column]
        if pd.isna(drawdown):
            color = ""
        elif drawdown < threshold_pct:
            color = "color: #d62728"
        else:
            color = "color: #198754"
        return [color for _ in row]

    formatters = {
        column: "{:,.0f}" if column in integer_columns else "{:,.2f}"
        for column in frame.select_dtypes(include="number").columns
    }
    return frame.style.apply(style_row, axis=1).format(formatters, na_rep="—")


def kpi_formulas() -> None:
    st.markdown("""
### KPI 计算公式与口径

- **单笔净收益率** = `出场价 ÷ 入场价 − 1 − 2 × 单边手续费(bps) ÷ 10,000`。
- **单笔盈亏金额** = `本笔本金 × 单笔净收益率`；复利模式的本笔本金是当前权益，固定仓位模式始终是初始资金。
- **交易次数** = 实际开仓后已平仓的次数；同一股票持仓期间的新信号会忽略。
- **胜率** = `盈利交易数 ÷ 总交易数 × 100%`；净收益率大于 0 才是盈利，持平计入非盈利。
- **平均单笔收益** = 所有单笔净收益率的算术平均值。
- **平均盈利 / 平均亏损** = 分别对盈利交易、亏损或持平交易的净收益率求平均。
- **盈亏比** = `平均盈利 ÷ |平均亏损|`；无亏损时不显示数值，避免除以零。
- **盈利因子（Profit Factor）** = `所有盈利金额之和 ÷ |所有亏损金额之和|`；大于 1 通常表示历史总盈利金额超过总亏损金额。
- **最终权益**：复利为 `初始资金 × ∏(1 + 每笔净收益率)`；固定仓位为 `初始资金 + 各笔盈亏金额之和`。
- **总盈亏** = `最终权益 − 初始资金`。
- **最大回撤** = `min(权益_t ÷ 截至 t 的历史最高权益 − 1) × 100%`。数值越接近 0，回撤越小。
- **平均持仓天数** = 所有交易从入场日到出场日的交易日间隔的平均值。

所有标的均独立以相同初始资金运行。因此跨标的总和仅用于横向比较，**不是**资金可在不同股票间自由调配的组合回测。
""")


st.title("Momentum Breakout 参数回测")
st.caption("策略条件已原子化：每条 t0、入场与出场规则均可独立启停；确认窗口只考察 tN-1、tN，不使用未来数据。")

with st.sidebar:
    st.header("单组策略参数")
    st.subheader("数据与预设")
    scope_options = available_data_scopes()
    selected_scope = st.selectbox(
        "数据范围", list(scope_options), index=0, key="data_scope",
        help="切换分组会清除上一组的预设选择与回测结果，不会自动运行回测。",
    )
    if selected_scope == "自定义路径":
        data_path = st.text_input("数据目录或单个 CSV", "data")
    else:
        data_path = scope_options[selected_scope]
        st.caption(f"当前目录：`{data_path}`")
    # 不在 selectbox 回调里操作 session state：回调和控件重建交错时会令
    # Streamlit 的前端丢失本次 rerun。这里在当前选择已确定后再做一次性清理。
    if st.session_state.get("active_data_path") != data_path:
        reset_group_session_state()
        st.session_state["active_data_path"] = data_path
    group_preset = load_group_preset(data_path)
    versioned_presets = group_preset_versions(group_preset) if group_preset else {}
    selected_preset = None
    if versioned_presets:
        selected_version = st.selectbox(
            "参数预设版本", ["custom", *versioned_presets],
            key=f"strategy_version_choice::{Path(data_path).name}",
            format_func=lambda value: "自定义参数" if value == "custom" else versioned_presets[value].get("label", value),
            help="每个版本固定记录启用条件、搜索域、生成时间与本组盈利因子最高组合。",
        )
        if selected_version != "custom":
            selected_preset = versioned_presets[selected_version]
            preset_token = f"{data_path}:{selected_version}:{selected_preset.get('generated_at', '')}"
            # 仅在新选预设或切换分组时同步；后续 rerun 不覆盖用户尚未触发的交互。
            if st.session_state.get("applied_preset_token") != preset_token:
                apply_group_preset(selected_preset, preset_token)
        else:
            st.session_state.pop("applied_preset_token", None)
    saved_group_combos = load_saved_combos(data_path)
    if saved_combos_path(data_path) is not None:
        st.subheader("本组已保存组合")
        if message := st.session_state.pop("saved_combo_flash", None):
            st.success(message)
        combo_by_id = {item.get("id", ""): item for item in saved_group_combos if item.get("id")}
        saved_combo_id = st.selectbox(
            "选择已保存组合", ["", *combo_by_id],
            key=f"saved_combo_choice::{Path(data_path).name}",
            format_func=lambda value: "请选择" if not value else (
                f"{combo_by_id[value]['name']}（{combo_by_id[value].get('created_at', '')}）"),
        )
        if saved_combo_id:
            selected_combo = combo_by_id[saved_combo_id]
            st.caption("启用条件：" + "、".join(selected_combo.get("active_condition_ids", [])))
            st.button("加载此组合到左侧参数", key="load_saved_combo",
                      on_click=load_saved_combo_to_controls, args=(selected_combo, data_path))
        elif not saved_group_combos:
            st.caption("本组尚未保存手工组合。")
    initialize_parameter_controls()
    file_limit = st.number_input("最多读取多少个标的（0 = 全部）", min_value=0, value=0, step=1)
    st.subheader("风险显示与资金")
    max_drawdown_limit = st.number_input(
        "最大可接受回撤 (%)", min_value=0.1, max_value=100.0,
        value=DEFAULT_MAX_DRAWDOWN_PCT, step=1.0, key="max_drawdown_limit_pct",
        help="以正数填写可承受的最大跌幅，例如 15 代表最大回撤不得低于 -15%。用于表格风险着色，不参与当前参数排名。",
    )
    st.subheader("基准点（t0）")
    st.caption("每个条件均可单独关闭；关闭后不会参与 t0 筛选。")
    use_signal_band = st.toggle("【T0-01】启用：t0 单日涨幅区间", key="use_signal_band", on_change=switch_to_custom_params)
    band_lo_pct, band_hi_pct = st.slider(
        "t0 单日涨幅区间 (%)", 0.1, 15.0, step=0.1,
        key="band_range", on_change=switch_to_custom_params, disabled=not use_signal_band,
    )
    use_baseline_prior_low = st.toggle("【T0-02】启用：回看期最低点不能是 t0", key="use_baseline_prior_low", on_change=switch_to_custom_params)
    use_baseline_max_rise = st.toggle("【T0-03】启用：t0 相对最低点最大涨幅", key="use_baseline_max_rise", on_change=switch_to_custom_params)
    st.caption("以下回看期由上面两条低点条件共用。")
    baseline_lookback = st.number_input("t0 低点回看交易日数", min_value=1, max_value=100,
                                        key="baseline_lookback", on_change=switch_to_custom_params,
                                        disabled=not (use_baseline_prior_low or use_baseline_max_rise))
    baseline_max_rise_pct = st.number_input("t0 相对低点最大涨幅 (%)", min_value=0.1, max_value=100.0,
                                             key="baseline_max_rise_pct", on_change=switch_to_custom_params,
                                             disabled=not use_baseline_max_rise)
    use_baseline_rsi = st.toggle("【T0-04】启用：t0 RSI 上限", key="use_baseline_rsi", on_change=switch_to_custom_params)
    baseline_rsi_period = st.number_input("t0 RSI 周期", min_value=2, max_value=100,
                                          key="baseline_rsi_period", on_change=switch_to_custom_params,
                                          disabled=not use_baseline_rsi)
    baseline_rsi_max = st.number_input("t0 RSI 上限", min_value=1, max_value=100,
                                       key="baseline_rsi_max", on_change=switch_to_custom_params,
                                       disabled=not use_baseline_rsi)
    use_baseline_close_above_fast_sma = st.toggle(
        "【T0-05】启用：t0 收盘价高于快线 SMA", key="use_baseline_close_above_fast_sma",
        on_change=switch_to_custom_params,
    )
    use_baseline_fast_above_slow_sma = st.toggle(
        "【T0-06】启用：t0 快线 SMA 高于慢线 SMA", key="use_baseline_fast_above_slow_sma",
        on_change=switch_to_custom_params,
    )
    st.caption("以下快线周期由上面两条基准点均线条件共用。")
    entry_trend_fast_sma = st.number_input("基准点快线 SMA 周期", min_value=2, max_value=100,
                                            key="entry_trend_fast_sma", on_change=switch_to_custom_params,
                                            disabled=not (use_baseline_close_above_fast_sma or use_baseline_fast_above_slow_sma))
    entry_trend_slow_sma = st.number_input("基准点慢线 SMA 周期", min_value=3, max_value=200,
                                            key="entry_trend_slow_sma", on_change=switch_to_custom_params,
                                            disabled=not use_baseline_fast_above_slow_sma)
    use_baseline_close_above_sma20 = st.toggle(
        "【T0-07】启用：t0 收盘价高于 SMA20", key="use_baseline_close_above_sma20",
        on_change=switch_to_custom_params,
    )
    use_baseline_volume_sma = st.toggle(
        "【T0-08】启用：t0 成交量短均线高于长均线", key="use_baseline_volume_sma",
        on_change=switch_to_custom_params,
    )
    entry_volume_fast_window = st.number_input("基准点成交量短期均线周期", min_value=1, max_value=100,
                                                key="entry_volume_fast_window", on_change=switch_to_custom_params,
                                                disabled=not use_baseline_volume_sma)
    entry_volume_slow_window = st.number_input("基准点成交量长期均线周期", min_value=2, max_value=250,
                                                key="entry_volume_slow_window", on_change=switch_to_custom_params,
                                                disabled=not use_baseline_volume_sma)
    st.subheader("入场点（tN）")
    entry_lag = st.selectbox("入场确认日", [1, 2, 3, 4, 5],
                             format_func=lambda value: f"t{value}", key="entry_lag",
                             on_change=switch_to_custom_params)
    use_entry_close_vs_t0 = st.toggle("【EN-01】启用：tN 收盘价不低于 t0 收盘价", key="use_entry_close_vs_t0", on_change=switch_to_custom_params)
    st.subheader("出场点：早期止损")
    use_early_stop = st.toggle("【EX-01】启用：早期止损", key="use_early_stop", on_change=switch_to_custom_params)
    stop_days_text = st.text_input(
        "早期止损观察日（相对 t0）",
        key="stop_days_text", on_change=switch_to_custom_params, disabled=not use_early_stop,
    )
    stop_pct = st.slider("早期止损幅度 (%)", 0.1, 20.0, step=0.1,
                         key="stop_pct", on_change=switch_to_custom_params, disabled=not use_early_stop)
    close_stop = st.checkbox("早期止损使用收盘价触发（否则盘中低价）",
                             key="close_stop", on_change=switch_to_custom_params, disabled=not use_early_stop)
    st.subheader("出场点：后期趋势")
    use_exit_below_entry = st.toggle("【EX-02】启用：收盘价跌破入场价出场", key="use_exit_below_entry", on_change=switch_to_custom_params)
    use_exit_below_sma = st.toggle("【EX-03】启用：收盘价跌破趋势 SMA 出场", key="use_exit_below_sma", on_change=switch_to_custom_params)
    sma_n = st.number_input("趋势出场 SMA 周期（早期观察窗口结束后启用）", min_value=2, max_value=100,
                            key="sma_n", on_change=switch_to_custom_params, disabled=not use_exit_below_sma)
    use_forced_exit = st.toggle("【EX-04】启用：入场后强制平仓（基准案例）", key="use_forced_exit", on_change=switch_to_custom_params)
    forced_exit_day = st.selectbox(
        "强制平仓日（相对 t0）", list(range(1, 31)),
        format_func=lambda value: f"t{value}", key="forced_exit_day", on_change=switch_to_custom_params,
        disabled=not use_forced_exit,
        help="默认 t5：例如 t2 入场时，默认会在 t5 收盘强制平仓。该日必须晚于入场确认日；若同日早期止损触发，早期止损优先。",
    )
    use_forced_exit_intraday_protection = st.toggle(
        "【EX-05】启用：强制平仓日日内保护", key="use_forced_exit_intraday_protection",
        on_change=switch_to_custom_params, disabled=not use_forced_exit,
    )
    forced_exit_intraday_stop_pct = st.slider(
        "强制平仓日日内保护幅度（相对入场价，%）", 0.1, 20.0, step=0.1,
        key="forced_exit_intraday_stop_pct", on_change=switch_to_custom_params,
        disabled=not (use_forced_exit and use_forced_exit_intraday_protection),
        help="例如 1%：仅在强制平仓当天，盘中任意即时价格触及入场价×99% 时立即出场；日线回测以 Low≤该价判断是否曾触及。否则在该日收盘强制平仓。",
    )
    st.subheader("执行成本与资金模式")
    cost_bps = st.number_input("单边手续费 (bps)", min_value=0.0, step=0.5,
                               key="cost_bps", on_change=switch_to_custom_params)
    capital = st.number_input("每标的初始资金", min_value=100.0, value=10_000.0, step=1_000.0)
    compound = st.toggle("使用复利（默认关闭，固定仓位）", value=False)

try:
    if selected_preset:
        parameters = normalize_parameters(selected_preset["best_by_profit_factor"]["parameters"])
        parameters["entry_trend_filter"] = True
    else:
        hard_stop_days = parse_ints(stop_days_text) if use_early_stop else (entry_lag + 1,)
        if use_early_stop and any(day <= entry_lag for day in hard_stop_days):
            raise ValueError("早期止损日必须晚于入场确认日")
        if use_signal_band and band_lo_pct >= band_hi_pct:
            raise ValueError("信号区间下限必须小于上限")
        if use_baseline_fast_above_slow_sma and entry_trend_fast_sma >= entry_trend_slow_sma:
            raise ValueError("基准点快线 SMA 周期必须小于慢线 SMA 周期")
        if use_baseline_volume_sma and entry_volume_fast_window >= entry_volume_slow_window:
            raise ValueError("基准点成交量短期均线周期必须小于长期均线周期")
        parameters = {
            "band_lo": band_lo_pct / 100, "band_hi": band_hi_pct / 100,
            "entry_lag": entry_lag, "hard_stop_days": hard_stop_days,
            "stop_pct": stop_pct / 100, "sma_n": int(sma_n),
            "entry_trend_fast_sma": int(entry_trend_fast_sma),
            "entry_trend_slow_sma": int(entry_trend_slow_sma),
            "baseline_rsi_period": int(baseline_rsi_period), "baseline_rsi_max": int(baseline_rsi_max),
            "entry_volume_fast_window": int(entry_volume_fast_window),
            "entry_volume_slow_window": int(entry_volume_slow_window),
            "baseline_lookback": int(baseline_lookback), "baseline_max_rise": baseline_max_rise_pct / 100,
            "use_signal_band": use_signal_band,
            "use_baseline_prior_low": use_baseline_prior_low,
            "use_baseline_max_rise": use_baseline_max_rise,
            "use_baseline_rsi": use_baseline_rsi,
            "use_baseline_close_above_fast_sma": use_baseline_close_above_fast_sma,
            "use_baseline_fast_above_slow_sma": use_baseline_fast_above_slow_sma,
            "use_baseline_close_above_sma20": use_baseline_close_above_sma20,
            "use_baseline_volume_sma": use_baseline_volume_sma,
            "use_entry_close_vs_t0": use_entry_close_vs_t0,
            "use_early_stop": use_early_stop,
            "use_exit_below_entry": use_exit_below_entry,
            "use_exit_below_sma": use_exit_below_sma,
            "use_forced_exit": use_forced_exit,
            "forced_exit_day": int(forced_exit_day),
            "use_forced_exit_intraday_protection": use_forced_exit_intraday_protection,
            "forced_exit_intraday_stop_pct": forced_exit_intraday_stop_pct / 100,
            "entry_trend_filter": True, "stop_intraday": not close_stop, "cost_bps": cost_bps,
        }
except (KeyError, ValueError) as exc:
    # 参数输入永远不应让整个 Streamlit 脚本中止；中止会在某些浏览器会话中
    # 只留下空白页。保留提示，并使用安全默认策略让用户仍可继续调整或加载组合。
    st.error(f"参数错误，已临时使用安全默认值：{exc}")
    parameters = normalize_parameters({
        "band_lo": .02, "band_hi": .025, "entry_lag": 2, "hard_stop_days": (3, 4),
        "stop_pct": .02, "sma_n": 5, "cost_bps": 0.0,
    })

st.subheader("当前设置")
st.info(strategy_narrative(parameters))
with st.sidebar:
    if saved_combos_path(data_path) is not None:
        st.subheader("保存当前组合到本组")
        combo_name = st.text_input("组合名称", key="saved_combo_name", placeholder="例如：高波动稳健版 v1")
        if st.button("保存当前组合", key="save_current_combo"):
            try:
                save_combo(data_path, combo_name, parameters)
                st.session_state["saved_combo_flash"] = f"已保存“{combo_name.strip()}”。"
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
if selected_preset:
    best_metrics = selected_preset["best_by_profit_factor"]["metrics"]
    preset_passes_drawdown = best_metrics["q25_individual_max_drawdown_pct"] >= -max_drawdown_limit
    st.warning(
        f"已使用本组预设：盈利因子 {best_metrics['profit_factor']:.3f}；"
        f"独立回撤 25 分位数 {best_metrics['q25_individual_max_drawdown_pct']:.2f}%；"
        f"{'满足' if preset_passes_drawdown else '不满足'}当前 -{max_drawdown_limit:.1f}% 回撤阈值。"
    )
tabs = st.tabs(["单组回测与 Top 100", "参数组合扫描", "全部分组组合", "分组标的分析", "KPI 公式"])

with tabs[0]:
    st.subheader("单组回测")
    if st.button("运行当前参数", type="primary", width="stretch"):
        try:
            paths = input_files(Path(data_path))
            if file_limit:
                paths = paths[:int(file_limit)]
            kpis, trades = collect_results(paths, parameters, capital, compound)
            st.session_state["single_kpis"] = kpis
            st.session_state["single_trades"] = trades
            st.session_state["single_params"] = parameters
            st.session_state["single_mode"] = "复利" if compound else "固定仓位"
        except Exception as exc:
            st.exception(exc)

    kpis = st.session_state.get("single_kpis")
    required_kpi_columns = {"n_trades", "gross_profit", "gross_loss", "cumulative_return_pct", "max_drawdown_pct"}
    if kpis is not None and not kpis.empty and required_kpi_columns <= set(kpis.columns):
        active = kpis[kpis["n_trades"] > 0]
        total_trades = int(kpis["n_trades"].sum())
        gross_profit, gross_loss = kpis["gross_profit"].fillna(0).sum(), kpis["gross_loss"].fillna(0).sum()
        columns = st.columns(6)
        columns[0].metric("标的数", len(kpis))
        columns[1].metric("总交易数", f"{total_trades:,}")
        columns[2].metric("有交易标的", len(active))
        columns[3].metric("平均标的胜率", f"{active['win_rate_pct'].mean():.2f}%" if len(active) else "不适用")
        columns[4].metric("合并盈利因子", f"{gross_profit / gross_loss:.3f}" if gross_loss else "不适用")
        columns[5].metric("中位标的累计收益", f"{active['cumulative_return_pct'].median():.2f}%" if len(active) else "不适用")
        st.caption(
            f"资金模式：{st.session_state['single_mode']}；所有收益率均按每个标的独立账户计算，不汇总为虚假的组合金额。"
        )

        st.subheader("Top 100 标的")
        metric_options = {
            "盈利因子（高→低）": ("profit_factor", False),
            "累计收益率（高→低）": ("cumulative_return_pct", False),
            "胜率（高→低）": ("win_rate_pct", False),
            "盈亏比（高→低）": ("payoff_ratio", False),
            "平均单笔收益（高→低）": ("avg_return_pct", False),
            "最大回撤（低→高，即更接近 0）": ("max_drawdown_pct", False),
        }
        rank_name = st.selectbox("排序指标", list(metric_options))
        min_trades = st.number_input("最少交易次数", min_value=0, value=1, step=1,
                                     help="03 高流动性高波动组在新过滤下单标的交易较少；默认显示所有至少有 1 笔交易的标的。")
        metric, ascending = metric_options[rank_name]
        ranked = kpis[(kpis["n_trades"] >= min_trades) & kpis[metric].notna()].copy()
        ranked = ranked.sort_values(metric, ascending=ascending).head(100)
        display_columns = ["标的", "n_trades", "win_rate_pct", "cumulative_return_pct", "payoff_ratio", "profit_factor",
                           "max_drawdown_pct", "avg_return_pct", "avg_days_held"]
        top_display = ranked[display_columns].rename(columns=KPI_LABELS).reset_index(drop=True)
        st.caption(
            f"绿色行：该标的最大回撤不低于 -{max_drawdown_limit:.1f}%，符合阈值；"
            "红色行：回撤超过阈值。点击任一行可查看该标的的交易 K 线。"
        )
        selection = st.dataframe(
            style_by_drawdown(top_display, "最大回撤 (%)", -max_drawdown_limit,
                               integer_columns=("交易次数",)), hide_index=True, width="stretch",
            column_config={
                column: st.column_config.Column(column, help=KPI_DEFINITIONS.get(column))
                for column in top_display.columns if column in KPI_DEFINITIONS
            },
            on_select="rerun", selection_mode="single-row", key="top_symbol_table",
        )
        selected_rows = selection.selection.rows if selection else []
        if selected_rows:
            selected_symbol = top_display.iloc[selected_rows[0]]["标的"]
            symbol_trades = st.session_state.get("single_trades", pd.DataFrame())
            symbol_trades = symbol_trades[symbol_trades["symbol"] == selected_symbol].reset_index(drop=True)
            if symbol_trades.empty:
                st.info(f"{selected_symbol} 没有可展示的已平仓交易。")
            else:
                st.subheader(f"{selected_symbol}：交易 K 线")
                trade_labels = [
                    f"第 {index + 1} 笔：t0 {row.signal}｜入场 {row.entry}｜出场 {row.exit}｜{row.ret_pct:+.2f}%"
                    for index, row in symbol_trades.iterrows()
                ]
                trade_index = st.selectbox("选择交易段", range(len(symbol_trades)),
                                           format_func=lambda value: trade_labels[value], key="chart_trade_index")
                trade = symbol_trades.iloc[trade_index]
                source_path = kpis.loc[kpis["标的"] == selected_symbol, "源文件"].iloc[0]
                ohlc = load_ohlc(Path(source_path))
                signal_date, entry_date, exit_date = (pd.Timestamp(trade[column]) for column in ("signal", "entry", "exit"))
                signal_position = ohlc.index[ohlc["Date"] == signal_date][0]
                exit_position = ohlc.index[ohlc["Date"] == exit_date][0]
                chart_data = ohlc.iloc[max(0, signal_position - 15):min(len(ohlc), exit_position + 11)]
                markers = [
                    {"Date": str(signal_date.date()), "Price": float(ohlc.loc[ohlc["Date"] == signal_date, "Close"].iloc[0]), "标记": "t0 基准点"},
                    {"Date": str(entry_date.date()), "Price": float(trade["entry_px"]), "标记": "入场"},
                    {"Date": str(exit_date.date()), "Price": float(trade["exit_px"]), "标记": "出场"},
                ]
                candle_data = chart_data.assign(Date=chart_data["Date"].dt.strftime("%Y-%m-%d"))
                price_span = float(chart_data["High"].max() - chart_data["Low"].min())
                label_price = float(chart_data["Low"].min() - max(price_span * 0.06, chart_data["Low"].min() * 0.005))
                holding_days = ohlc.iloc[signal_position:exit_position + 1]
                day_labels = [
                    {"Date": str(row.Date.date()), "LabelPrice": label_price, "持仓日": f"t{offset}"}
                    for offset, row in enumerate(holding_days.itertuples(index=False))
                ]
                # 图表保持在一个常见笔记本屏幕的可视高度内：短交易不会留下
                # 大片空白，长交易仍有足够的垂直空间辨识蜡烛图。
                chart_days = len(chart_data)
                price_height = min(520, max(340, chart_days * 12))
                volume_height = min(120, max(90, round(price_height * 0.23)))
                candle_width = min(12, max(5, round(300 / max(chart_days, 1))))
                price_axis = {"field": "Low", "type": "quantitative", "title": "价格", "scale": {"zero": False, "nice": True}}
                spec = {
                    "title": f"{selected_symbol}｜{trade.signal} → {trade.exit}｜{trade.reason}",
                    "vconcat": [
                        {
                            "height": price_height,
                            "encoding": {"x": {"field": "Date", "type": "temporal", "axis": {"title": None, "labels": False, "ticks": False}}},
                            "layer": [
                                {"mark": {"type": "rule"}, "encoding": {"y": price_axis, "y2": {"field": "High"}}},
                                {"mark": {"type": "bar", "size": candle_width}, "encoding": {
                                    "y": {"field": "Open", "type": "quantitative", "scale": {"zero": False, "nice": True}}, "y2": {"field": "Close"},
                                    "color": {"condition": {"test": "datum.Close >= datum.Open", "value": "#198754"}, "value": "#d62728", "legend": None},
                                    "tooltip": [
                                        {"field": "Date", "type": "temporal", "title": "日期"},
                                        {"field": "Open", "type": "quantitative", "title": "开盘", "format": ".2f"},
                                        {"field": "High", "type": "quantitative", "title": "最高", "format": ".2f"},
                                        {"field": "Low", "type": "quantitative", "title": "最低", "format": ".2f"},
                                        {"field": "Close", "type": "quantitative", "title": "收盘", "format": ".2f"},
                                        {"field": "Volume", "type": "quantitative", "title": "成交量", "format": ",.0f"},
                                    ],
                                }},
                                {"data": {"values": markers}, "mark": {"type": "point", "filled": True, "size": 100}, "encoding": {
                                    "x": {"field": "Date", "type": "temporal"}, "y": {"field": "Price", "type": "quantitative", "scale": {"zero": False, "nice": True}},
                                    "color": {"field": "标记", "type": "nominal", "title": "交易标记"},
                                }},
                                {"data": {"values": markers}, "mark": {"type": "text", "dy": -14}, "encoding": {
                                    "x": {"field": "Date", "type": "temporal"}, "y": {"field": "Price", "type": "quantitative", "scale": {"zero": False, "nice": True}},
                                    "text": {"field": "标记"}, "color": {"field": "标记", "type": "nominal", "legend": None},
                                }},
                                {"data": {"values": day_labels}, "mark": {"type": "text", "fontSize": 10, "baseline": "top", "color": "#4b5563"}, "encoding": {
                                    "x": {"field": "Date", "type": "temporal"}, "y": {"field": "LabelPrice", "type": "quantitative", "scale": {"zero": False, "nice": True}},
                                    "text": {"field": "持仓日"},
                                }},
                            ],
                        },
                        {
                            "height": volume_height,
                            "mark": {"type": "bar", "size": candle_width},
                            "encoding": {
                                "x": {"field": "Date", "type": "temporal", "title": "日期"},
                                "y": {"field": "Volume", "type": "quantitative", "title": "成交量", "scale": {"zero": True, "nice": True}},
                                "color": {"condition": {"test": "datum.Close >= datum.Open", "value": "#198754"}, "value": "#d62728", "legend": None},
                                "tooltip": [
                                    {"field": "Date", "type": "temporal", "title": "日期"},
                                    {"field": "Volume", "type": "quantitative", "title": "成交量", "format": ",.0f"},
                                ],
                            },
                        },
                    ],
                    "resolve": {"scale": {"x": "shared"}},
                    "autosize": {"type": "fit-x", "contains": "padding"},
                }
                st.vega_lite_chart(candle_data, spec, width="stretch", key=f"trade_chart_{selected_symbol}_{trade_index}")
                st.caption("K 线与成交量窗口：t0 前 15 个交易日至出场后 10 个交易日。t0、t1…标示基准点起的交易日；蓝色 = t0 基准点；绿色 = 入场；红色 = 出场；成交量颜色与当日 K 线涨跌一致。")
        with st.expander("指标定义：选择列名查看计算方式"):
            selected_kpi = st.selectbox("指标列", list(KPI_DEFINITIONS), key="top100_kpi_definition")
            st.markdown(f"**{selected_kpi}**：{KPI_DEFINITIONS[selected_kpi]}")
            st.caption("表头悬停也会显示同一说明。Streamlit 的原生数据表不提供表头点击回调，因此在此处提供可点击/选择的指标名称。")
        st.download_button("下载当前 Top 100 CSV", ranked.to_csv(index=False).encode("utf-8-sig"),
                           file_name="top100_backtest.csv", mime="text/csv")
        st.download_button("下载全部逐标的 KPI CSV", kpis.to_csv(index=False).encode("utf-8-sig"),
                           file_name="stock_kpis_backtest.csv", mime="text/csv")
        if not st.session_state["single_trades"].empty:
            st.download_button("下载逐笔交易 CSV", st.session_state["single_trades"].to_csv(index=False).encode("utf-8-sig"),
                               file_name="trades_backtest.csv", mime="text/csv")
    elif kpis is not None:
        st.warning("本次没有生成可展示的标的 KPI。请检查数据范围与参数后重新运行。")

with tabs[1]:
    st.subheader("参数组合扫描")
    st.caption("扫描保持相同策略结构。每个确认日 tN 自动使用 t(N+1)、t(N+2) 作为早期止损日。全 Nasdaq × 多组合可能需要较长时间。")
    scan_ranges = st.text_input("扫描信号区间（百分比，逗号分隔）", "2-2.5, 2-3")
    scan_entries = st.text_input("扫描入场确认日", "1,2,3")
    scan_stops = st.text_input("扫描早期止损幅度（百分比）", "2,3")
    scan_smas = st.text_input("扫描 SMA 周期", "5")
    if st.button("运行参数扫描", width="stretch"):
        try:
            ranges = parse_percent_ranges(scan_ranges)
            entries, stops, smas = parse_ints(scan_entries), parse_percent_list(scan_stops), parse_ints(scan_smas)
            combos = list(product(ranges, entries, stops, smas))
            paths = input_files(Path(data_path))
            if file_limit:
                paths = paths[:int(file_limit)]
            st.warning(f"即将运行 {len(combos)} 个组合 × {len(paths)} 个标的。")
            scan_rows = []
            combo_progress = st.progress(0, text="准备扫描")
            for index, ((lo, hi), entry, stop, sma) in enumerate(combos, start=1):
                scan_params = parameters | {"band_lo": lo, "band_hi": hi, "entry_lag": entry,
                                             "hard_stop_days": (entry + 1, entry + 2),
                                             "stop_pct": stop, "sma_n": sma}
                combo_kpis, _ = collect_results(paths, scan_params, capital, compound,
                                                 f"组合 {index}/{len(combos)}")
                scan_rows.append(aggregate_scan(combo_kpis, scan_params))
                combo_progress.progress(index / len(combos), text=f"已完成组合 {index}/{len(combos)}")
            combo_progress.empty()
            st.session_state["scan_results"] = pd.DataFrame(scan_rows).sort_values(
                ["盈利因子", "中位标的累计收益_%"], ascending=False)
        except Exception as exc:
            st.exception(exc)
    scan_results = st.session_state.get("scan_results")
    if scan_results is not None:
        st.caption(
            f"绿色行：独立标的最大回撤的 25 分位数不低于 -{max_drawdown_limit:.1f}%；"
            "红色行：超过阈值。该阈值目前是风险提示，不参与本表按盈利因子进行的排序。"
        )
        st.dataframe(
            style_by_drawdown(scan_results, "回撤25分位数_%", -max_drawdown_limit,
                               integer_columns=("交易数", "有交易标的", "SMA")),
            hide_index=True, width="stretch",
        )
        st.download_button("下载参数扫描结果 CSV", scan_results.to_csv(index=False).encode("utf-8-sig"),
                           file_name="parameter_scan.csv", mime="text/csv")

with tabs[2]:
    st.subheader("各组当前最佳组合总览")
    best_overview = load_group_best_overview()
    if best_overview.empty:
        st.info("尚未找到各组版本化 metadata。")
    else:
        st.caption("每组仅展示 metadata 当前激活策略版本中、按盈利因子选出的最佳组合。")
        st.dataframe(
            style_by_drawdown(best_overview, "回撤25分位数 (%)", -max_drawdown_limit,
                               integer_columns=("趋势 SMA 周期", "交易数")),
            hide_index=True, width="stretch",
        )
        st.download_button(
            "下载各组当前最佳组合总览 CSV", best_overview.to_csv(index=False).encode("utf-8-sig"),
            file_name="group_best_combo_overview.csv", mime="text/csv",
        )

    st.subheader("全部分组参数组合")
    st.caption("汇总九个成熟分组的第一阶段 18 组与第二阶段 36 组搜索结果，共 486 行；不包含历史不足一年的标的组。")
    all_group_results = load_all_group_combinations()
    if all_group_results.empty:
        st.info("尚未找到分组搜索结果。请先运行 scan_all_group_domains.py。")
    else:
        groups = list(all_group_results["策略分组"].drop_duplicates())
        selected_groups = st.multiselect("策略分组", groups, default=groups, key="all_group_filter")
        selected_stages = st.multiselect(
            "搜索阶段", ["第一阶段", "第二阶段"], default=["第一阶段", "第二阶段"], key="all_stage_filter",
        )
        sort_options = {
            "盈利因子（高→低）": ("盈利因子", False),
            "中位标的累计收益（高→低）": ("中位标的累计收益 (%)", False),
            "回撤25分位数（低风险→高风险）": ("回撤25分位数 (%)", False),
            "中位标的盈亏比（高→低）": ("中位标的盈亏比", False),
            "平均标的盈亏比（高→低）": ("平均标的盈亏比", False),
        }
        sort_label = st.selectbox("排序方式", list(sort_options), key="all_group_sort")
        sort_column, ascending = sort_options[sort_label]
        filtered = all_group_results[
            all_group_results["策略分组"].isin(selected_groups)
            & all_group_results["搜索阶段"].isin(selected_stages)
        ].copy()
        filtered = filtered.sort_values(sort_column, ascending=ascending, na_position="last")
        integer_columns = (
            "SMA 周期", "交易数", "有有效盈亏比标的数", "盈利标的数",
        )
        st.caption(
            f"共 {len(filtered):,} 个组合。绿色行：回撤25分位数不低于 -{max_drawdown_limit:.1f}%；"
            "红色行：超过阈值。平均/中位/最高标的盈亏比均只在有盈利与亏损交易的标的上计算。"
        )
        st.dataframe(
            style_by_drawdown(filtered, "回撤25分位数 (%)", -max_drawdown_limit,
                               integer_columns=integer_columns),
            hide_index=True, width="stretch", height=620,
        )
        st.download_button(
            "下载全部分组组合 CSV", filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="all_group_parameter_combinations.csv", mime="text/csv",
        )

with tabs[3]:
    st.subheader("分组标的质量分析")
    quality_path = Path("reports/group_quality_cohort_summary.csv")
    symbol_quality_path = Path("reports/group_symbol_quality_analysis.csv")
    if not quality_path.is_file() or not symbol_quality_path.is_file():
        st.info("尚未生成分组标的质量分析。运行 analyze_group_symbol_quality.py 后可显示。")
    else:
        cohorts = pd.read_csv(quality_path)
        symbols = pd.read_csv(symbol_quality_path)
        groups = list(cohorts["策略分组"].drop_duplicates())
        selected_quality_group = st.selectbox("选择分析分组", groups, key="quality_group")
        st.caption("仅纳入至少 3 笔交易的标的。特征来自数据末端近 252 日，因此只能用于提出候选规则，不能直接作为历史黑名单。")
        st.dataframe(cohorts[cohorts["策略分组"] == selected_quality_group], hide_index=True, width="stretch")
        detail = symbols[(symbols["策略分组"] == selected_quality_group) & (symbols["交易数"] >= 3)].copy()
        st.dataframe(detail.sort_values(["累计收益率 (%)", "胜率 (%)"], ascending=False), hide_index=True, width="stretch", height=480)
        st.download_button("下载当前分组标的分析 CSV", detail.to_csv(index=False).encode("utf-8-sig"),
                           file_name="group_symbol_quality_analysis.csv", mime="text/csv")

with tabs[4]:
    kpi_formulas()
