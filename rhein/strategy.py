"""Strategy parameter parsing, condition identifiers, and human-readable rules."""
from __future__ import annotations

ATOMIC_TOGGLE_KEYS = (
    "use_signal_band", "use_baseline_prior_low", "use_baseline_max_rise", "use_baseline_rsi",
    "use_entry_close_vs_t0", "use_entry_close_above_fast_sma", "use_entry_fast_above_slow_sma",
    "use_entry_volume_sma", "use_early_stop", "use_exit_below_entry", "use_exit_below_sma",
    "use_forced_exit", "use_forced_exit_intraday_protection",
)
DEFAULT_CONDITION_STATES = {key: True for key in ATOMIC_TOGGLE_KEYS} | {"use_forced_exit": False}

CONDITION_IDS = {
    "use_signal_band": "T0-01", "use_baseline_prior_low": "T0-02",
    "use_baseline_max_rise": "T0-03", "use_baseline_rsi": "T0-04",
    "use_entry_close_vs_t0": "EN-01", "use_entry_close_above_fast_sma": "EN-02",
    "use_entry_fast_above_slow_sma": "EN-03", "use_entry_volume_sma": "EN-04",
    "use_early_stop": "EX-01", "use_exit_below_entry": "EX-02",
    "use_exit_below_sma": "EX-03", "use_forced_exit": "EX-04",
    "use_forced_exit_intraday_protection": "EX-05",
}


def parse_ints(value: str) -> tuple[int, ...]:
    days = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not days or any(day < 1 for day in days):
        raise ValueError("请输入正整数，例如 3,4")
    return days


def parse_percent_ranges(value: str) -> list[tuple[float, float]]:
    """把 '2-2.5, 2-3' 解析为小数比例。"""
    ranges = []
    for part in value.split(","):
        low, high = (float(number.strip()) / 100 for number in part.strip().split("-"))
        if not 0 < low < high:
            raise ValueError("每个幅度区间必须满足下限 < 上限，例如 2-2.5")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("请至少输入一个幅度区间")
    return ranges


def parse_percent_list(value: str) -> list[float]:
    values = [float(part.strip()) / 100 for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("请输入正百分比，例如 2,3")
    return values


def active_condition_ids(params: dict) -> list[str]:
    return [condition_id for key, condition_id in CONDITION_IDS.items() if params.get(key, True)]


def params_to_text(params: dict) -> str:
    on = lambda key: params.get(key, True)
    rules = []
    if on("use_signal_band"): rules.append(f"信号 {params['band_lo']:.2%}–{params['band_hi']:.2%}")
    if on("use_baseline_prior_low"): rules.append(f"{params['baseline_lookback']}日低点不在t0")
    if on("use_baseline_max_rise"): rules.append(f"相对低点延伸≤{params['baseline_max_rise']:.0%}")
    if on("use_baseline_rsi"): rules.append(f"RSI{params['baseline_rsi_period']}≤{params['baseline_rsi_max']}")
    if on("use_entry_close_vs_t0"): rules.append("tN收盘≥t0")
    entry = []
    if on("use_entry_close_above_fast_sma"): entry.append(f"收盘>SMA{params['entry_trend_fast_sma']}")
    if on("use_entry_fast_above_slow_sma"): entry.append(f"SMA{params['entry_trend_fast_sma']}>SMA{params['entry_trend_slow_sma']}")
    if on("use_entry_volume_sma"): entry.append(f"量SMA{params['entry_volume_fast_window']}>量SMA{params['entry_volume_slow_window']}")
    if entry: rules.append("tN-1/tN任一天：" + "且".join(entry))
    if on("use_early_stop"): rules.append(f"早期止损t{','.join(map(str, params['hard_stop_days']))} / {params['stop_pct']:.2%}")
    exits = []
    if on("use_exit_below_entry"): exits.append("跌破入场价")
    if on("use_exit_below_sma"): exits.append(f"跌破SMA{params['sma_n']}")
    if on("use_forced_exit"):
        forced = f"t{params.get('forced_exit_day', 5)}强制平仓"
        if on("use_forced_exit_intraday_protection"):
            forced += f"（日内跌{params.get('forced_exit_intraday_stop_pct', .01):.1%}先卖）"
        exits.append(forced)
    if exits: rules.append("后期出场：" + "或".join(exits))
    return "；".join(rules) + f"；单边费用 {params['cost_bps']:.1f} bps"


def strategy_narrative(params: dict) -> str:
    """将当前参数整理为可直接阅读的策略规则。"""
    on = lambda key: params.get(key, True)
    stop_days = "、".join(f"t{day}" for day in params["hard_stop_days"])
    after_day = max(params["hard_stop_days"]) + 1 if on("use_early_stop") else params["entry_lag"] + 1
    t0, entry = [], []
    if on("use_signal_band"): t0.append(f"单日涨幅在 {params['band_lo']:.1%}–{params['band_hi']:.1%}")
    if on("use_baseline_prior_low"): t0.append(f"t0-{params['baseline_lookback']} 至t0低点不在t0")
    if on("use_baseline_max_rise"): t0.append(f"相对该低点涨幅不超过{params['baseline_max_rise']:.0%}")
    if on("use_baseline_rsi"): t0.append(f"RSI({params['baseline_rsi_period']})不高于{params['baseline_rsi_max']}")
    if on("use_entry_close_vs_t0"): entry.append("tN收盘不低于t0")
    if on("use_entry_close_above_fast_sma"): entry.append(f"收盘>SMA{params['entry_trend_fast_sma']}")
    if on("use_entry_fast_above_slow_sma"): entry.append(f"SMA{params['entry_trend_fast_sma']}>SMA{params['entry_trend_slow_sma']}")
    if on("use_entry_volume_sma"): entry.append(f"量SMA{params['entry_volume_fast_window']}>量SMA{params['entry_volume_slow_window']}")
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
            paragraphs.append(f"后期技术出场：原规则从 t{after_day} 起，收盘{'或'.join(exits)}时出场；但 EX-04 会在 t{forced_exit_day} 强制平仓并优先执行，因此本组合下该规则不会实际触发。")
        elif on("use_forced_exit"):
            paragraphs.append(f"后期技术出场：从 t{after_day} 至 t{forced_exit_day - 1}，收盘{'或'.join(exits)}时出场。")
        else:
            paragraphs.append(f"后期技术出场：从 t{after_day} 起，收盘{'或'.join(exits)}时出场。")
    else:
        paragraphs.append(f"后期技术出场：从 t{after_day} 起不设后期技术出场。")
    if on("use_forced_exit"):
        forced = f"强制平仓：t{forced_exit_day} 当天必须平仓。"
        if on("use_forced_exit_intraday_protection"):
            forced = f"强制平仓：t{forced_exit_day} 当天，若盘中任意即时价格触及入场价下{params.get('forced_exit_intraday_stop_pct', .01):.1%}，立即按保护规则出场；否则以 t{forced_exit_day} 收盘价强制平仓。"
        paragraphs.append(forced)
    return "\n\n".join(paragraphs)
