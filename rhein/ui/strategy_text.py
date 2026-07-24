"""UI wording for the currently selected strategy configuration."""


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
    """The UI-specific natural-language description of the current strategy."""
    stop_days = "、".join(f"t{day}" for day in params["hard_stop_days"])
    after_day = max(params["hard_stop_days"]) + 1 if params.get("use_early_stop", True) else params["entry_lag"] + 1
    on = lambda key: params.get(key, True)
    t0, entry = [], []
    if on("use_signal_band"): t0.append(f"单日涨幅在 {params['band_lo']:.1%}–{params['band_hi']:.1%}")
    if on("use_baseline_prior_low"): t0.append(f"t0-{params['baseline_lookback']} 至t0低点不在t0")
    if on("use_baseline_max_rise"): t0.append(f"相对该低点涨幅不超过{params['baseline_max_rise']:.0%}")
    if on("use_baseline_rsi"): t0.append(f"RSI({params['baseline_rsi_period']})不高于{params['baseline_rsi_max']}")
    if on("use_baseline_close_above_fast_sma"): t0.append(f"收盘>SMA{params['entry_trend_fast_sma']}")
    if on("use_baseline_fast_above_slow_sma"): t0.append(f"SMA{params['entry_trend_fast_sma']}>SMA{params['entry_trend_slow_sma']}")
    if on("use_baseline_close_above_sma20"): t0.append("收盘>SMA20")
    if on("use_baseline_volume_sma"): t0.append(f"量SMA{params['entry_volume_fast_window']}>量SMA{params['entry_volume_slow_window']}")
    if on("use_entry_close_vs_t0"): entry.append("tN收盘不低于t0")
    paragraphs = [f"基准点：t0 需满足“{'、'.join(t0) if t0 else '无基准筛选'}”。",
                  f"入场点：在 t{params['entry_lag']}，需满足“{'、'.join(entry) if entry else '无入场确认筛选'}”后按收盘价入场。"]
    if on("use_early_stop"):
        if params.get("stop_intraday", True):
            paragraphs.append(
                f"早期出场：在 {stop_days}，盘中任意即时价格触及 {params['stop_pct']:.1%} 止损幅度即出场。"
            )
        else:
            paragraphs.append(
                f"早期出场：在 {stop_days}，收盘价触及 {params['stop_pct']:.1%} 止损幅度即按收盘价出场。"
            )
    exits = (["跌破入场价"] if on("use_exit_below_entry") else []) + ([f"跌破SMA{params['sma_n']}"] if on("use_exit_below_sma") else [])
    forced_exit_day = params.get("forced_exit_day", 5)
    if exits:
        if on("use_forced_exit") and forced_exit_day <= after_day:
            paragraphs.append(f"后期技术出场：原规则从 t{after_day} 起，收盘{'或'.join(exits)}时出场；但 EX-04 会在 t{forced_exit_day} 强制平仓并优先执行，因此本组合下该规则不会实际触发。")
        elif on("use_forced_exit"):
            paragraphs.append(f"后期技术出场：从 t{after_day} 至 t{forced_exit_day - 1}，收盘{'或'.join(exits)}时出场。")
        else: paragraphs.append(f"后期技术出场：从 t{after_day} 起，收盘{'或'.join(exits)}时出场。")
    else: paragraphs.append(f"后期技术出场：从 t{after_day} 起不设后期技术出场。")
    if on("use_forced_exit"):
        text = f"强制平仓：t{forced_exit_day} 当天必须平仓。"
        if on("use_forced_exit_intraday_protection"):
            text = f"强制平仓：t{forced_exit_day} 当天，若盘中任意即时价格触及入场价下{params.get('forced_exit_intraday_stop_pct', .01):.1%}，立即按保护规则出场；否则以 t{forced_exit_day} 收盘价强制平仓。"
        paragraphs.append(text)
    return "\n\n".join(paragraphs)
