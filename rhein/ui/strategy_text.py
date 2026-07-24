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
