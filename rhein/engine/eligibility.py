"""Validation and t0/tN eligibility rules for the strategy engine."""
from __future__ import annotations

import numpy as np

MA20_RISE_MIN_PCT = 0.0001


def validate_run_parameters(params: dict) -> int:
    """Validate run-time parameters and return the normalized forced-exit day."""
    if (params["use_baseline_close_above_fast_sma"] or params["use_baseline_fast_above_slow_sma"]) and params["entry_trend_fast_sma"] < 2:
        raise ValueError("基准点快线 MA 周期至少为 2")
    if params["use_baseline_fast_above_slow_sma"] and params["entry_trend_slow_sma"] <= params["entry_trend_fast_sma"]:
        raise ValueError("基准点 MA 必须满足 2 ≤ 快线周期 < 慢线周期")
    if params["use_baseline_volume_sma"] and (params["entry_volume_fast_window"] < 1 or params["entry_volume_slow_window"] <= params["entry_volume_fast_window"]):
        raise ValueError("基准点成交量均线必须满足 1 ≤ 短期周期 < 长期周期")
    if (params["use_baseline_prior_low"] or params["use_baseline_max_rise"]) and params["baseline_lookback"] < 1:
        raise ValueError("基准点回看窗口至少为 1")
    if params["use_baseline_rsi"] and params["baseline_rsi_period"] < 2:
        raise ValueError("基准点回看窗口至少为 1，RSI 周期至少为 2")
    forced_exit_day = params["forced_exit_day"]
    if params["use_forced_exit"] and forced_exit_day <= params["entry_lag"]:
        raise ValueError("强制平仓日必须晚于入场确认日")
    if params["use_forced_exit"] and params["use_forced_exit_intraday_protection"] and params["forced_exit_intraday_stop_pct"] <= 0:
        raise ValueError("强制平仓日内保护幅度必须大于 0")
    return forced_exit_day


def baseline_is_eligible(i: int, *, close, open_, ret1, rsi, entry_fast_sma, entry_slow_sma,
                         baseline_sma20, vol_fast_sma, vol_slow_sma, params: dict) -> bool:
    """Return whether a candidate date satisfies every enabled t0 condition."""
    if params["use_signal_band"] and not params["band_lo"] <= ret1[i] <= params["band_hi"]:
        return False
    needs_window = params["use_baseline_prior_low"] or params["use_baseline_max_rise"]
    if needs_window and i < params["baseline_lookback"]:
        return False
    if params["use_baseline_rsi"] and (np.isnan(rsi[i]) or rsi[i] > params["baseline_rsi_max"]):
        return False
    if params["use_baseline_close_above_fast_sma"] and (np.isnan(entry_fast_sma[i]) or close[i] <= entry_fast_sma[i]):
        return False
    if params["use_baseline_fast_above_slow_sma"] and (np.isnan(entry_fast_sma[i]) or np.isnan(entry_slow_sma[i]) or entry_fast_sma[i] <= entry_slow_sma[i]):
        return False
    if params["use_baseline_close_above_sma20"] and (np.isnan(baseline_sma20[i]) or close[i] <= baseline_sma20[i]):
        return False
    if params["use_baseline_volume_sma"] and (np.isnan(vol_fast_sma[i]) or np.isnan(vol_slow_sma[i]) or vol_fast_sma[i] <= vol_slow_sma[i]):
        return False
    # t0 must be a bullish candle. A doji (Close == Open) is deliberately not bullish.
    if params["use_baseline_bullish_candle"] and close[i] <= open_[i]:
        return False
    if params["use_baseline_sma20_rising"]:
        if i < 2 or np.isnan(baseline_sma20[i]) or np.isnan(baseline_sma20[i - 2]):
            return False
        if baseline_sma20[i] < baseline_sma20[i - 2] * (1 + MA20_RISE_MIN_PCT):
            return False
    if needs_window:
        window = close[i - params["baseline_lookback"]:i + 1]
        tmin_rel = int(np.argmin(window))
        if params["use_baseline_prior_low"] and tmin_rel == len(window) - 1:
            return False
        if params["use_baseline_max_rise"] and close[i] > window[tmin_rel] * (1 + params["baseline_max_rise"]):
            return False
    return True
