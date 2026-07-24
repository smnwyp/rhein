"""Sidebar state lifecycle for parameter controls and saved presets."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rhein.domain.conditions import ATOMIC_TOGGLE_KEYS, DEFAULT_CONDITION_STATES
from rhein.ui.persistence import normalize_parameters


def reset_group_session_state() -> None:
    group_keys = ("applied_preset_token", "single_kpis", "single_trades", "single_params",
                  "single_mode", "chart_trade_index", "top_symbol_table", "saved_combo_name")
    parameter_keys = ("band_range", "entry_lag", "stop_days_text", "stop_pct", "close_stop", "sma_n",
                      "cost_bps", "entry_trend_fast_sma", "entry_trend_slow_sma",
                      "entry_volume_fast_window", "entry_volume_slow_window", "baseline_lookback",
                      "baseline_max_rise_pct", "baseline_rsi_period", "baseline_rsi_max",
                      "forced_exit_day", "forced_exit_intraday_stop_pct", *ATOMIC_TOGGLE_KEYS)
    for key in (*group_keys, *parameter_keys):
        st.session_state.pop(key, None)


def apply_parameters_to_controls(params: dict, token: str) -> None:
    params = normalize_parameters(params)
    st.session_state.update({
        "band_range": (round(params["band_lo"] * 100, 1), round(params["band_hi"] * 100, 1)),
        "entry_lag": int(params["entry_lag"]), "stop_days_text": ",".join(map(str, params["hard_stop_days"])),
        "stop_pct": round(params["stop_pct"] * 100, 1), "close_stop": not bool(params.get("stop_intraday", True)),
        "sma_n": int(params["sma_n"]), "entry_trend_fast_sma": int(params.get("entry_trend_fast_sma", 5)),
        "entry_trend_slow_sma": int(params.get("entry_trend_slow_sma", 10)),
        "entry_volume_fast_window": int(params.get("entry_volume_fast_window", 5)),
        "entry_volume_slow_window": int(params.get("entry_volume_slow_window", 20)),
        "baseline_lookback": int(params.get("baseline_lookback", 15)),
        "baseline_max_rise_pct": float(params.get("baseline_max_rise", .20) * 100),
        "baseline_rsi_period": int(params.get("baseline_rsi_period", 14)),
        "baseline_rsi_max": int(params.get("baseline_rsi_max", 90)), "cost_bps": float(params["cost_bps"]),
        "forced_exit_day": int(params.get("forced_exit_day", params.get("entry_lag", 2) + params.get("forced_exit_days_after_entry", 3))),
        "forced_exit_intraday_stop_pct": float(params.get("forced_exit_intraday_stop_pct", .01) * 100),
        **{key: bool(params.get(key, DEFAULT_CONDITION_STATES[key])) for key in ATOMIC_TOGGLE_KEYS},
        "applied_preset_token": token,
    })


def load_saved_combo_to_controls(combo: dict, data_path: str) -> None:
    apply_parameters_to_controls(combo["parameters"], f"saved:{data_path}:{combo['id']}")
    st.session_state[f"strategy_version_choice::{Path(data_path).name}"] = "custom"


def switch_to_custom_params() -> None:
    data_path = st.session_state.get("active_data_path")
    if data_path:
        st.session_state[f"strategy_version_choice::{Path(data_path).name}"] = "custom"
    st.session_state.pop("applied_preset_token", None)


def initialize_parameter_controls() -> None:
    defaults = {
        "band_range": (2.0, 2.5), "entry_lag": 2, "stop_days_text": "3,4", "stop_pct": 2.0,
        "close_stop": False, "sma_n": 5, "cost_bps": 0.0, "entry_trend_fast_sma": 5,
        "entry_trend_slow_sma": 10, "entry_volume_fast_window": 5, "entry_volume_slow_window": 20,
        "baseline_lookback": 15, "baseline_max_rise_pct": 20.0, "baseline_rsi_period": 14,
        "baseline_rsi_max": 90, "forced_exit_day": 5, "forced_exit_intraday_stop_pct": 1.0,
        **DEFAULT_CONDITION_STATES,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
