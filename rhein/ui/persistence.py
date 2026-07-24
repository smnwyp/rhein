"""Persistence and backwards-compatible normalization for group combinations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rhein.domain.conditions import DEFAULT_CONDITION_STATES, active_condition_ids


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
    """补齐旧 metadata 中不存在的策略字段，作为唯一兼容边界。"""
    params = dict(raw_params)
    legacy_flags = {
        "use_entry_close_above_fast_sma": "use_baseline_close_above_fast_sma",
        "use_entry_fast_above_slow_sma": "use_baseline_fast_above_slow_sma",
        "use_entry_volume_sma": "use_baseline_volume_sma",
    }
    for old_key, new_key in legacy_flags.items():
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
    params.setdefault("forced_exit_day", params["entry_lag"] + params.get("forced_exit_days_after_entry", 3))
    for toggle_key, default in DEFAULT_CONDITION_STATES.items():
        params.setdefault(toggle_key, default)
    return params


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
    serializable = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
    now = datetime.now().isoformat(timespec="seconds")
    combos.append({
        "id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "name": clean_name, "created_at": now,
        "active_condition_ids": active_condition_ids(params), "parameters": serializable,
    })
    path.write_text(json.dumps({"schema_version": 1, "group_folder": Path(data_path).name,
                                "updated_at": now, "combos": combos}, ensure_ascii=False, indent=2), encoding="utf-8")
