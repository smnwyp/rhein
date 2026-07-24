"""CLI profile and parameter-grid configuration parsing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_days(value: str) -> tuple[int, ...]:
    try:
        days = tuple(int(day.strip()) for day in value.split(",") if day.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("止损日必须是逗号分隔的整数，例如 3,4") from exc
    if not days or any(day < 1 for day in days):
        raise argparse.ArgumentTypeError("止损日必须是正整数，例如 3,4")
    return days


def load_profiles(path: str | None, profile_keys: set[str]) -> dict:
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
        unknown = set(values) - profile_keys
        if unknown:
            raise ValueError(f"{symbol} 的未知参数：{', '.join(sorted(unknown))}")
        profile = dict(values)
        if "hard_stop_days" in profile:
            profile["hard_stop_days"] = tuple(profile["hard_stop_days"])
        profiles[symbol.upper()] = profile
    return profiles


def load_sweep_grid(path: str) -> dict:
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
