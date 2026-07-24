"""Read-only helpers for group metadata and versioned strategy presets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rhein.paths import DATA_ROOT, GROUP_ROOT


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
            scopes[f"{folder.name.replace('_', ' ')}（{count} 个标的）"] = str(folder)
    scopes["自定义路径"] = ""
    return scopes


def load_group_preset(data_path: str) -> dict | None:
    """读取分组搜索完成后写入的最佳组合；非分组目录返回 None。"""
    metadata_path = Path(data_path) / "search_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
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


def best_combo_for_record(record: dict, fallback: dict | None = None) -> dict:
    """取得任一版本的最佳组合，兼容旧版“盈利因子最高”metadata。"""
    for source in (record, fallback or {}):
        for key in ("best_combo", "best_by_median_symbol_return", "best_by_profit_factor"):
            candidate = source.get(key)
            if isinstance(candidate, dict) and candidate.get("parameters"):
                return candidate
    return {}
