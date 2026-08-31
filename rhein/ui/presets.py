"""Read-only helpers for group metadata and versioned strategy presets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rhein.data.a_share_industry import (
    CLASSIFICATION_STANDARD,
    GROUP_MANIFEST_FILENAME,
    GROUP_METADATA_FILENAME,
    GROUP_ROOT_NAME,
    SNAPSHOT_FILENAME,
)
from rhein.paths import A_SHARE_ROOT, DATA_ROOT, GROUP_ROOT, NASDAQ_ROOT


def available_data_scopes() -> dict[str, str]:
    """Return only data ranges that are actually installed on this runtime."""
    scopes: dict[str, str] = {}
    if (DATA_ROOT / "nvda.csv").is_file() and (DATA_ROOT / "tesla.csv").is_file():
        scopes["示例数据（NVDA、TSLA）"] = str(DATA_ROOT)
    if NASDAQ_ROOT.is_dir() and any(NASDAQ_ROOT.glob("*.csv")):
        scopes["全部 Nasdaq 当前股票池"] = str(NASDAQ_ROOT)
    if A_SHARE_ROOT.is_dir():
        # A release can contain CSV, Parquet, or both during migration. Count
        # symbols rather than files so the UI neither loses a Parquet-only
        # universe nor double-counts the same symbol in both formats.
        a_share_count = len(
            {
                path.stem.upper()
                for pattern in ("*.parquet", "*.csv")
                for path in A_SHARE_ROOT.glob(pattern)
                if path.name not in {"conversion_failures.csv", SNAPSHOT_FILENAME, "a_share_industry_map_unmapped.csv"}
            }
        )
        scopes[f"全部 A 股（{a_share_count} 个可回测标的，前复权日线）"] = str(A_SHARE_ROOT)
        industry_root = A_SHARE_ROOT / GROUP_ROOT_NAME
        if industry_root.is_dir():
            for folder in sorted(path for path in industry_root.iterdir() if path.is_dir()):
                manifest = folder / GROUP_MANIFEST_FILENAME
                metadata_path = folder / GROUP_METADATA_FILENAME
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    industry = str(metadata["industry"]).strip()
                    if metadata.get("classification_source") != CLASSIFICATION_STANDARD:
                        continue
                    count = len(pd.read_csv(manifest, usecols=["symbol"]))
                except (FileNotFoundError, KeyError, OSError, ValueError):
                    continue
                if industry and count:
                    scopes[f"A 股行业 · {industry}（{count} 个标的）"] = str(folder)
    if NASDAQ_ROOT.is_dir() and GROUP_ROOT.is_dir():
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
