"""Build and consume a reproducible current A-share industry snapshot.

The OHLCV files intentionally contain no issuer metadata.  This module keeps
that separation: it obtains a current, explicitly named classification
snapshot, joins it to the locally installed OHLCV universe, and writes small
manifest-only groups.  A group never copies or symlinks market data.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .discovery import input_files


EASTMONEY_A_SHARE_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_A_SHARE_URLS = (
    EASTMONEY_A_SHARE_URL,
    "https://17.push2.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
)
EASTMONEY_A_SHARE_FILTER = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
LEGULEGU_SW_OVERVIEW_URL = "https://legulegu.com/stockdata/sw-industry-overview"
LEGULEGU_SW_COMPOSITION_URL = "https://legulegu.com/stockdata/index-composition"
CLASSIFICATION_STANDARD = "申万 2021 一级行业（乐咕乐股当前成分股快照）"
SNAPSHOT_FILENAME = "a_share_industry_map.csv"
SNAPSHOT_METADATA_FILENAME = "a_share_industry_map_metadata.json"
GROUP_ROOT_NAME = "industry_groups"
GROUP_MANIFEST_FILENAME = "group_manifest.csv"
GROUP_METADATA_FILENAME = "group_metadata.json"
MAP_COLUMNS = ("symbol", "代码", "名称", "行业板块", "数据源", "分类快照UTC")
UNMAPPED_COLUMNS = ("symbol", "原因")
_SYMBOL_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")


class AShareIndustryError(ValueError):
    """An industry classification source or snapshot cannot be used safely."""


def normalize_a_share_symbol(value: object) -> str | None:
    """Return the canonical market-prefixed A-share symbol, if valid."""
    symbol = str(value).strip().upper()
    return symbol if _SYMBOL_PATTERN.fullmatch(symbol) else None


def market_prefixed_symbol(code: object, market: object | None = None) -> str | None:
    """Convert an Eastmoney code to the OHLCV filename convention.

    Eastmoney's market field distinguishes Shanghai and Shenzhen, while Beijing
    codes are most reliably identified by their 4/8/9 prefix.  The latter is
    handled first because Eastmoney may represent Beijing under the Shenzhen
    market identifier in a list response.
    """
    digits = str(code).strip()
    if not digits.isdigit() or len(digits) != 6:
        return None
    if digits[0] in {"4", "8", "9"}:
        return f"BJ{digits}"
    try:
        is_shanghai = int(float(str(market).strip())) == 1
    except (TypeError, ValueError):
        is_shanghai = False
    if is_shanghai or digits.startswith(("6", "5")):
        return f"SH{digits}"
    return f"SZ{digits}"


def local_a_share_symbols(data_root: Path) -> list[str]:
    """Discover the installed OHLCV universe, preserving Parquet preference."""
    return [path.stem.upper() for path in input_files(data_root)]


def fetch_eastmoney_a_share_snapshot(*, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch the current Eastmoney one-stock/one-industry list page by page."""
    client = session or requests.Session()
    page_size = 100
    common_params = {
        "pz": str(page_size), "po": "1", "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
        "fid": "f3", "fs": EASTMONEY_A_SHARE_FILTER,
        "fields": "f12,f13,f14,f100",
    }
    headers = {"User-Agent": "rhein-a-share-industry/1.0", "Referer": "https://quote.eastmoney.com/"}

    def get_page(page: int) -> tuple[list[dict[str, object]], int]:
        params = {"pn": str(page), **common_params}
        failures: list[str] = []
        # Numbered push2 hosts are interchangeable mirrors, but an individual
        # host can occasionally close a long pagination run.  Retrying the
        # page on another mirror keeps a generated snapshot all-or-nothing.
        for _attempt in range(3):
            for url in EASTMONEY_A_SHARE_URLS:
                try:
                    response = client.get(url, params=params, headers=headers, timeout=60)
                    response.raise_for_status()
                    payload = response.json()
                    data = payload["data"]
                    rows = data["diff"]
                    if not isinstance(rows, list):
                        raise AShareIndustryError("东方财富 A 股行业快照为空或响应结构已变化。")
                    total = data.get("total", len(rows))
                    if not isinstance(total, int) or total < len(rows):
                        raise AShareIndustryError("东方财富 A 股行业快照总数无效。")
                    return rows, total
                except requests.RequestException as error:
                    failures.append(str(error))
        raise AShareIndustryError(f"东方财富行业快照第 {page} 页请求失败：{failures[-1] if failures else '未知错误'}")

    try:
        rows, total = get_page(1)
        for page in range(2, (total + page_size - 1) // page_size + 1):
            page_rows, _ = get_page(page)
            rows.extend(page_rows)
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        raise AShareIndustryError(f"无法下载东方财富 A 股行业快照：{error}") from error
    if not isinstance(rows, list) or not rows:
        raise AShareIndustryError("东方财富 A 股行业快照为空或响应结构已变化。")
    frame = pd.DataFrame(rows)
    required = {"f12", "f13", "f14", "f100"}
    missing = required - set(frame.columns)
    if missing:
        raise AShareIndustryError(f"东方财富 A 股行业快照缺少字段：{', '.join(sorted(missing))}")
    frame["symbol"] = [market_prefixed_symbol(code, market) for code, market in zip(frame["f12"], frame["f13"], strict=True)]
    frame["行业板块"] = frame["f100"].fillna("").astype(str).str.strip()
    frame["代码"] = frame["f12"].astype(str).str.strip().str.zfill(6)
    frame["名称"] = frame["f14"].fillna("").astype(str).str.strip()
    return frame.loc[frame["symbol"].notna(), ["symbol", "代码", "名称", "行业板块"]].drop_duplicates("symbol")


def _legulegu_response_text(
    client: requests.Session, url: str, *, params: dict[str, str] | None = None, retry_wait_seconds: float = 5.0,
    cache_path: Path | None = None,
) -> str:
    if cache_path is not None and cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")
    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            response = client.get(
                url, params=params, headers={"User-Agent": "Mozilla/5.0 (rhein-a-share-industry/1.0)"}, timeout=60,
            )
            response.raise_for_status()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(retry_wait_seconds)
    raise AShareIndustryError(f"无法下载申万行业快照：{last_error}") from last_error


def fetch_legulegu_sw_level_one_snapshot(
    *, session: requests.Session | None = None, request_delay_seconds: float = 0.0, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the current Shenwan 2021 level-one industry constituents.

    The overview page exposes one ``level1Item`` per first-level industry;
    each composition page provides market-suffixed stock codes.  Keeping the
    source at the first level gives every mapped stock exactly one sector.
    """
    client = session or requests.Session()
    overview = _legulegu_response_text(
        client, LEGULEGU_SW_OVERVIEW_URL, cache_path=(cache_dir / "overview.html" if cache_dir else None),
    )
    industries: list[tuple[str, str]] = []
    for block in re.findall(r'<li class="level1Item">(.*?)</li>', overview, flags=re.DOTALL):
        code_match = re.search(r'<div id="(801\d{3}\.SI)"', block)
        name_match = re.search(r'lg-industries-item-number">\s*([^<(]+?)\s*\(\d+\)', block)
        if code_match and name_match:
            industries.append((code_match.group(1), name_match.group(1).strip()))
    if not industries:
        raise AShareIndustryError("申万行业概览没有可解析的一级行业。")
    rows: list[dict[str, str]] = []
    for position, (industry_code, industry_name) in enumerate(industries):
        cache_path = cache_dir / f"{industry_code}.html" if cache_dir else None
        if position and (cache_path is None or not cache_path.is_file()):
            time.sleep(request_delay_seconds)
        page = _legulegu_response_text(
            client, LEGULEGU_SW_COMPOSITION_URL, params={"industryCode": industry_code},
            retry_wait_seconds=max(request_delay_seconds, 5.0),
            cache_path=cache_path,
        )
        for stock_code, market in re.findall(r'data-stock-code="(\d{6})\.(SH|SZ|BJ)"', page):
            symbol = f"{market}{stock_code}"
            rows.append({"symbol": symbol, "代码": stock_code, "名称": "", "行业板块": industry_name})
    if not rows:
        raise AShareIndustryError("申万一级行业成分股页面没有可解析的股票代码。")
    frame = pd.DataFrame(rows).drop_duplicates()
    conflicts = frame.groupby("symbol")["行业板块"].nunique()
    if (conflicts > 1).any():
        samples = conflicts[conflicts > 1].index[:10].tolist()
        raise AShareIndustryError(f"申万一级行业快照中同一股票归属多个行业：{samples}")
    return frame.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def build_industry_snapshot(
    symbols: Iterable[object], source: pd.DataFrame, *, snapshot_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join an explicit local universe to a fetched industry snapshot."""
    universe = pd.DataFrame({"symbol": [normalize_a_share_symbol(symbol) for symbol in symbols]})
    universe = universe.dropna().drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    required = {"symbol", "代码", "名称", "行业板块"}
    missing = required - set(source.columns)
    if missing:
        raise AShareIndustryError(f"行业数据缺少字段：{', '.join(sorted(missing))}")
    classification = source.loc[:, ["symbol", "代码", "名称", "行业板块"]].copy()
    classification["symbol"] = classification["symbol"].map(normalize_a_share_symbol)
    classification["行业板块"] = classification["行业板块"].fillna("").astype(str).str.strip()
    classification = classification.dropna(subset=["symbol"]).drop_duplicates("symbol")
    joined = universe.merge(classification, on="symbol", how="left", validate="one_to_one")
    mapped = joined[joined["行业板块"].notna() & joined["行业板块"].ne("")].copy()
    mapped["数据源"] = CLASSIFICATION_STANDARD
    mapped["分类快照UTC"] = snapshot_at
    unmapped = joined[joined["行业板块"].isna() | joined["行业板块"].eq("")].loc[:, ["symbol"]].copy()
    unmapped["原因"] = "当前行业快照未包含该代码或行业字段为空"
    return (
        mapped.loc[:, MAP_COLUMNS].sort_values(["行业板块", "symbol"]).reset_index(drop=True),
        unmapped.loc[:, UNMAPPED_COLUMNS].sort_values("symbol").reset_index(drop=True),
    )


def _group_folder_name(position: int, industry: str) -> str:
    safe_name = re.sub(r"[\\/:*?\"<>|]", "_", industry).strip(" .")
    if not safe_name:
        raise AShareIndustryError("行业名称为空，无法创建分组目录。")
    return f"{position:02d}_{safe_name}"


def write_industry_groups(*, data_root: Path, mapped: pd.DataFrame, unmapped: pd.DataFrame, snapshot_at: str) -> list[Path]:
    """Persist a map and manifest-only industry groups beside A-share OHLCV."""
    data_root.mkdir(parents=True, exist_ok=True)
    mapped.loc[:, MAP_COLUMNS].to_csv(data_root / SNAPSHOT_FILENAME, index=False, encoding="utf-8")
    unmapped.loc[:, UNMAPPED_COLUMNS].to_csv(data_root / "a_share_industry_map_unmapped.csv", index=False, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "classification_standard": CLASSIFICATION_STANDARD,
        "source": LEGULEGU_SW_OVERVIEW_URL,
        "generated_at_utc": snapshot_at,
        "mapped_count": int(len(mapped)),
        "unmapped_count": int(len(unmapped)),
        "industry_count": int(mapped["行业板块"].nunique()),
        "historical_limit": "当前静态行业快照用于历史回测时，存在幸存者偏差和行业归属变更偏差。",
    }
    (data_root / SNAPSHOT_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    # Rewrite manifests in place so a refresh never deletes a user's saved
    # parameter combinations stored beside an existing industry group.  Stale
    # generator folders are harmless: the UI accepts only metadata matching
    # the current classification standard.
    group_root = data_root / GROUP_ROOT_NAME
    group_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for position, (industry, sector) in enumerate(mapped.groupby("行业板块", sort=True), start=1):
        folder = group_root / _group_folder_name(position, str(industry))
        folder.mkdir(parents=True, exist_ok=True)
        sector.loc[:, ["symbol", "代码", "名称"]].sort_values("symbol").to_csv(
            folder / GROUP_MANIFEST_FILENAME, index=False, encoding="utf-8",
        )
        group_metadata = {
            "schema_version": 1,
            "group_type": "a_share_industry",
            "industry": str(industry),
            "source_data_root": "../..",
            "classification_source": metadata["classification_standard"],
            "snapshot_at_utc": snapshot_at,
            "symbol_count": int(len(sector)),
        }
        (folder / GROUP_METADATA_FILENAME).write_text(
            json.dumps(group_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        written.append(folder)
    return written


def build_and_write_current_industry_groups(*, data_root: Path) -> dict[str, int]:
    """Fetch, join, and materialize groups for the installed OHLCV universe."""
    snapshot_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    mapped, unmapped = build_industry_snapshot(
        local_a_share_symbols(data_root),
        fetch_legulegu_sw_level_one_snapshot(
            request_delay_seconds=float(os.getenv("A_SHARE_INDUSTRY_REQUEST_DELAY_SECONDS", "2.5")),
            cache_dir=data_root / ".a_share_industry_download_cache",
        ),
        snapshot_at=snapshot_at,
    )
    groups = write_industry_groups(data_root=data_root, mapped=mapped, unmapped=unmapped, snapshot_at=snapshot_at)
    shutil.rmtree(data_root / ".a_share_industry_download_cache", ignore_errors=True)
    return {"mapped": len(mapped), "unmapped": len(unmapped), "groups": len(groups)}
