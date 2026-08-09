"""Build a reproducible current Nasdaq sector/industry snapshot for OHLCV data."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests


NASDAQ_SCREENER_DOWNLOAD = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq&download=true"
NASDAQ_TO_STANDARD_SECTOR = {
    "Basic Materials": "原材料",
    "Consumer Discretionary": "非必需消费",
    "Consumer Staples": "必需消费",
    "Energy": "能源",
    "Finance": "金融",
    "Health Care": "医疗保健",
    "Industrials": "工业",
    "Real Estate": "房地产",
    "Technology": "信息技术",
    "Telecommunications": "通信服务",
    "Utilities": "公用事业",
}
MAP_COLUMNS = ["代码", "行业板块", "Nasdaq板块", "细分行业", "数据源", "分类快照UTC"]


def fetch_nasdaq_screener() -> pd.DataFrame:
    """Fetch the public Nasdaq screener's current full download payload."""
    try:
        response = requests.get(
            NASDAQ_SCREENER_DOWNLOAD,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; rhein-sector-research/1.0)",
                "Accept": "application/json",
                "Origin": "https://www.nasdaq.com",
            },
            timeout=60,
        )
        response.raise_for_status()
        rows = response.json()["data"]["rows"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"无法从 Nasdaq screener 下载行业分类：{error}") from error
    frame = pd.DataFrame(rows)
    required = {"symbol", "sector", "industry"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Nasdaq screener 响应缺少字段：{', '.join(sorted(missing))}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    return frame.loc[:, ["symbol", "sector", "industry"]].drop_duplicates("symbol")


def build_industry_map(universe: pd.DataFrame, screener: pd.DataFrame, snapshot: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join only explicitly classified symbols; every exclusion is diagnosable."""
    merged = universe.merge(screener, left_on="symbol", right_on="symbol", how="left", validate="one_to_one")
    merged["sector"] = merged["sector"].fillna("").astype(str).str.strip()
    merged["industry"] = merged["industry"].fillna("").astype(str).str.strip()
    merged["行业板块"] = merged["sector"].map(NASDAQ_TO_STANDARD_SECTOR)
    mapped = merged[merged["行业板块"].notna()].copy()
    mapped = mapped.rename(columns={"symbol": "代码", "sector": "Nasdaq板块", "industry": "细分行业"})
    mapped["数据源"] = "Nasdaq screener download"
    mapped["分类快照UTC"] = snapshot
    unmapped = merged[merged["行业板块"].isna()].copy()
    unmapped["原因"] = unmapped.apply(
        lambda row: "Nasdaq 当前快照没有该代码" if not row["sector"] else f"Nasdaq 板块未纳入十一大板块映射：{row['sector']}",
        axis=1,
    )
    unmapped = unmapped.rename(columns={"symbol": "代码", "sector": "Nasdaq板块", "industry": "细分行业"})
    return (
        mapped.loc[:, MAP_COLUMNS].sort_values("代码").reset_index(drop=True),
        unmapped.loc[:, ["代码", "Nasdaq板块", "细分行业", "原因"]].sort_values("代码").reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Nasdaq 公开 screener 生成当前行业板块快照")
    parser.add_argument("--data-dir", default="data/nasdaq_10y")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "downloaded_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到日线清单：{manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype=str)
    if "symbol" not in manifest.columns:
        raise ValueError("downloaded_manifest.csv 必须含 symbol")
    universe = manifest.loc[:, ["symbol"]].dropna().drop_duplicates("symbol").copy()
    universe["symbol"] = universe["symbol"].str.upper().str.strip()
    universe = universe.sort_values("symbol")
    snapshot = datetime.now(UTC).replace(microsecond=0).isoformat()
    mapped, unmapped = build_industry_map(universe, fetch_nasdaq_screener(), snapshot)
    output_path = data_dir / "industry_map.csv"
    unmapped_path = data_dir / "industry_map_unmapped.csv"
    metadata_path = data_dir / "industry_map_metadata.json"
    mapped.to_csv(output_path, index=False, encoding="utf-8")
    unmapped.to_csv(unmapped_path, index=False, encoding="utf-8")
    metadata = {
        "generated_at_utc": snapshot,
        "classification_source": NASDAQ_SCREENER_DOWNLOAD,
        "scope": "downloaded_manifest.csv 中有有效 OHLCV 的当前 Nasdaq 代码",
        "universe_count": int(len(universe)),
        "mapped_count": int(len(mapped)),
        "unmapped_count": int(len(unmapped)),
        "sector_standard": "Nasdaq 当前 sector 映射为中文十一大板块；无板块或非十一大板块的代码不纳入 industry_map.csv",
        "historical_limit": "这是当前静态快照；用于历史回测时存在幸存者偏差与行业归属变更偏差。",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output_path}：{len(mapped)} 条映射；未映射 {len(unmapped)} 条见 {unmapped_path}")


if __name__ == "__main__":
    main()
