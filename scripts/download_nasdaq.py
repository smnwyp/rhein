"""下载当前 Nasdaq 普通股的近十年日线数据，可中断后继续执行。"""
import argparse
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq"
EXCLUDED_NAME = re.compile(r"ETF|FUND|TRUST|WARRANT|RIGHT|UNIT|NOTES|PREFERRED|DEPOSITARY|BOND|ETN", re.I)
COMMON_STOCK_NAME = re.compile(r"COMMON STOCK|ORDINARY SHARES", re.I)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch_universe() -> pd.DataFrame:
    response = requests.get(NASDAQ_SCREENER, headers={
        "User-Agent": "Mozilla/5.0 (compatible; strategy-research/1.0)",
        "Accept": "application/json", "Origin": "https://www.nasdaq.com",
    }, timeout=30)
    response.raise_for_status()
    rows = response.json()["data"]["table"]["rows"]
    universe = pd.DataFrame(rows)
    selected = universe[
        universe["name"].str.contains(COMMON_STOCK_NAME, na=False)
        & ~universe["name"].str.contains(EXCLUDED_NAME, na=False)
        & universe["symbol"].str.fullmatch(r"[A-Z.]+", na=False)
    ].copy()
    selected["yahoo_symbol"] = selected["symbol"].str.replace(".", "-", regex=False)
    return selected.sort_values("symbol").reset_index(drop=True)


def extract_ticker(download: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        if yahoo_symbol in download.columns.get_level_values(0):
            frame = download[yahoo_symbol].copy()
        elif yahoo_symbol in download.columns.get_level_values(-1):
            frame = download.xs(yahoo_symbol, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = download.copy()
    available = [col for col in OHLCV if col in frame.columns]
    if set(OHLCV) - set(available):
        return pd.DataFrame()
    frame = frame[OHLCV].dropna(subset=["Close"])
    if frame.empty:
        return frame
    frame.index.name = "Date"
    return frame.reset_index()


def build_manifest(output_dir: Path, universe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """核验所有已存在数据文件，并生成覆盖整个股票池的合并 manifest。"""
    manifest, invalid = [], []
    for row in universe.itertuples(index=False):
        path = output_dir / f"{row.symbol}.csv"
        if not path.exists():
            invalid.append({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol, "reason": "文件不存在"})
            continue
        try:
            frame = pd.read_csv(path, usecols=["Date", *OHLCV])
            if frame.empty or set(["Date", *OHLCV]) - set(frame.columns):
                raise ValueError("缺少 OHLCV 字段或数据为空")
            manifest.append({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol,
                             "rows": len(frame), "first_date": str(frame["Date"].iloc[0])[:10],
                             "last_date": str(frame["Date"].iloc[-1])[:10]})
        except Exception as exc:
            invalid.append({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol, "reason": str(exc)})
    return pd.DataFrame(manifest), invalid


def main() -> None:
    ap = argparse.ArgumentParser(description="下载当前 Nasdaq 普通股近十年日线 OHLCV 数据")
    ap.add_argument("--output-dir", default="data/nasdaq_10y")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--pause", type=float, default=0.8, help="批次间暂停秒数，降低数据源限流风险")
    ap.add_argument("--limit", type=int, help="仅下载前 N 个股票，用于测试")
    ap.add_argument("--refresh", action="store_true", help="重新下载已存在的文件")
    args = ap.parse_args()
    if args.years < 1 or args.batch_size < 1:
        raise ValueError("years 和 batch-size 必须为正数")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = fetch_universe()
    universe.to_csv(output_dir / "universe_current_nasdaq_common_stocks.csv", index=False)
    if args.limit:
        universe = universe.head(args.limit)
    start = date.today() - timedelta(days=365 * args.years + 3)
    end = date.today() + timedelta(days=1)  # Yahoo 的 end 为排他日期
    todo, existing = [], 0
    for row in universe.itertuples(index=False):
        path = output_dir / f"{row.symbol}.csv"
        if path.exists() and not args.refresh:
            existing += 1
        else:
            todo.append(row)
    print(f"股票池：{len(universe)}；已有文件跳过：{existing}；本次待下载：{len(todo)}")

    completed, failed = [], []
    for offset in range(0, len(todo), args.batch_size):
        batch = todo[offset:offset + args.batch_size]
        symbols = [row.yahoo_symbol for row in batch]
        try:
            downloaded = yf.download(symbols, start=start.isoformat(), end=end.isoformat(),
                                     group_by="ticker", auto_adjust=False, progress=False,
                                     threads=True)
        except Exception as exc:  # 网络/限流错误：记录后继续，下一次运行可重试
            failed.extend({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol,
                           "reason": str(exc)} for row in batch)
            continue
        for row in batch:
            frame = extract_ticker(downloaded, row.yahoo_symbol)
            if frame.empty:
                failed.append({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol,
                               "reason": "未取得有效 OHLCV 数据"})
                continue
            frame.to_csv(output_dir / f"{row.symbol}.csv", index=False)
            completed.append({"symbol": row.symbol, "yahoo_symbol": row.yahoo_symbol,
                              "rows": len(frame), "first_date": str(frame["Date"].iloc[0])[:10],
                              "last_date": str(frame["Date"].iloc[-1])[:10]})
        print(f"已处理 {min(offset + len(batch), len(todo))}/{len(todo)}；成功 {len(completed)}；失败 {len(failed)}")
        if offset + len(batch) < len(todo):
            time.sleep(args.pause)

    manifest, invalid = build_manifest(output_dir, universe)
    all_failures = failed + invalid
    manifest.to_csv(output_dir / "downloaded_manifest.csv", index=False)
    pd.DataFrame(all_failures, columns=["symbol", "yahoo_symbol", "reason"]).to_csv(
        output_dir / "failed_downloads.csv", index=False)
    metadata = {"downloaded_at": date.today().isoformat(), "source": NASDAQ_SCREENER,
                "years_requested": args.years, "start": start.isoformat(), "end_exclusive": end.isoformat(),
                "universe_count": len(universe), "skipped_existing": existing,
                "downloaded_this_run": len(completed), "failed_this_run": len(failed),
                "valid_data_files": len(manifest), "invalid_or_missing_files": len(invalid),
                "scope": "当前 Nasdaq 上市、名称包含 Common Stock 或 Ordinary Shares 的证券；排除 ETF、基金、权证等"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。有效数据文件：{len(manifest)}；无效或缺失：{len(invalid)}。数据与清单位于：{output_dir}")


if __name__ == "__main__":
    main()
