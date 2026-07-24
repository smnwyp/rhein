"""OHLC CSV loading with the repository's yfinance compatibility rules."""
from pathlib import Path

import pandas as pd


def load_ohlc(path: Path) -> pd.DataFrame:
    """读取普通 CSV 或 yfinance 导出的三行表头 CSV。"""
    with path.open(encoding="utf-8") as fh:
        first = fh.readline()
    if first.startswith("Price,"):
        df = pd.read_csv(path, skiprows=3, header=None)
        cols = [c.strip() for c in first.strip().split(",")]
        cols[0] = "Date"
        df.columns = cols
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    need = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: CSV 缺少字段：{', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"])
    for column in need - {"Date"}:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df.sort_values("Date").reset_index(drop=True)
