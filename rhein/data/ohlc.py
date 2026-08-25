"""OHLC loading with CSV compatibility and Parquet as the primary format."""
from pathlib import Path

import pandas as pd


def load_ohlc(path: Path) -> pd.DataFrame:
    """Read a project OHLCV Parquet file or one of the legacy CSV layouts."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
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


def load_ohlc_dates(path: Path) -> pd.Series:
    """Read only Date for group-wide coverage checks without loading OHLCV."""
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["Date"])
    else:
        with path.open(encoding="utf-8") as fh:
            first = fh.readline()
        frame = pd.read_csv(path, skiprows=3, header=None, names=["Date"], usecols=[0]) if first.startswith("Price,") else pd.read_csv(path, usecols=["Date"])
    return pd.to_datetime(frame["Date"], errors="coerce").dropna()
