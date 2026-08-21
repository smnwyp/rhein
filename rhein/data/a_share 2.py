"""Convert TongDaXin A-share daily exports into the project's OHLCV CSV schema."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
_SOURCE_COLUMNS = [*OUTPUT_COLUMNS, "Amount"]


def _read_source(path: Path) -> pd.DataFrame:
    """Read a TongDaXin tab-delimited daily export without relying on its Chinese headers."""
    try:
        frame = pd.read_csv(
            path,
            encoding="gb18030",
            sep=r"\s+",
            skiprows=2,
            skipfooter=1,
            header=None,
            names=_SOURCE_COLUMNS,
            engine="python",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path}: 无法读取通达信日线导出文件") from exc
    if frame.empty:
        raise ValueError(f"{path}: 没有可转换的日线记录")
    return frame


def convert_a_share_file(source: Path, destination: Path) -> int:
    """Convert one source file and return the number of converted daily bars.

    The source format is the TongDaXin export currently stored under ``data/A股``:
    two descriptive rows, daily rows in ``DD/MM/YYYY`` format, and one source footer.
    Prices are preserved as supplied (the current files identify themselves as 前复权).
    """
    frame = _read_source(source)
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="raise")
        for column in OUTPUT_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: 日线数据包含无法解析的日期或 OHLCV 数值") from exc
    if frame[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError(f"{source}: 日线数据包含空值")
    if (frame[["Open", "High", "Low", "Close", "Volume"]] < 0).any().any():
        raise ValueError(f"{source}: 日线数据包含负的价格或成交量")
    if (frame["High"] < frame[["Open", "Low", "Close"]].max(axis=1)).any() or (
        frame["Low"] > frame[["Open", "High", "Close"]].min(axis=1)
    ).any():
        raise ValueError(f"{source}: OHLC 高低价关系无效")
    if frame["Date"].duplicated().any():
        raise ValueError(f"{source}: 存在重复交易日期")

    converted = frame[OUTPUT_COLUMNS].sort_values("Date").reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(destination, index=False, date_format="%Y-%m-%d", quoting=csv.QUOTE_MINIMAL)
    return len(converted)
