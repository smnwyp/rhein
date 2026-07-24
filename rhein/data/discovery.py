"""Identify valid OHLC files without treating reports as market data."""
from pathlib import Path

import pandas as pd


def input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        # 特征组以 manifest 为唯一股票清单，避免手工副本重复回测。
        manifest_path = input_path / "group_manifest.csv"
        if manifest_path.is_file():
            try:
                symbols = pd.read_csv(manifest_path, usecols=["symbol"])["symbol"].dropna()
                manifest_files = [input_path / f"{str(symbol).upper()}.csv" for symbol in symbols]
                files = [path for path in manifest_files if path.is_file()]
                if files:
                    return files
            except (OSError, ValueError, KeyError):
                pass
        files = []
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        for path in sorted(input_path.glob("*.csv")):
            try:
                with path.open(encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if first.startswith("Price,"):
                    files.append(path)
                    continue
                columns = {column.strip().capitalize() for column in first.split(",")}
                if required <= columns:
                    files.append(path)
            except OSError:
                continue
        if not files:
            raise ValueError(f"{input_path} 中没有可识别的 OHLC CSV 文件")
        return files
    if input_path.is_file():
        return [input_path]
    raise ValueError(f"找不到输入路径：{input_path}")
