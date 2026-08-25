"""Identify valid OHLC files without treating reports as market data."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        # 特征组以 manifest 为唯一股票清单，避免手工副本重复回测。
        manifest_path = input_path / "group_manifest.csv"
        if manifest_path.is_file():
            try:
                symbols = pd.read_csv(manifest_path, usecols=["symbol"])["symbol"].dropna()
                # Prefer Parquet when a group has been migrated, while keeping
                # the existing CSV groups readable during the rollout.
                manifest_files = [
                    next((input_path / f"{str(symbol).upper()}{suffix}" for suffix in (".parquet", ".csv")
                         if (input_path / f"{str(symbol).upper()}{suffix}").is_file()), None)
                    for symbol in symbols
                ]
                files = [path for path in manifest_files if path is not None]
                if files:
                    return files
            except (OSError, ValueError, KeyError):
                pass
        files_by_symbol: dict[str, Path] = {}
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        for path in sorted([*input_path.glob("*.parquet"), *input_path.glob("*.csv")]):
            try:
                if path.suffix.lower() == ".parquet":
                    # Schema inspection must never materialise an entire
                    # A-share file just to discover its column names.
                    columns = {str(column).strip().capitalize() for column in pq.ParquetFile(path).schema.names}
                    if required <= columns:
                        files_by_symbol[path.stem.upper()] = path
                    continue
                with path.open(encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if first.startswith("Price,"):
                    files_by_symbol.setdefault(path.stem.upper(), path)
                    continue
                columns = {column.strip().capitalize() for column in first.split(",")}
                if required <= columns:
                    files_by_symbol.setdefault(path.stem.upper(), path)
            except OSError:
                continue
        files = sorted(files_by_symbol.values())
        if not files:
            raise ValueError(f"{input_path} 中没有可识别的 OHLC CSV 或 Parquet 文件")
        return files
    if input_path.is_file():
        return [input_path]
    raise ValueError(f"找不到输入路径：{input_path}")
