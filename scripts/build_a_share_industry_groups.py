"""Create selectable A-share industry groups from a current static snapshot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhein.data.a_share_industry import build_and_write_current_industry_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="按申万 2021 一级行业当前快照构建 A 股标的分组")
    parser.add_argument("--data-root", default="data/a_share_ohlcv", help="A 股 OHLCV Parquet/CSV 目录")
    args = parser.parse_args()
    result = build_and_write_current_industry_groups(data_root=Path(args.data_root))
    print(f"已生成 {result['groups']} 个 A 股行业分组：映射 {result['mapped']} 个，未映射 {result['unmapped']} 个。")


if __name__ == "__main__":
    main()
