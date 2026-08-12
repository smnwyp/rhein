"""Create the A-share OHLCV release asset without adding market data to Git."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhein.data.a_share_package import build_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 A 股 OHLCV CSV，作为 GitHub Release / 对象存储的下载资产")
    parser.add_argument("--source-dir", default="data/a_share_ohlcv")
    parser.add_argument("--output", default="artifacts/a_share_ohlcv.tar.gz")
    args = parser.parse_args()

    count, digest = build_archive(source_dir=Path(args.source_dir), destination=Path(args.output))
    print(f"已打包 {count} 个 CSV：{args.output}")
    print(f"SHA-256：{digest}")
    print("请将此文件上传为 GitHub Release asset；它不应提交进 Git。")


if __name__ == "__main__":
    main()
