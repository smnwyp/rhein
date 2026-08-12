"""Download and install the external A-share OHLCV release asset."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhein.data.a_share_package import download_archive, install_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并安装 A 股 OHLCV 数据包")
    parser.add_argument("--url", required=True, help="GitHub Release asset 或对象存储的直接下载链接")
    parser.add_argument("--sha256", help="发布时记录的 SHA-256；强烈建议提供")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--replace", action="store_true", help="覆盖已有 data/a_share_ohlcv 数据")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="a_share_download_") as temporary:
        archive_path = Path(temporary) / "a_share_ohlcv.tar.gz"
        download_archive(url=args.url, destination=archive_path, expected_sha256=args.sha256)
        target = install_archive(archive_path=archive_path, data_root=Path(args.data_root), replace=args.replace)
    print(f"安装完成：{target}")


if __name__ == "__main__":
    main()
