"""Convert TongDaXin A-share text exports into Rhein-compatible OHLCV CSV files."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Direct ``python scripts/...`` execution puts ``scripts/`` (not the repository
# root) on sys.path.  Make the local compatibility loader available without
# requiring users to set PYTHONPATH or install the legacy ``rhein`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhein.data.a_share import convert_a_share_file


def destination_for(source: Path, output_dir: Path) -> Path:
    """Use a shell-friendly market-prefixed symbol, e.g. SH#600831 -> SH600831.csv."""
    return output_dir / f"{source.stem.replace('#', '')}.csv"


def source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.txt"))
    raise ValueError(f"找不到输入文件或目录：{input_path}")


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    """Write a UTF-8 failure report so a long batch can be reviewed or retried."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="将通达信 A 股日线 TXT 转为 Rhein 回测所需的 OHLCV CSV")
    parser.add_argument("--input-path", default="data/A股", help="单个 TXT 文件或包含 TXT 的目录")
    parser.add_argument("--output-dir", default="data/a_share_ohlcv", help="转换后 CSV 的目录")
    parser.add_argument("--limit", type=int, help="最多转换 N 个文件，便于试运行")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已经存在的 CSV")
    parser.add_argument("--failure-report", help="失败文件清单 CSV 路径；默认写入输出目录")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须为正整数")

    sources = source_files(Path(args.input_path))
    if args.limit is not None:
        sources = sources[:args.limit]
    if not sources:
        raise ValueError("没有发现可转换的 TXT 文件")

    output_dir = Path(args.output_dir)
    converted = skipped = 0
    failures: list[dict[str, str]] = []
    for source in sources:
        destination = destination_for(source, output_dir)
        if destination.exists() and not args.overwrite:
            print(f"跳过（已存在）：{destination}")
            skipped += 1
            continue
        try:
            rows = convert_a_share_file(source, destination)
        except ValueError as exc:
            error = str(exc)
            failures.append({"source_file": str(source), "error": error})
            print(f"转换失败：{source}：{error}")
            continue
        print(f"已转换：{source} -> {destination}（{rows} 行）")
        converted += 1
    if failures:
        report = Path(args.failure_report) if args.failure_report else output_dir / "conversion_failures.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        write_failures(report, failures)
        print(f"完成：转换 {converted} 个，跳过 {skipped} 个，失败 {len(failures)} 个。失败清单：{report}")
    else:
        print(f"完成：转换 {converted} 个，跳过 {skipped} 个，失败 0 个。")


if __name__ == "__main__":
    main()
