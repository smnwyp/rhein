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

from rhein.data.a_share import SourceDataError, convert_a_share_file


def destination_for(source: Path, output_dir: Path) -> Path:
    """Use a shell-friendly market-prefixed symbol, e.g. SH#600831 -> SH600831.csv."""
    return output_dir / f"{source.stem.replace('#', '')}.csv"


def source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.txt"))
    raise ValueError(f"找不到输入文件或目录：{input_path}")


FAILURE_FIELDS = ["source_file", "output_file", "category", "code", "error"]


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    """Write a UTF-8 failure report so a long batch can be reviewed or retried."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        writer.writerows(failures)


def write_markdown_report(
    path: Path, *, total: int, converted: int, skipped: int, failures: list[dict[str, str]],
) -> None:
    """Create a human-readable report that separates source data from converter faults."""
    source_failures = [failure for failure in failures if failure["category"] == "source_data"]
    transform_failures = [failure for failure in failures if failure["category"] == "transformation"]
    lines = [
        "# A 股日线转换报告", "",
        f"- 输入文件：{total}",
        f"- 成功转换：{converted}",
        f"- 已跳过：{skipped}",
        f"- 源文件问题：{len(source_failures)}",
        f"- 转换器问题：{len(transform_failures)}", "",
        "## 结论", "",
    ]
    if transform_failures:
        lines.append("发现转换器问题；这些需要修复脚本后重试。")
    else:
        lines.append("未发现转换器问题。失败均来自源文件本身，未生成或覆盖对应的 CSV。")
    lines += ["", "## 源文件问题", "", "| 源文件 | 分类 | 原因 |", "|---|---|---|"]
    lines.extend(
        f"| {failure['source_file']} | {failure['code']} | {failure['error']} |"
        for failure in source_failures
    )
    if not source_failures:
        lines.append("| 无 | - | - |")
    lines += ["", "## 转换器问题", "", "| 源文件 | 分类 | 原因 |", "|---|---|---|"]
    lines.extend(
        f"| {failure['source_file']} | {failure['code']} | {failure['error']} |"
        for failure in transform_failures
    )
    if not transform_failures:
        lines.append("| 无 | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="将通达信 A 股日线 TXT 转为 Rhein 回测所需的 OHLCV CSV")
    parser.add_argument("--input-path", default="data/A股", help="单个 TXT 文件或包含 TXT 的目录")
    parser.add_argument("--output-dir", default="data/a_share_ohlcv", help="转换后 CSV 的目录")
    parser.add_argument("--limit", type=int, help="最多转换 N 个文件，便于试运行")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已经存在的 CSV")
    parser.add_argument("--failure-report", help="失败文件清单 CSV 路径；默认写入输出目录")
    parser.add_argument("--report", help="转换报告 Markdown 路径；默认写入输出目录")
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
        except SourceDataError as exc:
            error = str(exc)
            failures.append({
                "source_file": str(source), "output_file": str(destination), "category": "source_data",
                "code": exc.code, "error": error,
            })
            if args.overwrite and destination.exists():
                destination.unlink()
            print(f"源文件问题：{source}：{error}")
            continue
        except Exception as exc:
            error = str(exc)
            failures.append({
                "source_file": str(source), "output_file": str(destination), "category": "transformation",
                "code": type(exc).__name__, "error": error,
            })
            print(f"转换器问题：{source}：{error}")
            continue
        print(f"已转换：{source} -> {destination}（{rows} 行）")
        converted += 1
    failure_report = Path(args.failure_report) if args.failure_report else output_dir / "conversion_failures.csv"
    markdown_report = Path(args.report) if args.report else output_dir / "conversion_report.md"
    # A short-lived earlier converter version emitted this report while
    # truncating invalid source series.  It no longer reflects current output.
    legacy_warning_report = output_dir / "conversion_warnings.csv"
    if args.overwrite and legacy_warning_report.is_file():
        legacy_warning_report.unlink()
    failure_report.parent.mkdir(parents=True, exist_ok=True)
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    write_failures(failure_report, failures)
    write_markdown_report(
        markdown_report, total=len(sources), converted=converted, skipped=skipped, failures=failures,
    )
    source_failures = sum(failure["category"] == "source_data" for failure in failures)
    transform_failures = sum(failure["category"] == "transformation" for failure in failures)
    print(
        f"完成：转换 {converted} 个，跳过 {skipped} 个，失败 {len(failures)} 个"
        f"（源文件 {source_failures}，转换器 {transform_failures}）。"
    )
    print(f"失败清单：{failure_report}；转换报告：{markdown_report}")


if __name__ == "__main__":
    main()
