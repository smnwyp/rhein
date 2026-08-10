"""运行 v1.1 steady 分量单独检验并输出独立研究报告。"""
from __future__ import annotations

import argparse

from rhein.paths import REPORTS_ROOT
from rhein.steady_supplement_reporting import write_steady_supplement_report
from steady_supplement_runner import load_steady_supplement_config, run_steady_supplement


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 steady 分量单独检验")
    parser.add_argument("--data-path", default="data/nasdaq_10y")
    parser.add_argument("--config", default="config/steady_supplement.toml")
    parser.add_argument("--output-directory", default=str(REPORTS_ROOT / "steady_supplement"))
    parser.add_argument("--adjustment-confirmed", action="store_true", help="明确确认输入日线为一致复权价格")
    args = parser.parse_args()
    result = run_steady_supplement(
        data_path=args.data_path,
        adjustment_confirmed=args.adjustment_confirmed,
        config=load_steady_supplement_config(args.config),
    )
    report = write_steady_supplement_report(result, output_directory=args.output_directory)
    print(f"steady 增补研究报告已写入：{report}")
    print(result.verdict.outcome)


if __name__ == "__main__":
    main()
