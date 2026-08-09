"""运行板块萌芽现象的统计检验研究并输出图表与报告。"""
from __future__ import annotations

import argparse
from pathlib import Path

from emergence_runner import load_research_config, run_real_study
from rhein.emergence_reporting import write_research_report
from rhein.paths import REPORTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="运行板块萌芽现象统计检验研究")
    parser.add_argument("--data-path", default="data/nasdaq_10y")
    parser.add_argument("--config", default="config/emergence_study.toml")
    parser.add_argument("--output-directory", default=str(REPORTS_ROOT / "emergence_study"))
    parser.add_argument("--adjustment-confirmed", action="store_true", help="明确确认输入日线为一致复权价格")
    args = parser.parse_args()
    result = run_real_study(
        data_path=Path(args.data_path),
        adjustment_confirmed=args.adjustment_confirmed,
        config=load_research_config(args.config),
    )
    report = write_research_report(result, output_directory=args.output_directory)
    print(f"合成对照四关已通过；研究报告已写入：{report}")
    for horizon, verdict in result.verdicts.items():
        print(f"Horizon {horizon}: {'接受 H1' if verdict.accepted else '拒绝 / 未获支持 H1'}")


if __name__ == "__main__":
    main()
