"""Run the configured front-70% sector-threshold grid and save its report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rhein.data.discovery import input_files
from rhein.paths import CONFIG_ROOT, REPORTS_ROOT
from rhein.trend_tracking import (
    aggregate_sector_daily,
    calibrate_threshold_grid,
    clean_stock_daily,
    fixed_holding_diagnostics,
    load_industry_map,
    load_stock_daily,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行板块趋势追踪的前七年候选阈值网格")
    parser.add_argument("--data-path", default="data/nasdaq_10y")
    parser.add_argument("--industry-map", default="data/nasdaq_10y/industry_map.csv")
    parser.add_argument("--config", default=str(CONFIG_ROOT / "trend_tracking.json"))
    parser.add_argument("--output", default=str(REPORTS_ROOT / "trend_tracking_calibration_grid.csv"))
    parser.add_argument("--fixed-holding-output", default=str(REPORTS_ROOT / "trend_tracking_fixed_holding_diagnostic.csv"))
    parser.add_argument("--adjustment-policy", choices=["adjusted", "unknown_conservative_drop_outliers"], default="adjusted")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    calibration = config["calibration"]
    locked = config["locked_thresholds"]
    if locked.get("entry_threshold") is not None or locked.get("exit_threshold") is not None:
        raise ValueError("当前任务是标定；locked_thresholds 必须保持为空")
    paths = input_files(Path(args.data_path))
    print(f"读取 {len(paths)} 个日线文件…")
    cleaned = clean_stock_daily(
        load_stock_daily(paths), load_industry_map(Path(args.industry_map)), adjustment_policy=args.adjustment_policy
    )
    sector_daily = aggregate_sector_daily(cleaned)
    grid = calibrate_threshold_grid(
        cleaned,
        sector_daily,
        entry_thresholds=calibration["entry_thresholds"],
        exit_gaps=calibration["exit_gaps"],
        adjustment_policy=args.adjustment_policy,
        transaction_cost=float(calibration["transaction_cost"]),
        calibration_ratio=float(calibration["calibration_ratio"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False, encoding="utf-8")
    print(f"已输出 {len(grid)} 组前七年标定结果：{output}")
    fixed_holding = fixed_holding_diagnostics(
        cleaned,
        sector_daily,
        entry_thresholds=calibration["entry_thresholds"],
        transaction_cost=float(calibration["transaction_cost"]),
        calibration_ratio=float(calibration["calibration_ratio"]),
    )
    fixed_output = Path(args.fixed_holding_output)
    fixed_output.parent.mkdir(parents=True, exist_ok=True)
    fixed_holding.to_csv(fixed_output, index=False, encoding="utf-8")
    print(f"已输出 {len(fixed_holding)} 组前七年固定持有期诊断：{fixed_output}")


if __name__ == "__main__":
    main()
