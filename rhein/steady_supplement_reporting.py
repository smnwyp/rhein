"""steady 增补研究的图表与 Markdown 报告。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rhein.emergence_reporting import _save, _synthetic_panel
from rhein.emergence_study import HorizonStudyResult
from steady_supplement_runner import SteadySupplementResult


def _plot_standard_outputs(study: HorizonStudyResult, *, quantiles: int, horizon: int, directory: Path) -> list[Path]:
    """以 steady 为 score 复用 v1.0 的六种标准统计图。"""
    paths: list[Path] = []
    ic = study.composite_ic.daily_ic
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ic.index, ic.values, color="#8aa6c1", linewidth=0.7, label="Daily IC")
    axis.plot(ic.index, ic.rolling(60, min_periods=20).mean(), color="#1b5e8c", linewidth=1.5, label="60-day mean")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Date", ylabel="IC", title=f"Steady IC time series (N={horizon})")
    axis.legend()
    path = directory / f"steady_ic_time_series_n{horizon}.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ic.index, ic.cumsum(), color="#1b5e8c", linewidth=1.5)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Date", ylabel="Cumulative IC", title=f"Steady cumulative IC (N={horizon})")
    path = directory / f"steady_cumulative_ic_n{horizon}.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(ic.values, bins=30, color="#3e7ba8", edgecolor="white")
    axis.axvline(float(ic.mean()), color="#c43c35", linewidth=1.5, label="Mean")
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="IC", ylabel="Frequency", title=f"Steady IC distribution (N={horizon})")
    axis.legend()
    path = directory / f"steady_ic_histogram_n{horizon}.png"
    _save(figure, path)
    paths.append(path)

    summary = study.quantiles.bucket_summary.query("bucket_count == @quantiles").sort_values("bucket")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([f"Q{value}" for value in summary["bucket"]], summary["mean_forward_excess"], color="#2f7d4a")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Steady quantile", ylabel="Mean forward excess", title=f"Steady quintiles (N={horizon})")
    path = directory / f"steady_quintile_bar_n{horizon}.png"
    _save(figure, path)
    paths.append(path)

    daily = study.quantiles.daily_bucket_means.query("bucket_count == @quantiles")
    figure, axis = plt.subplots(figsize=(10, 4))
    pivot = daily.pivot(index="date", columns="bucket", values="mean_forward_excess").sort_index()
    for bucket in pivot.columns:
        axis.plot(pivot.index, pivot[bucket].fillna(0).cumsum(), label=f"Q{bucket}")
    axis.set(xlabel="Date", ylabel="Cumulative forward excess", title=f"Steady quintile cumulative comparison (N={horizon})")
    axis.legend(ncol=min(5, len(pivot.columns)))
    path = directory / f"steady_quintile_cumulative_n{horizon}.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(study.yearly["year"].astype(str), study.yearly["mean_ic"], color="#2f7d4a")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Year", ylabel="Mean IC", title=f"Steady yearly IC (N={horizon})")
    path = directory / f"steady_yearly_ic_n{horizon}.png"
    _save(figure, path)
    paths.append(path)
    return paths


def _comparison_plot(result: SteadySupplementResult, *, horizon: int, directory: Path) -> Path:
    """并排展示 steady 与已否决合成分的五档结构。"""
    count = result.base.config.study_config.quantiles
    steady = result.steady_horizon_results[horizon].quantiles.bucket_summary.query("bucket_count == @count").sort_values("bucket")
    composite = result.base.horizon_results[horizon].quantiles.bucket_summary.query("bucket_count == @count").sort_values("bucket")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, name, summary, color in (
        (axes[0], "Steady", steady, "#2f7d4a"),
        (axes[1], "Composite (rejected in v1.0)", composite, "#8f4b4b"),
    ):
        axis.bar([f"Q{value}" for value in summary["bucket"]], summary["mean_forward_excess"], color=color)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set(title=f"{name} quintiles (N={horizon})", xlabel="Score quantile", ylabel="Mean forward excess")
    path = directory / f"steady_vs_composite_quintiles_n{horizon}.png"
    _save(figure, path)
    return path


def write_steady_supplement_report(result: SteadySupplementResult, *, output_directory: Path | str) -> Path:
    """输出 v1.1 预注册判据、逐年价差和与合成分的并排图。"""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    quantiles = result.base.config.study_config.quantiles
    standard_paths: list[Path] = []
    comparison_paths: list[Path] = []
    for horizon, study in result.steady_horizon_results.items():
        standard_paths.extend(_plot_standard_outputs(study, quantiles=quantiles, horizon=horizon, directory=directory))
        comparison_paths.append(_comparison_plot(result, horizon=horizon, directory=directory))
    synthetic_path = _synthetic_panel(result.base.config.study_config, directory)
    verdict = result.verdict
    flag = lambda value: "通过" if value else "未通过"
    lines = [
        f"# {result.config.study_name}", "",
        "## 预注册判据结论", "",
        "- v1.0 合成萌芽分已否决；本报告不重新加权、不调 steady 常数，只把既有 steady 单独送考。",
        f"- 板块定义：{result.base.config.sector_definition}",
        f"- 偏差声明：{result.config.static_universe_bias_notice}",
        "- 合成对照闸门：全部通过（沿用同一 v1.0 真实研究入口）。",
        f"- 分位单调且 Q5>0（N={list(result.config.required_monotonic_horizons)}）：{flag(verdict.quantile_requirement_passed)}",
        f"- Top−Bottom 均值>0 且 HAC |t|>{result.config.top_bottom_hac_abs_t_minimum:g}（全部 horizon）：{flag(verdict.top_bottom_requirement_passed)}",
        f"- 逐年价差稳定（≥{result.config.minimum_positive_spread_years}/10 年为正，2017–2020 不得全负）：{flag(verdict.yearly_requirement_passed)}",
        f"- 非重叠 IC 核对与 HAC 一致（全部 horizon）：{flag(verdict.non_overlapping_requirement_passed)}",
        f"- 裁决：**{verdict.outcome}**",
    ]
    for check in result.base.synthetic_gate.checks:
        lines.append(f"  - {check.name}：{'通过' if check.passed else '失败'}；{dict(check.details)}")
    for horizon, study in result.steady_horizon_results.items():
        item = result.horizon_verdicts[horizon]
        spread_hac = study.quantiles.hac
        ic_hac = study.composite_ic.hac
        lines.extend(["", f"## Horizon {horizon}", ""])
        lines.extend([
            f"- 五档严格向上：{flag(item.quantiles_strictly_upward)}；Q5>0：{flag(item.q5_positive)}。",
            f"- Top−Bottom：{flag(item.top_bottom_positive_and_significant)}；均值 {spread_hac.mean:.6f}，HAC t {spread_hac.t_statistic:.3f}，p {spread_hac.p_value:.6f}。" if spread_hac else "- Top−Bottom：无足够观测。",
            f"- 逐年价差为正：{item.positive_spread_years} 年；2017–2020 全负：{flag(item.early_years_all_negative)}。",
            f"- steady IC：均值 {study.composite_ic.mean_ic:.6f}，HAC t {ic_hac.t_statistic:.3f}；非重叠核对：{flag(item.non_overlapping_consistent)}。" if ic_hac else "- steady IC：无足够观测。",
            "",
            "| 年份 | steady IC 均值 | Top−Bottom 平均超额 | IC 观测 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for _, row in study.yearly.iterrows():
            lines.append(f"| {int(row['year'])} | {row['mean_ic']:.6f} | {row['mean_top_bottom_excess']:.6f} | {int(row['ic_observations'])} |")
    lines.extend(["", "## 图表", "", f"![Synthetic controls]({synthetic_path.name})"])
    for path in comparison_paths:
        lines.append(f"![{path.stem}]({path.name})")
    for path in standard_paths:
        lines.append(f"![{path.stem}]({path.name})")
    report_path = directory / "steady_增补研究报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
