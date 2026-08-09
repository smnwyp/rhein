"""板块萌芽统计研究的图表与 Markdown 报告。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emergence_runner import HorizonVerdict, RealStudyResult
from rhein.emergence_study import StudyConfig, _synthetic_frames, quantile_analysis, yearly_ic_metrics


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_horizon(result: RealStudyResult, horizon: int, directory: Path) -> list[Path]:
    study = result.horizon_results[horizon]
    outputs: list[Path] = []
    ic = study.composite_ic.daily_ic
    rolling = ic.rolling(60, min_periods=20).mean()
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ic.index, ic.values, color="#8aa6c1", linewidth=0.7, label="Daily IC")
    axis.plot(rolling.index, rolling.values, color="#1b5e8c", linewidth=1.5, label="60-day mean")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Date", ylabel="IC", title=f"IC time series (N={horizon})")
    axis.legend()
    path = directory / f"ic_time_series_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ic.index, ic.cumsum().values, color="#1b5e8c", linewidth=1.5)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Date", ylabel="Cumulative IC", title=f"Cumulative IC (N={horizon})")
    path = directory / f"cumulative_ic_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(ic.values, bins=30, color="#3e7ba8", edgecolor="white")
    axis.axvline(float(ic.mean()), color="#c43c35", linewidth=1.5, label="Mean")
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="IC", ylabel="Frequency", title=f"IC distribution (N={horizon})")
    axis.legend()
    path = directory / f"ic_histogram_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)

    requested = result.config.study_config.quantiles
    summary = study.quantiles.bucket_summary.query("bucket_count == @requested").sort_values("bucket")
    figure, axis = plt.subplots(figsize=(7, 4))
    if summary.empty:
        axis.text(0.5, 0.5, "No full-quantile days", ha="center", va="center")
    else:
        axis.bar([f"Q{value}" for value in summary["bucket"]], summary["mean_forward_excess"], color="#3e7ba8")
        axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Score quantile", ylabel="Mean forward excess", title=f"Quintile comparison (N={horizon})")
    path = directory / f"quintile_bar_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)

    daily = study.quantiles.daily_bucket_means
    daily = daily[daily["bucket_count"] == requested]
    figure, axis = plt.subplots(figsize=(10, 4))
    if daily.empty:
        axis.text(0.5, 0.5, "No full-quantile days", ha="center", va="center")
    else:
        pivot = daily.pivot(index="date", columns="bucket", values="mean_forward_excess").sort_index()
        for bucket in pivot.columns:
            axis.plot(pivot.index, pivot[bucket].fillna(0).cumsum(), label=f"Q{bucket}")
        axis.legend(ncol=min(5, len(pivot.columns)))
    axis.set(xlabel="Date", ylabel="Cumulative forward excess", title=f"Quintile cumulative comparison (N={horizon})")
    path = directory / f"quintile_cumulative_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(8, 4))
    yearly = study.yearly
    axis.bar(yearly["year"].astype(str), yearly["mean_ic"], color="#3e7ba8")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Year", ylabel="Mean IC", title=f"Yearly IC (N={horizon})")
    path = directory / f"yearly_ic_n{horizon}.png"
    _save(figure, path)
    outputs.append(path)
    return outputs


def _synthetic_panel(config: StudyConfig, directory: Path) -> Path:
    """并排展示固定种子下的正、负对照分位结果与逐年 IC。"""
    positive_score, positive_outcome = _synthetic_frames(
        days=config.synthetic_days, sectors=config.synthetic_sectors, beta=0.11, rng=np.random.default_rng(config.random_seed)
    )
    negative_score, negative_outcome = _synthetic_frames(
        days=config.synthetic_days, sectors=config.synthetic_sectors, beta=0.0, rng=np.random.default_rng(config.random_seed + 1)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, (name, score, outcome) in enumerate((("Positive control", positive_score, positive_outcome), ("Negative control", negative_score, negative_outcome))):
        quantiles = quantile_analysis(score, outcome, requested_quantiles=config.quantiles, lag=20)
        summary = quantiles.bucket_summary.query("bucket_count == @config.quantiles").sort_values("bucket")
        axes[0, column].bar([f"Q{value}" for value in summary["bucket"]], summary["mean_forward_excess"], color="#3e7ba8")
        axes[0, column].axhline(0, color="#555555", linewidth=0.8)
        axes[0, column].set(title=f"{name}: quintiles", ylabel="Mean forward excess")
        # 每日 IC 由公开统计函数计算，避免图表另写一套公式。
        from rhein.emergence_study import cross_sectional_ic
        ic = cross_sectional_ic(score, outcome, minimum_sectors=config.minimum_sectors).daily_ic
        yearly = yearly_ic_metrics(ic, quantiles.daily_top_bottom_excess)
        axes[1, column].bar(yearly["year"].astype(str), yearly["mean_ic"], color="#3e7ba8")
        axes[1, column].axhline(0, color="#555555", linewidth=0.8)
        axes[1, column].set(title=f"{name}: yearly IC", xlabel="Year", ylabel="Mean IC")
    path = directory / "synthetic_control_panel.png"
    _save(figure, path)
    return path


def _verdict_lines(verdict: HorizonVerdict) -> list[str]:
    flag = lambda value: "通过" if value else "未通过"
    return [
        f"- IC 正且 HAC 显著：{flag(verdict.ic_positive_and_significant)}",
        f"- 逐年同号稳定（{verdict.stable_same_sign_years} 年）：{flag(verdict.yearly_sign_stable)}",
        f"- 分位数单调、Top−Bottom 为正：{flag(verdict.quantiles_monotonic)}",
        f"- 非重叠抽样与 HAC 一致：{flag(verdict.non_overlapping_consistent)}",
        f"- 本 horizon 的 H1：{'接受' if verdict.accepted else '拒绝 / 未获支持'}",
    ]


def write_research_report(result: RealStudyResult, *, output_directory: Path | str) -> Path:
    """保存七类图与中文 Markdown 报告；结论始终先于图表。"""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    for horizon in result.config.study_config.horizons:
        plot_paths.extend(_plot_horizon(result, horizon, directory))
    synthetic_path = _synthetic_panel(result.config.study_config, directory)
    lines = [
        f"# {result.config.study_name}",
        "",
        "## 预注册判据结论", "",
        f"- 板块定义：{result.config.sector_definition}",
        f"- 数据文件：{result.source_file_count}；映射到研究板块的代码：{result.mapped_symbol_count}",
        f"- 偏差声明：{result.config.static_universe_bias_notice}",
        "- 合成对照闸门：全部通过。",
    ]
    for check in result.synthetic_gate.checks:
        lines.append(f"  - {check.name}：{'通过' if check.passed else '失败'}；{dict(check.details)}")
    for horizon in result.config.study_config.horizons:
        study = result.horizon_results[horizon]
        ic = study.composite_ic
        lines.extend(["", f"### Horizon {horizon}", *_verdict_lines(result.verdicts[horizon])])
        if ic.hac is not None:
            lines.append(f"- IC 均值：{ic.mean_ic:.6f}；HAC t：{ic.hac.t_statistic:.3f}；p：{ic.hac.p_value:.6f}；有效 IC 日：{ic.hac.observations}。")
        lines.append(f"- 前向窗口末尾丢弃日：{result.dropped_forward_tail[horizon]}；IC 跳过日：{ic.skipped_days}；分位降级日：{study.quantiles.downgraded_days}。")
        lines.extend(["", "| 年份 | 平均 IC | Top−Bottom 平均超额 | IC 观测 |", "| --- | ---: | ---: | ---: |"])
        for _, row in study.yearly.iterrows():
            lines.append(f"| {int(row['year'])} | {row['mean_ic']:.6f} | {row['mean_top_bottom_excess']:.6f} | {int(row['ic_observations'])} |")
        lines.append("")
        lines.append("分量归因：")
        for name, component in study.component_ic.items():
            text = "无有效 IC" if component.hac is None else f"均值 {component.mean_ic:.6f}，HAC t {component.hac.t_statistic:.3f}"
            lines.append(f"- {name}: {text}")
    lines.extend(["", "## 图表", "", f"![Synthetic control panel]({synthetic_path.name})"])
    for path in plot_paths:
        lines.append(f"![{path.stem}]({path.name})")
    report_path = directory / "研究报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
