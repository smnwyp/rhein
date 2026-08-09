"""板块萌芽统计检验研究的 Streamlit 界面。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from emergence_runner import RealStudyResult, load_research_config, run_real_study
from rhein.emergence_reporting import write_research_report
from rhein.emergence_study import EmergenceStudyError
from rhein.paths import REPORTS_ROOT


def _render_error(error: EmergenceStudyError) -> None:
    st.error(f"{error.code}: {error.message}")
    if error.details:
        with st.expander("查看机器可读错误详情"):
            st.json(error.details)


def _summary(result: RealStudyResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, study in result.horizon_results.items():
        verdict = result.verdicts[horizon]
        hac = study.composite_ic.hac
        rows.append({
            "前向 horizon（交易日）": horizon,
            "IC 均值": study.composite_ic.mean_ic,
            "HAC t": hac.t_statistic if hac else None,
            "HAC p": hac.p_value if hac else None,
            "有效 IC 日": hac.observations if hac else 0,
            "逐年同号数": verdict.stable_same_sign_years,
            "分位数单调": verdict.quantiles_monotonic,
            "非重叠核对一致": verdict.non_overlapping_consistent,
            "H1 判据": "接受" if verdict.accepted else "拒绝 / 未获支持",
        })
    return pd.DataFrame(rows)


def _show_result(result: RealStudyResult, report_path: Path) -> None:
    st.subheader("预注册判据")
    st.caption(f"板块定义：{result.config.sector_definition}。{result.config.static_universe_bias_notice}")
    st.success("合成对照四关均已通过，真实数据研究结果有效。")
    st.dataframe(_summary(result), hide_index=True, width="stretch")
    for horizon, study in result.horizon_results.items():
        with st.expander(f"Horizon {horizon}：逐年分解、分量归因与图表", expanded=horizon == min(result.horizon_results)):
            st.dataframe(study.yearly, hide_index=True, width="stretch")
            components = pd.DataFrame([
                {"分量": name, "IC 均值": metric.mean_ic, "HAC t": metric.hac.t_statistic if metric.hac else None, "HAC p": metric.hac.p_value if metric.hac else None}
                for name, metric in study.component_ic.items()
            ])
            st.dataframe(components, hide_index=True, width="stretch")
            for stem in ("ic_time_series", "cumulative_ic", "ic_histogram", "quintile_bar", "quintile_cumulative", "yearly_ic"):
                path = report_path.parent / f"{stem}_n{horizon}.png"
                if path.is_file():
                    st.image(str(path), caption=f"{stem}（N={horizon}）", width="stretch")
    synthetic_path = report_path.parent / "synthetic_control_panel.png"
    if synthetic_path.is_file():
        st.subheader("合成对照面板")
        st.image(str(synthetic_path), caption="正对照与负对照", width="stretch")
    st.download_button("下载研究报告 Markdown", report_path.read_bytes(), file_name=report_path.name, mime="text/markdown")


def render_trend_tracking(*, data_path: str, scope_label: str) -> None:
    """渲染仅做统计检验的板块萌芽研究页面。"""
    st.title("板块萌芽现象 · 统计检验研究")
    st.caption("研究问题是萌芽分是否横截面预示随后板块强弱；本页面不包含交易、成本、净值或回撤。")
    try:
        config = load_research_config()
    except EmergenceStudyError as error:
        _render_error(error)
        return
    with st.sidebar:
        st.header("研究设置")
        st.caption("参数来自预注册配置，不能在页面上调参或网格搜索。")
        st.text_input("细粒度行业映射路径", value=str(config.mapping_path.relative_to(Path.cwd())), disabled=True)
        confirmed = st.checkbox("我确认日线价格已一致复权", value=False, key="emergence_adjustment_confirmed")
        submitted = st.button("运行统计检验研究", type="primary", width="stretch")
    if submitted:
        try:
            with st.status("正在运行合成对照闸门、宽表聚合、特征与统计检验…", expanded=True) as status:
                status.write("• 先运行正对照、负对照、置换与手算四关")
                result = run_real_study(data_path=data_path, adjustment_confirmed=confirmed, config=config)
                status.write("• 合成闸门已通过；真实数据仅用于横截面统计检验")
                directory = REPORTS_ROOT / "emergence_study"
                report_path = write_research_report(result, output_directory=directory)
                status.update(label="统计检验研究完成", state="complete", expanded=False)
            st.session_state["emergence_study_result"] = {"scope": str(Path(data_path).resolve()), "result": result, "report_path": str(report_path), "scope_label": scope_label}
        except EmergenceStudyError as error:
            _render_error(error)
        except Exception as error:
            st.error(f"emergence_study_unexpected_failure: 统计研究失败：{error}")
    state = st.session_state.get("emergence_study_result")
    if not isinstance(state, dict) or state.get("scope") != str(Path(data_path).resolve()):
        st.info("确认复权状态后，点击左侧“运行统计检验研究”。真实数据运行前会自动通过四道合成对照。")
        return
    result = state.get("result")
    report_path = state.get("report_path")
    if isinstance(result, RealStudyResult) and isinstance(report_path, str) and Path(report_path).is_file():
        _show_result(result, Path(report_path))
