"""Chinese Streamlit presentation for deterministic sector calibration."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rhein.data.discovery import input_files
from rhein.paths import CONFIG_ROOT
from rhein.trend_tracking import (
    TrendTrackingError,
    aggregate_sector_daily,
    calibrate_threshold_grid,
    clean_stock_daily,
    fixed_holding_diagnostics,
    load_industry_map,
    load_stock_daily,
    normalize_industry_map,
)


def _render_error(error: TrendTrackingError) -> None:
    st.error(f"{error.code}: {error.message}")
    if error.details:
        with st.expander("查看机器可读的错误详情"):
            st.json(error.details)


def _load_calibration_config() -> dict[str, object]:
    path = CONFIG_ROOT / "trend_tracking.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        calibration = config["calibration"]
        entries = calibration["entry_thresholds"]
        gaps = calibration["exit_gaps"]
        ratio = calibration["calibration_ratio"]
        cost = calibration["transaction_cost"]
        locked = config["locked_thresholds"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrendTrackingError("trend_config_invalid", "板块趋势追踪配置无效。", details={"path": str(path), "error": str(error)}) from error
    if locked.get("entry_threshold") is not None or locked.get("exit_threshold") is not None:
        raise TrendTrackingError("locked_thresholds_not_supported_here", "当前页面处于标定阶段，锁定阈值必须保持为空。")
    if not isinstance(entries, list) or not isinstance(gaps, list):
        raise TrendTrackingError("calibration_grid_invalid", "候选进场阈值和离场阈值差必须是列表。")
    return {"entry_thresholds": entries, "exit_gaps": gaps, "calibration_ratio": ratio, "transaction_cost": cost, "locked_status": locked.get("status")}


def _load_mapping(uploaded_mapping: object, local_mapping_path: str) -> pd.DataFrame:
    if uploaded_mapping is not None:
        try:
            return normalize_industry_map(pd.read_csv(uploaded_mapping))
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise TrendTrackingError("industry_map_read_failed", "上传的行业对照表无法解析。", details={"error": str(error)}) from error
    if local_mapping_path.strip():
        return load_industry_map(Path(local_mapping_path.strip()))
    raise TrendTrackingError("industry_map_required", "请上传或填写行业对照表路径后再运行。")


def render_trend_tracking(*, data_path: str, scope_label: str) -> None:
    """Render phase-five calibration only; daily execution stays unavailable."""
    st.title("板块趋势追踪 · 第五步网格标定")
    st.caption("先用前七年遍历候选阈值并选择稳定平台区；锁定值和后三年检验属于后续阶段。全程使用确定性 Python，不调用模型。")
    try:
        config = _load_calibration_config()
    except TrendTrackingError as error:
        _render_error(error)
        return
    entries = config["entry_thresholds"]
    gaps = config["exit_gaps"]
    candidate_count = len(entries) * len(gaps) if isinstance(entries, list) and isinstance(gaps, list) else 0
    st.info(f"本次会遍历 {candidate_count} 组：进场阈值 {entries}；离场阈值 = 进场阈值 − {gaps}。锁定阈值当前为未设置。", icon="ℹ️")
    with st.sidebar:
        st.header("板块标定设置")
        st.caption("此阶段不填写进场/离场常数；它们是网格标定与平台区选择的输出。")
        with st.form("trend_tracking_calibration"):
            uploaded_mapping = st.file_uploader("行业对照表 CSV（代码 + 行业板块）", type=["csv"], key="trend_tracking_industry_upload")
            local_mapping_path = st.text_input(
                "或填写本地行业对照表路径",
                value=st.session_state.get("trend_tracking_industry_path", "data/nasdaq_10y/industry_map.csv"),
                key="trend_tracking_industry_path",
                help="上传文件优先。支持列名：代码/行业板块，或 symbol/sector。",
            )
            adjustment_label = st.selectbox(
                "价格复权状态",
                ["已确认使用复权价", "未知：保守剔除单日涨跌幅超过 ±40% 的记录"],
                key="trend_tracking_adjustment_policy",
            )
            submitted = st.form_submit_button("运行前七年阈值网格标定", type="primary", width="stretch")
    if submitted:
        try:
            mapping = _load_mapping(uploaded_mapping, local_mapping_path)
            adjustment_policy = "adjusted" if adjustment_label == "已确认使用复权价" else "unknown_conservative_drop_outliers"
            paths = input_files(Path(data_path))
            with st.status("正在执行清洗、聚合、前七年 24 组回测与固定持有期诊断…", expanded=True) as status:
                status.write(f"• 读取 {len(paths)} 个日线文件")
                cleaned = clean_stock_daily(load_stock_daily(paths), mapping, adjustment_policy=adjustment_policy)
                sector_daily = aggregate_sector_daily(cleaned)
                grid = calibrate_threshold_grid(
                    cleaned,
                    sector_daily,
                    entry_thresholds=config["entry_thresholds"],  # type: ignore[arg-type]
                    exit_gaps=config["exit_gaps"],  # type: ignore[arg-type]
                    adjustment_policy=adjustment_policy,
                    transaction_cost=float(config["transaction_cost"]),
                    calibration_ratio=float(config["calibration_ratio"]),
                )
                status.write("• 前七年截断测试已通过：历史分数不受未来数据影响")
                fixed_holding = fixed_holding_diagnostics(
                    cleaned,
                    sector_daily,
                    entry_thresholds=config["entry_thresholds"],  # type: ignore[arg-type]
                    transaction_cost=float(config["transaction_cost"]),
                    calibration_ratio=float(config["calibration_ratio"]),
                )
                status.update(label="前七年阈值网格与固定持有期诊断完成", state="complete", expanded=False)
            st.session_state["trend_tracking_calibration_result"] = {
                "scope": str(Path(data_path).resolve()), "scope_label": scope_label,
                "grid": grid, "cleaned_rows": len(cleaned), "sector_rows": len(sector_daily),
                "adjustment_policy": adjustment_policy, "fixed_holding": fixed_holding,
            }
        except TrendTrackingError as error:
            _render_error(error)
        except Exception as error:
            st.error(f"trend_tracking_unexpected_failure: 网格标定失败：{error}")
    result = st.session_state.get("trend_tracking_calibration_result")
    if not isinstance(result, dict) or result.get("scope") != str(Path(data_path).resolve()):
        st.info("在左侧选择行业对照表与复权状态后，运行前七年阈值网格标定。")
        return
    grid = result.get("grid")
    if not isinstance(grid, pd.DataFrame):
        return
    st.subheader("前七年候选阈值网格")
    st.caption(
        f"数据范围：{result['scope_label']} · 清洗后 {result['cleaned_rows']:,} 条个股日线 · "
        f"{result['sector_rows']:,} 条板块日度记录。表中未使用后三年检验数据。"
    )
    st.dataframe(grid, hide_index=True, width="stretch")
    fixed_holding = result.get("fixed_holding")
    if isinstance(fixed_holding, pd.DataFrame):
        st.subheader("固定持有期对照（不使用评分离场）")
        st.caption("同一三日进场条件分别机械持有 20、40、60 个交易日。同板块持有期内的重复候选会忽略，样本末尾无法完成的候选不强制平仓；全表仍只使用前七年。")
        st.dataframe(fixed_holding, hide_index=True, width="stretch")
    st.warning("请从相邻组合表现相近的区域人工选择平台区。当前配置中的锁定阈值仍为 null；在选定组合前，日常筛选、完整回测和后三年检验均不可运行。")
    with st.expander("本次标定的机器可读设置"):
        st.json({
            "entry_thresholds": config["entry_thresholds"], "exit_gaps": config["exit_gaps"],
            "calibration_ratio": config["calibration_ratio"], "transaction_cost": config["transaction_cost"],
            "locked_thresholds": {"entry_threshold": None, "exit_threshold": None, "status": config["locked_status"]},
        })
