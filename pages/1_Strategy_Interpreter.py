"""Local Streamlit page for manually testing the Sprint 1 interpreter."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
from rhein.data.discovery import input_files
from rhein.ui.presets import available_data_scopes
from alpha_agent.domain.interpretation import ClarificationAnswer, StrategyInterpretationRequest
from alpha_agent.domain.interpretation import ParsedStrategy
from alpha_agent.errors import AlphaAgentError
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient
from alpha_agent.monitoring import InMemoryInterpreterMonitor
from alpha_agent.parser.review import build_review_items
from alpha_agent.parser.completeness import segment_source_clauses
from alpha_agent.parser.mock_timeline import build_mock_candle_timeline
from alpha_agent.parser.service import StrategyInterpreterService
from alpha_agent.interpretation_cache import JsonParsedInterpretationCache
from alpha_agent.strategy_library import JsonStrategyLibrary, SavedStrategy, StrategyLibraryError
from alpha_agent.backtest_history import BacktestHistoryError, JsonBacktestHistory, SavedBacktestRun, dataframe_records, recalculated_trade_level_kpis, strategy_fingerprint
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.research.legacy_adapter import compile_timed_strategy
from alpha_agent.research.v03_engine import run_v03_backtest
from alpha_agent.research.reporting import aggregate_gross_pnl
from rhein.backtest import input_files as backtest_input_files, load_ohlc as backtest_load_ohlc, run_backtest as legacy_run_backtest
from rhein.ui.result_runner import collect_results
from rhein.ui.gauges import kpi_gauge_html
from rhein.ui.summaries import style_by_drawdown
from rhein.ui.trade_chart import trade_selector_options
from rhein.ui.chart_theme import event_annotation_style
from rhein.ui.history_paths import portable_data_path, resolve_history_data_path, same_data_scope

# The project-local file is the explicit source of truth for this local app.
load_dotenv(ROOT / ".env", override=True)

st.set_page_config(page_title="自然语言策略解释器", layout="wide")
st.title("自然语言策略解释器")
st.caption("策略解释、逐条审阅与当前分组的确定性回测。解释覆盖关系与合成 K 线仅用于核对，不是市场数据。")
interpretation_notice = st.session_state.pop("interpretation_notice", None)
if interpretation_notice is not None:
    st.success(interpretation_notice)

if "interpreter_monitor" not in st.session_state: st.session_state.interpreter_monitor = InMemoryInterpreterMonitor()
if "interpreter_strategy_text" not in st.session_state:
    st.session_state.interpreter_strategy_text = "当收盘价向上穿越 20 日均线时买入；当它向下穿越该均线时卖出。"


def load_strategy_text(text: str) -> None:
    st.session_state.interpreter_strategy_text = text
    st.session_state.strategy_source = "新建策略"
    for key in ("active_saved_strategy_id", "active_saved_strategy_text", "active_saved_strategy_fingerprint"):
        st.session_state.pop(key, None)
    # A clarification belongs to one immutable source text. Never show it
    # after the user has explicitly switched to a different strategy.
    st.session_state.pop("pending_clarification", None)
    st.session_state.pop("pending_clarification_text", None)


def activate_saved_strategy(strategy: object, text: str, assumptions: object, warnings: object, source_clauses: object = (), coverage: object = (), strategy_id: object | None = None) -> None:
    st.session_state["last_parsed_strategy"] = ParsedStrategy(status="parsed", strategy=strategy, assumptions=assumptions, warnings=warnings, source_clauses=source_clauses, coverage=coverage)
    st.session_state["last_strategy_text"] = text
    st.session_state.interpreter_strategy_text = text
    if strategy_id is not None:
        st.session_state["active_saved_strategy_id"] = strategy_id
        st.session_state["active_saved_strategy_text"] = text
        st.session_state["active_saved_strategy_fingerprint"] = strategy_fingerprint(strategy.model_dump_json())  # type: ignore[union-attr]


def load_selected_saved_strategy() -> None:
    """Synchronize the selected library item without leaving its management view."""
    selected_id = st.session_state.get("strategy_source_saved")
    if not selected_id:
        return
    saved = next((item for item in library.list() if str(item.strategy_id) == selected_id), None)
    if saved is None:
        return
    if saved.strategy is not None:
        activate_saved_strategy(saved.strategy, saved.original_language, saved.assumptions, saved.warnings, saved.source_clauses, saved.coverage, saved.strategy_id)
    else:
        # Keep the user in the saved-strategy branch so a draft can still be
        # renamed or deleted. “继续编辑草稿” is the explicit branch change.
        st.session_state.interpreter_strategy_text = saved.original_language
        for key in ("active_saved_strategy_id", "active_saved_strategy_text", "active_saved_strategy_fingerprint"):
            st.session_state.pop(key, None)
    st.session_state["loaded_saved_strategy_id"] = selected_id


def delete_saved_strategy(strategy_id: object) -> None:
    """Delete from a Streamlit callback, before widgets are instantiated."""
    try:
        library.delete(strategy_id)  # type: ignore[arg-type]
        backtest_history.delete_for_strategy(strategy_id)  # type: ignore[arg-type]
    except (StrategyLibraryError, BacktestHistoryError) as error:
        st.session_state["strategy_library_notice"] = ("error", error.message)
        return
    for key in ("strategy_source_saved", "loaded_saved_strategy_id", "confirm_delete_saved_strategy", "active_saved_strategy_id", "active_saved_strategy_text", "active_saved_strategy_fingerprint"):
        st.session_state.pop(key, None)
    st.session_state.strategy_source = "新建策略"
    st.session_state["strategy_library_notice"] = ("success", "策略草稿已删除。")


def replace_backtest_view(
    *,
    kpis: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_text: str,
    view_scope: str,
    loaded_run: object | None,
) -> None:
    """Install one result set and reset only state scoped to its chart view.

    Loading a historical run must be equivalent to having just run it.  The
    selected Top-100 row and trade dropdown belong to a particular result set,
    so retaining those controls across a result change can point them at the
    wrong dataframe.  They are deliberately reset here; switching a trade
    *within* the same result set never calls this function.
    """
    st.session_state["dsl_backtest_kpis"] = kpis
    st.session_state["dsl_backtest_trades"] = trades
    st.session_state["dsl_backtest_strategy_text"] = strategy_text
    st.session_state["dsl_backtest_view_scope"] = view_scope
    if loaded_run is None:
        st.session_state.pop("dsl_backtest_loaded_run", None)
    else:
        st.session_state["dsl_backtest_loaded_run"] = loaded_run
    for key in (
        "dsl_symbols_table",
        "dsl_chart_result_scope",
        "dsl_chart_selected_symbol",
        "dsl_chart_trade_scope",
        "dsl_chart_trade_id",
    ):
        st.session_state.pop(key, None)


def render_error_details(details: dict[str, object]) -> None:
    """Show machine-readable details with an explicit browser-side copy action."""
    serialized = json.dumps(details, ensure_ascii=False, indent=2, default=str)
    safe_javascript_string = json.dumps(serialized).replace("</", "<\\/")
    st.code(serialized, language="json")
    components.html(
        f"""
        <button id="copy-error-details" type="button">复制错误详情</button>
        <span id="copy-status" aria-live="polite" style="margin-left: 0.5rem;"></span>
        <script>
          const details = {safe_javascript_string};
          const button = document.getElementById("copy-error-details");
          const status = document.getElementById("copy-status");
          button.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(details);
            }} catch (error) {{
              const area = document.createElement("textarea");
              area.value = details;
              document.body.appendChild(area);
              area.select();
              document.execCommand("copy");
              area.remove();
            }}
            status.textContent = "已复制";
          }});
        </script>
        """,
        height=42,
    )


def chart_event_points(trade: pd.Series) -> list[dict[str, object]]:
    """Read engine-provided point annotations, with a legacy three-point fallback."""
    raw_points = trade.get("event_points")
    if isinstance(raw_points, list):
        points = [
            {"event_id": str(point.get("event_id", point["label"])), "label": str(point["label"]), "date": str(point["date"]), "price": point.get("price")}
            for point in raw_points
            if isinstance(point, dict) and "label" in point and "date" in point
        ]
        if points:
            return points
    return [
        {"event_id": "anchor", "label": "t0 基准点", "date": str(trade["signal"]), "price": trade.get("entry_px")},
        {"event_id": "entry", "label": "入场", "date": str(trade["entry"]), "price": trade.get("entry_px")},
        {"event_id": "exit", "label": "出场", "date": str(trade["exit"]), "price": trade.get("exit_px")},
    ]


def anchor_check_rows(trade: pd.Series) -> list[dict[str, object]]:
    """Convert deterministic C/t0 condition snapshots into a review table."""
    raw_checks = trade.get("anchor_checks")
    if not isinstance(raw_checks, list):
        return []
    operator_labels = {"greater_than": ">", "less_than": "<", "greater_than_or_equal": "≥", "less_than_or_equal": "≤", "equal": "="}
    return [
        {
            "DSL 路径": str(check["dsl_path"]),
            "左侧": str(check["left"]),
            "左侧数值": check.get("left_value"),
            "比较": operator_labels.get(str(check["operator"]), str(check["operator"])),
            "右侧": str(check["right"]),
            "右侧数值": check.get("right_value"),
            "前一日左侧": check.get("prior_left_value"),
            "前一日右侧": check.get("prior_right_value"),
            "结果": "通过" if check.get("passed") else "未通过",
        }
        for check in raw_checks
        if isinstance(check, dict) and {"dsl_path", "left", "operator", "right"}.issubset(check)
    ]


library = JsonStrategyLibrary(ROOT / "config" / "saved_strategies.json")
backtest_history = JsonBacktestHistory(ROOT / "config" / "saved_backtest_results.json")
# Provider configuration is infrastructure, not a research-user control.
model = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
region = os.getenv("AWS_REGION", "us-east-1")

scope_options = available_data_scopes()
scope_labels = list(scope_options)
main_scope = st.session_state.get("data_scope")
default_scope_index = scope_labels.index(main_scope) if main_scope in scope_options else 0
selected_scope = st.sidebar.selectbox(
    "标的分组",
    scope_labels,
    index=default_scope_index,
    key="data_scope",
    help="与主回测页面共享同一个数据范围和 Nasdaq 分组选择。",
)
if selected_scope == "自定义路径":
    data_path = st.sidebar.text_input("自定义数据目录或 CSV", value="data", key="interpreter_custom_path")
else:
    data_path = scope_options[selected_scope]
    st.sidebar.caption(f"当前数据范围：`{data_path}`")
try:
    group_symbols = [path.stem.upper() for path in input_files(Path(data_path))]
except ValueError as error:
    st.sidebar.error(f"无法加载此分组的标的：{error}")
    group_symbols = []
if group_symbols:
    symbol = st.sidebar.selectbox(
        "标的",
        group_symbols,
        index=group_symbols.index("AAPL") if "AAPL" in group_symbols else 0,
        key="interpreter_symbol",
        help="仅显示当前所选组（或数据范围）中的标的。",
    )
else:
    symbol = None
st.sidebar.divider()
st.sidebar.subheader("策略")
library_notice = st.session_state.pop("strategy_library_notice", None)
if library_notice is not None:
    getattr(st.sidebar, library_notice[0])(library_notice[1])
strategy_source = st.sidebar.radio("策略来源", ["新建策略", "已保存策略"], horizontal=True, key="strategy_source")
if strategy_source == "新建策略":
    strategy_text = st.sidebar.text_area("自然语言策略", height=180, key="interpreter_strategy_text")
else:
    # Do not use ``interpreter_strategy_text`` as the selected strategy's
    # identity here.  That key belongs to the new-strategy text widget, which
    # Streamlit removes when the user switches to this saved-strategy branch.
    # On the next button/selectbox rerun it would be recreated with the sample
    # text, making a perfectly valid saved DSL look unavailable.
    strategy_text = ""
    try:
        source_strategies = library.list()
        saved_by_id = {str(item.strategy_id): item for item in source_strategies}
        if saved_by_id:
            source_id = st.sidebar.selectbox("选择已保存策略", list(saved_by_id), format_func=lambda item: f"{saved_by_id[item].strategy_name} · {'已验证' if saved_by_id[item].strategy else '草稿'}", key="strategy_source_saved", on_change=load_selected_saved_strategy)
            source_saved = saved_by_id[source_id]
            if st.session_state.get("loaded_saved_strategy_id") != source_id:
                load_selected_saved_strategy()
            # The persisted library text, rather than a conditionally-rendered
            # textarea widget, is the stable identity across every rerun
            # caused by loading a backtest or changing a chart dropdown.
            strategy_text = source_saved.original_language
            st.sidebar.text_area("策略全文", value=source_saved.original_language, height=180, disabled=True, key="saved_strategy_full_text")
            if source_saved.strategy is not None:
                st.sidebar.button("载入并使用", key="load_source_saved", on_click=activate_saved_strategy, args=(source_saved.strategy, source_saved.original_language, source_saved.assumptions, source_saved.warnings, source_saved.source_clauses, source_saved.coverage, source_saved.strategy_id), width="stretch")
            else:
                st.sidebar.button("继续编辑草稿", key="load_source_draft", on_click=load_strategy_text, args=(source_saved.original_language,), width="stretch")
            with st.sidebar.expander("管理此策略"):
                renamed = st.text_input("新名称", value=source_saved.strategy_name, key="rename_saved_strategy")
                if st.button("重命名", key="rename_saved_strategy_button"):
                    try:
                        library.rename(source_saved.strategy_id, renamed)
                        st.rerun()
                    except StrategyLibraryError as error:
                        st.error(error.message)
                confirm_delete = st.checkbox("我确认删除此策略", key="confirm_delete_saved_strategy")
                st.button("删除此策略", key="delete_saved_strategy", disabled=not confirm_delete, on_click=delete_saved_strategy, args=(source_saved.strategy_id,))
        else:
            st.sidebar.info("还没有已保存策略。")
    except StrategyLibraryError as error:
        st.sidebar.error(f"无法读取策略库：{error.message}")

last_result = st.session_state.get("last_parsed_strategy")
matching_result = last_result if st.session_state.get("last_strategy_text") == strategy_text else None
saved_backtest_strategy_id = None
current_strategy_fingerprint = None
current_backtest_view_scope = None
if matching_result is not None:
    current_fingerprint = strategy_fingerprint(matching_result.strategy.model_dump_json())
    current_strategy_fingerprint = current_fingerprint
    current_backtest_view_scope = f"{current_fingerprint}|{Path(data_path).resolve()}"
    if (
        st.session_state.get("active_saved_strategy_text") == strategy_text
        and st.session_state.get("active_saved_strategy_fingerprint") == current_fingerprint
        and st.session_state.get("active_saved_strategy_id") is not None
    ):
        saved_backtest_strategy_id = st.session_state["active_saved_strategy_id"]
if strategy_source == "新建策略":
    st.sidebar.subheader("保存与策略库")
    strategy_name = st.sidebar.text_input(
        "策略名称",
        value=(matching_result.strategy.strategy_name if matching_result is not None and matching_result.strategy.strategy_name else "未命名策略"),
        key="saved_strategy_name",
    )
    if matching_result is None:
        st.sidebar.caption("可先保存原文草稿；解析成功后会连同 DSL、映射和假设一并保存。")
    else:
        st.sidebar.success(f"可保存 · DSL v{matching_result.strategy.schema_version} · 包含原文与逐条映射")
    if st.sidebar.button("保存策略", type="secondary", key="save_current_strategy", width="stretch"):
        try:
            saved = library.save(SavedStrategy(
                strategy_name=strategy_name,
                original_language=strategy_text,
                strategy=matching_result.strategy if matching_result is not None else None,
                assumptions=matching_result.assumptions if matching_result is not None else [],
                warnings=matching_result.warnings if matching_result is not None else [],
                source_clauses=matching_result.source_clauses if matching_result is not None else [],
                coverage=matching_result.coverage if matching_result is not None else [],
            ))
            st.session_state["active_saved_strategy_id"] = saved.strategy_id
            st.session_state["active_saved_strategy_text"] = strategy_text
            if matching_result is not None:
                st.session_state["active_saved_strategy_fingerprint"] = strategy_fingerprint(matching_result.strategy.model_dump_json())
            st.sidebar.success(f"已保存跨组策略：{saved.strategy_name}")
        except StrategyLibraryError as error:
            st.sidebar.error(f"{error.code}: {error.message}")


def interpret_and_render(request: StrategyInterpretationRequest, source_text: str) -> None:
    model_client = BedrockStrategyModelClient(model=model, region=region)
    try:
        # This is intentionally a phase log rather than a fake percentage: a
        # provider call has variable latency, while the deterministic checks
        # are meaningful checkpoints users can audit.
        with st.status("正在准备策略解释…", expanded=True) as interpretation_status:
            def show_interpretation_progress(message: str) -> None:
                interpretation_status.write(f"• {message}")
                interpretation_status.update(label=message, state="running", expanded=True)

            service = StrategyInterpreterService(
                model_client,
                monitor=st.session_state.interpreter_monitor,
                inventory_client=model_client,
                parsed_cache=JsonParsedInterpretationCache(ROOT / ".cache" / "parsed_interpretations.json"),
                progress=show_interpretation_progress,
            )
            try:
                result = service.interpret(request)
            except Exception:
                interpretation_status.update(label="解释未完成：已停止在最后一个可见步骤。", state="error", expanded=True)
                raise
            interpretation_status.update(label="策略解释完成", state="complete", expanded=False)
        if result.status == "parsed":
            st.session_state["last_parsed_strategy"] = result
            st.session_state["last_strategy_text"] = source_text
            st.session_state["last_review_request_id"] = st.session_state.interpreter_monitor.events[-1].request_id
            st.session_state.pop("pending_clarification", None)
            st.session_state.pop("pending_clarification_text", None)
            st.session_state["interpretation_notice"] = "策略已解释完成。请在“解释审阅”中逐条核对原文与 DSL 映射。"
            # ``matching_result`` and ``pending_clarification`` were read
            # earlier in this Streamlit run. Restart from a coherent session
            # snapshot so a freshly parsed strategy immediately renders both
            # tabs; saving must never be a prerequisite for review/backtest.
            st.rerun()
        else:
            st.session_state["pending_clarification"] = result
            st.session_state["pending_clarification_text"] = source_text
            st.rerun()
    except AlphaAgentError as error:
        st.error(f"{error.code}: {error.message}")
        if error.code == "model_client_failure":
            st.info("这是模型服务或模型配置问题，不表示策略本身已被 DSL 拒绝。请查看下方详情中的 provider_message，并确认侧边栏模型对当前 API 项目可用。")
        elif error.code == "unsupported_strategy_feature":
            st.info("这不是模型连接问题：当前 DSL 尚没有保真表达这些条件所需的原语，因此系统没有发送 Bedrock 请求，也没有删除或猜测任何规则。")
        with st.expander("机器可读的错误详情"):
            render_error_details(error.details)


if strategy_source == "新建策略" and st.sidebar.button("解释策略", type="primary", width="stretch"):
    if not symbol:
        st.error("先选择一个包含有效 OHLC 数据的标的分组。")
    elif not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        st.error("未找到 API 密钥。请在项目根目录 `.env` 中设置 AWS_BEARER_TOKEN_BEDROCK 后重试。")
    else:
        interpret_and_render(StrategyInterpretationRequest(strategy_text=strategy_text, symbol=symbol), strategy_text)

pending_clarification = st.session_state.get("pending_clarification")
pending_text = st.session_state.get("pending_clarification_text")
if pending_clarification is not None and pending_text is not None:
    st.subheader("补充澄清信息")
    st.caption("回答会以问题 ID 关联原策略，并由同一个解释器再次处理；不会启动另一个澄清 Agent。")
    with st.form("clarification_answers"):
        answer_values: dict[str, str] = {}
        for question in pending_clarification.questions:
            st.markdown(f"**{question.question}**  ")
            if question.answer_kind == "choice":
                answer_values[question.question_id] = st.selectbox(
                    "建议答案", question.suggested_answers, key=f"clarify_{question.question_id}",
                )
            else:
                answer_values[question.question_id] = st.text_input(
                    "你的补充", key=f"clarify_{question.question_id}",
                )
        submitted_answers = st.form_submit_button("提交补充并重新解释", type="primary")
    if submitted_answers:
        if not symbol:
            st.error("先选择一个包含有效 OHLC 数据的标的分组。")
        elif any(not answer.strip() for answer in answer_values.values()):
            st.error("请回答全部澄清问题后再继续。")
        else:
            answers = [ClarificationAnswer(question_id=question_id, answer=answer.strip()) for question_id, answer in answer_values.items()]
            interpret_and_render(StrategyInterpretationRequest(strategy_text=pending_text, symbol=symbol, clarification_answers=answers), pending_text)

if matching_result is not None:
    backtest_tab, review_tab = st.tabs(["回测结果", "解释审阅"])
    review_tab.__enter__()
    st.subheader("策略解释映射")
    st.caption("请先核对每一段原文 C01…Cn 是否被正确落实到 DSL 路径与中文规则中；这比查看底层 JSON 更直接。")
    st.markdown("#### 原始策略")
    st.text_area("策略原文", value=strategy_text, height=150, disabled=True, label_visibility="collapsed", key="review_source_text")
    st.markdown("#### 条件逐条映射")
    st.caption("每个 C 编号都必须有落点。若某项无法安全映射，解释器应先提问，而不是遗漏或猜测。")
    coverage_by_id = {item.clause_id: item for item in matching_result.coverage}
    source_clauses = matching_result.source_clauses or segment_source_clauses(strategy_text)
    coverage_rows = []
    for clause in source_clauses:
        item = coverage_by_id.get(clause.clause_id)
        coverage_rows.append({
            "条件": clause.clause_id,
            "原文": clause.text,
            "处理": {"mapped": "已映射", "assumption": "明确假设", "clarification_required": "需要澄清", "unsupported": "当前不支持"}.get(item.disposition, "缺少覆盖记录") if item else "缺少覆盖记录",
            "DSL 路径": "；".join(item.dsl_paths) if item else "—",
            "说明": item.explanation if item else "此为旧版已保存策略；请重新解释以生成覆盖记录。",
        })
    st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, width="stretch")
    with st.expander("查看 DSL 规则的中文展开", expanded=True):
        for item in build_review_items(matching_result.strategy):
            st.markdown(f"**{item.title}** · `{item.dsl_path}`  ")
            st.write(item.explanation)
        if matching_result.assumptions:
            st.caption("明确假设")
            for note in matching_result.assumptions:
                st.write(f"- {note.message}")
        if matching_result.warnings:
            st.caption("提示")
            for note in matching_result.warnings:
                st.write(f"- {note.message}")
    st.markdown("#### 条件时序示意（合成 K 线）")
    st.caption("这是一段不用于回测的示意价格路径。图上每一个 C 标签与上表一一对应，帮助核对 t0 / t1 / t2… 的条件落点。")
    candles, markers = build_mock_candle_timeline(source_clauses, matching_result.coverage)
    candle_x = {"field": "day", "type": "quantitative", "axis": {"title": "相对基准日（t0 = 0）", "tickMinStep": 1}}
    mock_spec = {
        "height": 350,
        "layer": [
            {"mark": "rule", "encoding": {"x": candle_x, "y": {"field": "low", "type": "quantitative", "scale": {"zero": False}}, "y2": {"field": "high"}}},
            {"mark": {"type": "bar", "size": 10}, "encoding": {"x": candle_x, "y": {"field": "open", "type": "quantitative", "scale": {"zero": False}}, "y2": {"field": "close"}, "color": {"condition": {"test": "datum.close >= datum.open", "value": "#198754"}, "value": "#d62728"}}},
            {"transform": [{"fold": ["ma5", "ma10"], "as": ["均线", "价格"]}], "mark": {"type": "line", "strokeWidth": 2}, "encoding": {"x": candle_x, "y": {"field": "价格", "type": "quantitative", "scale": {"zero": False}}, "color": {"field": "均线", "type": "nominal", "scale": {"range": ["#2563eb", "#f59e0b"]}}}},
            {"data": {"values": markers.to_dict(orient="records")}, "mark": {"type": "point", "filled": True, "size": 70}, "encoding": {"x": candle_x, "y": {"field": "price", "type": "quantitative"}, "color": {"field": "disposition", "type": "nominal", "scale": {"domain": ["mapped", "assumption", "clarification_required", "unsupported"], "range": ["#2563eb", "#f59e0b", "#dc2626", "#6b7280"]}, "legend": {"title": "处理状态"}}, "tooltip": [{"field": "label", "title": "条件"}, {"field": "day", "title": "相对日"}, {"field": "disposition", "title": "处理"}]}},
            {"data": {"values": markers.to_dict(orient="records")}, "mark": {"type": "text", "dy": -10, "fontWeight": "bold"}, "encoding": {"x": candle_x, "y": {"field": "price", "type": "quantitative"}, "text": {"field": "label"}}},
        ],
    }
    st.vega_lite_chart(candles, mock_spec, width="stretch", key="interpretation_mock_timeline")
    with st.expander("解释器监控（当前浏览器会话）"):
        summary = st.session_state.interpreter_monitor.summary()
        metrics = st.columns(4)
        metrics[0].metric("请求数", summary["total_requests"])
        metrics[1].metric("已解析率", f"{summary['parsed_rate']:.0%}")
        metrics[2].metric("澄清率", f"{summary['clarification_rate']:.0%}")
        metrics[3].metric("失败率", f"{summary['failure_rate']:.0%}")
        st.caption("该监控只反映本浏览器会话的运行状况；它不替代上方逐条映射核对。")
    review_tab.__exit__(None, None, None)

    st.sidebar.divider()
    st.sidebar.subheader("当前已确认策略")
    st.sidebar.success(f"已验证 · DSL v{matching_result.strategy.schema_version}")
    if strategy_source == "新建策略":
        st.sidebar.text_area("策略全文", value=strategy_text, height=180, disabled=True, key="confirmed_strategy_full_text")
    backtest_tab.__enter__()
    if saved_backtest_strategy_id is not None:
        current_group_path = str(Path(data_path).resolve())
        try:
            all_saved_runs = [
                run for run in backtest_history.list_for_strategy(saved_backtest_strategy_id)
                if same_data_scope(run.data_path, current_group_path, project_root=ROOT)
            ]
            # A previous engine-wide failure may have created an empty record
            # before the UI could surface its per-file errors.  That is not a
            # usable research result (unlike a normal zero-trade run, which
            # still has one KPI row per symbol), so do not offer it for load.
            saved_runs = [run for run in all_saved_runs if run.kpis]
        except BacktestHistoryError as error:
            st.warning(f"无法读取此策略的已保存回测：{error.message}")
            saved_runs = []
        if saved_runs:
            st.caption(f"此策略在当前组已有 {len(saved_runs)} 次已保存回测。可载入历史结果，不会重新调用模型或回测引擎。")
            run_by_id = {str(run.run_id): run for run in saved_runs}
            run_id = st.selectbox(
                "已保存回测",
                list(run_by_id),
                format_func=lambda item: f"{run_by_id[item].created_at.astimezone().strftime('%Y-%m-%d %H:%M')} · {run_by_id[item].input_file_count} 个标的 · {len(run_by_id[item].trades)} 笔交易",
                key="saved_backtest_run",
            )
            if st.button("载入这次已保存回测", key="load_saved_backtest", width="stretch"):
                selected_run = run_by_id[run_id]
                assert current_backtest_view_scope is not None
                replace_backtest_view(
                    kpis=recalculated_trade_level_kpis(selected_run.kpis, selected_run.trades, selected_run.settings),
                    trades=pd.DataFrame(selected_run.trades),
                    strategy_text=strategy_text,
                    view_scope=current_backtest_view_scope,
                    loaded_run=selected_run,
                )
                st.rerun()
        else:
            if 'all_saved_runs' in locals() and all_saved_runs:
                st.warning("已忽略一条没有任何标的 KPI 的失败回测记录；请重新运行当前组回测。")
            st.caption("此已保存策略尚未在当前数据组保存过回测。运行后会自动建立第一条记录。")
    else:
        st.caption("当前是未保存策略或已修改版本：仍可临时回测；先保存策略后，结果才会按“策略 × 数据组”持久化。")
    st.subheader("研究设置")
    st.caption("v0.3 的此类策略只按日线收盘价成交；旧 v0.2 的日内触发才会采用当日 Low 的保守近似。")
    initial_capital = st.number_input("初始资金", min_value=1_000.0, value=10_000.0, step=1_000.0, key="dsl_initial_capital")
    cost_bps = st.number_input("单边费用（bps）", min_value=0.0, value=0.0, step=0.5, key="dsl_cost_bps")
    if st.button("运行当前组回测", type="primary", key="run_dsl_backtest"):
        if matching_result.strategy.schema_version == "0.1":
            st.error("当前仅可回测时序 DSL（v0.2/v0.3）；静态 v0.1 通用执行器将在下一步接入。")
        else:
            try:
                timed_strategy = TimedStrategyDefinition.model_validate(matching_result.strategy.model_dump(mode="json"))
                if timed_strategy.schema_version == "0.3":
                    params, runner, label = {"strategy": timed_strategy, "cost_bps": cost_bps}, run_v03_backtest, "DSL v0.3 分组回测"
                else:
                    params, runner, label = compile_timed_strategy(timed_strategy, approximate_intraday_with_daily_low=True), legacy_run_backtest, "DSL v0.2 分组回测"
                    params["cost_bps"] = cost_bps
                paths = backtest_input_files(Path(data_path))
                kpis, trades = collect_results(paths, params, initial_capital, True, load_ohlc=backtest_load_ohlc, run_backtest=runner, progress_label=label)
                if kpis.empty:
                    st.warning("本次所有标的均运行失败，未保存为空白回测记录；请检查上方逐文件错误后重试。")
                    raise BacktestHistoryError("all input files failed; no empty backtest result was saved")
                assert current_backtest_view_scope is not None
                replace_backtest_view(
                    kpis=kpis,
                    trades=trades,
                    strategy_text=strategy_text,
                    view_scope=current_backtest_view_scope,
                    loaded_run=None,
                )
                if saved_backtest_strategy_id is not None:
                    saved_run = SavedBacktestRun(
                        strategy_id=saved_backtest_strategy_id,
                        strategy_fingerprint=strategy_fingerprint(timed_strategy.model_dump_json()),
                        schema_version=timed_strategy.schema_version,
                        group_label=selected_scope,
                        data_path=portable_data_path(data_path, project_root=ROOT),
                        input_file_count=len(paths),
                        settings={"initial_capital": float(initial_capital), "compound": True, "cost_bps": float(cost_bps), "runner": label},
                        kpis=dataframe_records(kpis),
                        trades=dataframe_records(trades),
                    )
                    backtest_history.save(saved_run)
                    st.success(f"回测已保存：{selected_scope} · {saved_run.run_id}")
                else:
                    st.info("本次为临时回测。保存当前策略后再次运行，即可持久保存该数据组的结果。")
            except (AlphaAgentError, BacktestHistoryError) as error:
                st.error(f"{error.code}: {error.message}")
            except Exception as error:
                st.error(f"回测失败：{error}")

    if st.session_state.get("dsl_backtest_view_scope") == current_backtest_view_scope:
        st.subheader("回测结果")
        loaded_run = st.session_state.get("dsl_backtest_loaded_run")
        if loaded_run is not None:
            st.caption(f"当前展示已保存运行：{loaded_run.created_at.astimezone().strftime('%Y-%m-%d %H:%M')} · {loaded_run.group_label} · 设置 {dict(loaded_run.settings)}")
        kpis = st.session_state.get("dsl_backtest_kpis", pd.DataFrame())
        if not kpis.empty:
            active = kpis[kpis["n_trades"] > 0]
            limit = 15.0
            gauges = st.columns(3)
            gauges[0].html(kpi_gauge_html("sharpe", active["sharpe_ratio"].median() if len(active) else None, max_drawdown_limit=limit))
            gauges[1].html(kpi_gauge_html("annualized_return", active["annualized_return_pct"].median() if len(active) else None, max_drawdown_limit=limit))
            gauges[2].html(kpi_gauge_html("drawdown", active["max_drawdown_pct"].median() if len(active) else None, max_drawdown_limit=limit))
            gross_profit, gross_loss = aggregate_gross_pnl(
                kpis,
                st.session_state.get("dsl_backtest_trades", pd.DataFrame()),
            )
            overview_metrics = [
                ("标的数", len(kpis)),
                ("总交易数", int(kpis["n_trades"].sum())),
                ("有交易标的", len(active)),
                ("平均标的胜率", f"{active['win_rate_pct'].mean():.2f}%" if len(active) else "不适用"),
                ("合并盈利因子", f"{gross_profit / gross_loss:.3f}" if gross_loss else "不适用"),
                ("中位标的累计收益", f"{active['cumulative_return_pct'].median():.2f}%" if len(active) else "不适用"),
            ]
            for metric_row in (overview_metrics[:3], overview_metrics[3:]):
                columns = st.columns(3)
                for column, (label, value) in zip(columns, metric_row):
                    column.metric(label, value)
            open_position_rows: list[dict[str, object]] = []
            if "open_positions" in kpis:
                for _, kpi_row in kpis.iterrows():
                    positions = kpi_row.get("open_positions")
                    if not isinstance(positions, list):
                        continue
                    for position in positions:
                        if isinstance(position, dict):
                            open_position_rows.append({"标的": kpi_row["标的"], **position})
            if open_position_rows:
                with st.expander(f"样本末尾未平仓头寸（{len(open_position_rows)}）", expanded=False):
                    st.caption("这些头寸按策略要求保留，未计入已平仓交易 KPI 或全部标的表中的交易统计。")
                    st.dataframe(pd.DataFrame(open_position_rows), hide_index=True, width="stretch")
            st.subheader(f"全部标的（{len(kpis)}）")
            options = {"盈利因子（高→低）": "profit_factor", "累计收益率（高→低）": "cumulative_return_pct", "年化收益率（高→低）": "annualized_return_pct", "夏普比率（高→低）": "sharpe_ratio", "胜率（高→低）": "win_rate_pct", "最大回撤（低→高）": "max_drawdown_pct"}
            rank_label = st.selectbox("排序指标", list(options), key="dsl_rank_metric")
            min_trades = st.number_input("最少交易次数", min_value=0, value=0, step=1, key="dsl_min_trades")
            metric = options[rank_label]
            ranked = kpis[kpis["n_trades"] >= min_trades].sort_values(
                metric,
                ascending=metric == "max_drawdown_pct",
                na_position="last",
            )
            display = ranked[["标的", "n_trades", "sharpe_ratio", "annualized_return_pct", "max_drawdown_pct", "win_rate_pct", "cumulative_return_pct", "payoff_ratio", "profit_factor", "avg_return_pct", "avg_days_held"]].rename(columns={"n_trades":"交易次数", "sharpe_ratio":"夏普比率", "annualized_return_pct":"全样本年化收益率 (%)", "max_drawdown_pct":"最大回撤 (%)", "win_rate_pct":"胜率 (%)", "cumulative_return_pct":"累计收益率 (%)", "payoff_ratio":"盈亏比", "profit_factor":"盈利因子", "avg_return_pct":"平均单笔收益 (%)", "avg_days_held":"平均持仓天数"})
            st.caption(f"展示当前分组全部 {len(display)} 个标的。绿色行符合回撤阈值；红色行超过阈值。选择行后可查看该标的交易。")
            selection = st.dataframe(style_by_drawdown(display, "最大回撤 (%)", -limit, integer_columns=("交易次数",)), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="dsl_symbols_table")
            rows = selection.selection.rows if selection else []
            selected_symbol = None
            if rows:
                selected_symbol = str(display.iloc[rows[0]]["标的"])
                st.session_state["dsl_chart_result_scope"] = current_backtest_view_scope
                st.session_state["dsl_chart_selected_symbol"] = selected_symbol
            elif (
                st.session_state.get("dsl_chart_result_scope") == current_backtest_view_scope
                and st.session_state.get("dsl_chart_selected_symbol") in set(display["标的"].astype(str))
            ):
                # Selectbox interactions rerun the page but do not necessarily
                # preserve a dataframe's transient row-selection payload.
                # Keep showing the chart for the symbol chosen in this exact
                # result set instead of making the user reload the strategy.
                selected_symbol = str(st.session_state["dsl_chart_selected_symbol"])
            if selected_symbol is not None:
                symbol_trades = st.session_state.get("dsl_backtest_trades", pd.DataFrame())
                symbol_trades = symbol_trades[symbol_trades["symbol"] == selected_symbol].reset_index(drop=True)
                if symbol_trades.empty:
                    st.info(f"{selected_symbol} 没有可展示的已平仓交易。")
                else:
                    st.subheader(f"{selected_symbol}：交易 K 线")
                    trade_option_ids, trade_labels = trade_selector_options(symbol_trades)
                    # A table click reruns the page. Scope the dropdown to the
                    # selected symbol plus its stable trade IDs so that a choice
                    # from the previous symbol can never index the new table.
                    trade_scope = f"{selected_symbol}::{'||'.join(trade_option_ids)}"
                    if st.session_state.get("dsl_chart_trade_scope") != trade_scope:
                        st.session_state["dsl_chart_trade_scope"] = trade_scope
                        st.session_state["dsl_chart_trade_id"] = trade_option_ids[0]
                    elif st.session_state.get("dsl_chart_trade_id") not in trade_labels:
                        st.session_state["dsl_chart_trade_id"] = trade_option_ids[0]
                    trade_id = st.selectbox("选择交易段", trade_option_ids, format_func=trade_labels.__getitem__, key="dsl_chart_trade_id")
                    trade = symbol_trades.iloc[trade_option_ids.index(trade_id)]
                    source_matches = kpis.loc[kpis["标的"] == selected_symbol, "源文件"]
                    source_path = (
                        resolve_history_data_path(source_matches.iloc[0], project_root=ROOT)
                        if not source_matches.empty
                        else None
                    )
                    if source_path is None:
                        st.warning(f"找不到 {selected_symbol} 的回测源文件，无法绘制 K 线。")
                    else:
                        chart = backtest_load_ohlc(source_path).copy()
                        # Display-only overlays are calculated from the same
                        # source OHLC data. They never alter stored trades or
                        # trigger a group backtest rerun.
                        for window in (5, 10, 20, 60):
                            chart[f"MA{window}"] = chart["Close"].rolling(window).mean()
                        signal, entry, exit_ = (pd.Timestamp(trade[column]).normalize() for column in ("signal", "entry", "exit"))
                        chart_dates = pd.to_datetime(chart["Date"], errors="coerce").dt.normalize()
                        signal_positions = chart.index[chart_dates == signal]
                        exit_positions = chart.index[chart_dates == exit_]
                        if not len(signal_positions) or not len(exit_positions):
                            st.warning("这笔交易的日期不在当前标的的 OHLC 数据中，无法安全绘制 K 线；回测结果未被修改。")
                        else:
                            event_points = chart_event_points(trade)
                            event_positions: list[int] = []
                            for event_point in event_points:
                                positions = chart.index[chart_dates == pd.Timestamp(event_point["date"]).normalize()]
                                if len(positions):
                                    event_positions.append(int(positions[0]))
                            # Include every recorded structural event (not just
                            # t0/entry/exit), and retain one hundred completed
                            # trading candles after the exit whenever the source
                            # contains them.  Then widen to at least 100
                            # consecutive trading candles when the source has
                            # enough history. `Index` is later drawn on a
                            # compact ordinal x-axis, so weekends and holidays
                            # never create gaps.
                            important_positions = event_positions or [int(signal_positions[0]), int(exit_positions[0])]
                            start = max(0, min(important_positions) - 15)
                            post_exit_candles = 100
                            end = min(
                                len(chart),
                                max(max(important_positions) + 1, int(exit_positions[0]) + 1 + post_exit_candles),
                            )
                            minimum_candles = min(100, len(chart))
                            if end - start < minimum_candles:
                                missing = minimum_candles - (end - start)
                                expand_left = min(start, missing // 2)
                                start -= expand_left
                                end = min(len(chart), end + (missing - expand_left))
                                start = max(0, end - minimum_candles)
                            chart = chart.iloc[start:end].copy().reset_index(drop=True)
                            chart["Index"] = range(len(chart))
                            marker_labels: dict[int, list[str]] = {}
                            marker_prices: dict[int, list[float]] = {}
                            chart_dates_for_markers = pd.to_datetime(chart["Date"], errors="coerce").dt.normalize()
                            audit_rows: list[dict[str, object]] = []
                            for event_point in event_points:
                                date = pd.Timestamp(event_point["date"]).normalize()
                                positions = chart.index[chart_dates_for_markers == date]
                                if len(positions):
                                    position = int(positions[0])
                                    marker_labels.setdefault(position, []).append(event_point["label"])
                                    price = event_point.get("price")
                                    event_price = float(price) if isinstance(price, (int, float)) else float(chart.loc[position, "Close"])
                                    marker_prices.setdefault(position, []).append(event_price)
                                    audit_rows.append({
                                        "事件": event_point["label"],
                                        "日期": date.strftime("%Y-%m-%d"),
                                        "记录价格": round(event_price, 4),
                                        "图内交易日": position + 1,
                                    })
                            price_span = max(float(chart["High"].max() - chart["Low"].min()), 0.01)
                            label_offset = max(price_span * 0.04, float(chart["High"].max()) * 0.004)
                            price_scale = {
                                "zero": False,
                                "nice": False,
                                "domain": [
                                    float(chart["Low"].min()) - price_span * 0.02,
                                    float(chart["High"].max()) + label_offset * 4,
                                ],
                            }
                            markers = [
                                {
                                    "Index": position,
                                    "EventPrice": marker_prices[position][0],
                                    "LabelPrice": float(chart.loc[position, "High"]) + label_offset * (1 + position % 3 * 0.7),
                                    "Label": " · ".join(labels),
                                    "ShortLabel": " / ".join(str(label).split("：", 1)[0] for label in labels),
                                    "Date": pd.Timestamp(chart.loc[position, "Date"]).strftime("%Y-%m-%d"),
                                }
                                for position, labels in marker_labels.items()
                            ]
                            chart["Date"] = pd.to_datetime(chart["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                            # A tightly packed ordinal date band gives each
                            # trading day its own slot while showing readable
                            # calendar labels and removing weekend/holiday gaps.
                            # Every layer shares this band, so event arrows
                            # keep pointing to their exact candle.
                            x = {
                                "field": "Date",
                                "type": "ordinal",
                                "scale": {"paddingInner": 0.06, "paddingOuter": 0.01},
                                "axis": {
                                    "title": "交易日期",
                                    "tickCount": 12,
                                    "labelAngle": -40,
                                    "labelOverlap": "greedy",
                                    "labelLimit": 90,
                                },
                            }
                            ohlc_tooltip = [
                                {"field":"Date","type":"nominal","title":"日期"},
                                {"field":"Open","type":"quantitative","title":"开盘","format":".4f"},
                                {"field":"High","type":"quantitative","title":"最高","format":".4f"},
                                {"field":"Low","type":"quantitative","title":"最低","format":".4f"},
                                {"field":"Close","type":"quantitative","title":"收盘","format":".4f"},
                                {"field":"Volume","type":"quantitative","title":"成交量","format":",.0f"},
                                {"field":"Index","type":"quantitative","title":"图内第几根","format":"d"},
                            ]
                            event_tooltip = [
                                {"field":"Label","type":"nominal","title":"事件"},
                                {"field":"Date","type":"nominal","title":"日期"},
                                {"field":"EventPrice","type":"quantitative","title":"事件价格","format":".4f"},
                                {"field":"Index","type":"quantitative","title":"图内零基索引","format":"d"},
                            ]
                            annotation_style = event_annotation_style(st.context.theme.type)
                            event_color = annotation_style["color"]
                            spec = {
                                "vconcat": [
                                    {
                                        "height": 360,
                                        "layer": [
                                            {"mark": "rule", "encoding": {"x": x, "y": {"field": "Low", "type": "quantitative", "scale": {"zero": False}}, "y2": {"field": "High"}, "tooltip": ohlc_tooltip}},
                                            {"mark": {"type": "bar"}, "encoding": {"x": x, "y": {"field": "Open", "type": "quantitative", "scale": {"zero": False}}, "y2": {"field": "Close"}, "color": {"condition": {"test": "datum.Close >= datum.Open", "value": "#198754"}, "value": "#d62728"}, "tooltip": ohlc_tooltip}},
                                            {"transform": [{"fold": ["MA5", "MA10", "MA20"], "as": ["MA", "Value"]}], "mark": {"type": "line", "strokeWidth": 1.8}, "encoding": {"x": x, "y": {"field": "Value", "type": "quantitative", "scale": {"zero": False}}, "color": {"field": "MA", "type": "nominal", "scale": {"domain": ["MA5", "MA10", "MA20"], "range": ["#2563eb", "#f59e0b", "#dc2626"]}, "legend": {"title": "均线"}}}},
                                            {"transform": [{"calculate": "'MA60（60日）'", "as": "MA"}], "mark": {"type": "line", "stroke": "#334155", "strokeWidth": 3, "strokeDash": [7, 3]}, "encoding": {"x": x, "y": {"field": "MA60", "type": "quantitative", "scale": {"zero": False}}, "color": {"field": "MA", "type": "nominal", "scale": {"domain": ["MA5", "MA10", "MA20", "MA60（60日）"], "range": ["#2563eb", "#f59e0b", "#dc2626", "#334155"]}, "legend": {"title": "均线"}}}},
                                            {"data": {"values": markers}, "mark": {"type": "rule", "color": event_color, "strokeWidth": 1.2}, "encoding": {"x": x, "y": {"field": "EventPrice", "type": "quantitative"}, "y2": {"field": "LabelPrice"}, "tooltip": event_tooltip}},
                                            {"data": {"values": markers}, "mark": {"type": "point", "filled": True, "size": 90, "color": event_color}, "encoding": {"x": x, "y": {"field": "EventPrice", "type": "quantitative"}, "tooltip": event_tooltip}},
                                            {"data": {"values": markers}, "mark": {"type": "text", "fontWeight": "bold", "color": event_color}, "encoding": {"x": x, "y": {"field": "LabelPrice", "type": "quantitative"}, "text": {"field": "ShortLabel"}, "tooltip": event_tooltip}},
                                        ],
                                    },
                                    {"height": 100, "mark": {"type": "bar"}, "encoding": {"x": x, "y": {"field": "Volume", "type": "quantitative"}, "color": {"condition": {"test": "datum.Close >= datum.Open", "value": "#198754"}, "value": "#d62728"}, "tooltip": ohlc_tooltip}},
                                ],
                            }
                            price_layers = spec["vconcat"][0]["layer"]
                            for layer in price_layers:
                                y_encoding = layer["encoding"].get("y")
                                if isinstance(y_encoding, dict):
                                    y_encoding["scale"] = price_scale
                            text_mark = price_layers[-1]["mark"]
                            text_mark.update({"fontSize": 13, "stroke": annotation_style["halo_color"], "strokeWidth": annotation_style["halo_width"], "align": "left", "dx": 4, "baseline": "bottom"})
                            price_layers[-1]["encoding"]["text"]["type"] = "nominal"
                            st.vega_lite_chart(chart, spec, width="stretch", key=f"dsl_chart_{selected_symbol}_{trade_id}")
                            checks = anchor_check_rows(trade)
                            if checks:
                                with st.expander("C 点入场条件核对", expanded=True):
                                    st.caption("这是一笔实际被接受的 C/t0 的逐项计算快照；全部条件必须为“通过”。")
                                    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
                            if audit_rows:
                                with st.expander("标注核对", expanded=False):
                                    st.caption("A/B/C 等点直接来自本笔交易的确定性回测 artifact；A/B 均按 DSL 指定的最早并列规则计算。")
                                    st.dataframe(pd.DataFrame(audit_rows), hide_index=True, width="stretch")
        else:
            st.warning("该分组没有生成已平仓交易。")
    else:
        st.info("在左侧点击“运行当前组回测”后，这里会显示 KPI 概览、全部标的和交易 K 线。")
    backtest_tab.__exit__(None, None, None)
else:
    st.info("在左侧选择或输入策略，完成解释后即可查看策略与 DSL，并运行当前分组回测。")
