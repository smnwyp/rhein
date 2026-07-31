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
from alpha_agent.strategy_library import JsonStrategyLibrary, SavedStrategy, StrategyLibraryError
from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.research.legacy_adapter import compile_timed_strategy
from rhein.backtest import input_files as backtest_input_files, load_ohlc as backtest_load_ohlc, run_backtest as legacy_run_backtest
from rhein.ui.result_runner import collect_results
from rhein.ui.gauges import kpi_gauge_html
from rhein.ui.summaries import style_by_drawdown

# The project-local file is the explicit source of truth for this local app.
load_dotenv(ROOT / ".env", override=True)

st.set_page_config(page_title="自然语言策略解释器", layout="wide")
st.title("自然语言策略解释器")
st.caption("策略解释、逐条审阅与当前分组的确定性回测。解释覆盖关系与合成 K 线仅用于核对，不是市场数据。")

if "interpreter_monitor" not in st.session_state: st.session_state.interpreter_monitor = InMemoryInterpreterMonitor()
if "interpreter_strategy_text" not in st.session_state:
    st.session_state.interpreter_strategy_text = "当收盘价向上穿越 20 日均线时买入；当它向下穿越该均线时卖出。"


def load_strategy_text(text: str) -> None:
    st.session_state.interpreter_strategy_text = text
    st.session_state.strategy_source = "新建策略"


def activate_saved_strategy(strategy: object, text: str, assumptions: object, warnings: object, source_clauses: object = (), coverage: object = ()) -> None:
    st.session_state["last_parsed_strategy"] = ParsedStrategy(status="parsed", strategy=strategy, assumptions=assumptions, warnings=warnings, source_clauses=source_clauses, coverage=coverage)
    st.session_state["last_strategy_text"] = text
    st.session_state.interpreter_strategy_text = text


def load_selected_saved_strategy() -> None:
    """Selecting a library item is an explicit load action, not merely a preview."""
    selected_id = st.session_state.get("strategy_source_saved")
    if not selected_id:
        return
    saved = next((item for item in library.list() if str(item.strategy_id) == selected_id), None)
    if saved is None:
        return
    if saved.strategy is not None:
        activate_saved_strategy(saved.strategy, saved.original_language, saved.assumptions, saved.warnings, saved.source_clauses, saved.coverage)
    else:
        load_strategy_text(saved.original_language)
    st.session_state["loaded_saved_strategy_id"] = selected_id


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


library = JsonStrategyLibrary(ROOT / "config" / "saved_strategies.json")
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
strategy_source = st.sidebar.radio("策略来源", ["新建策略", "已保存策略"], horizontal=True, key="strategy_source")
if strategy_source == "新建策略":
    strategy_text = st.sidebar.text_area("自然语言策略", height=180, key="interpreter_strategy_text")
else:
    strategy_text = st.session_state.interpreter_strategy_text
    try:
        source_strategies = library.list()
        saved_by_id = {str(item.strategy_id): item for item in source_strategies}
        if saved_by_id:
            source_id = st.sidebar.selectbox("选择已保存策略", list(saved_by_id), format_func=lambda item: f"{saved_by_id[item].strategy_name} · {'已验证' if saved_by_id[item].strategy else '草稿'}", key="strategy_source_saved", on_change=load_selected_saved_strategy)
            source_saved = saved_by_id[source_id]
            if st.session_state.get("loaded_saved_strategy_id") != source_id:
                load_selected_saved_strategy()
            st.sidebar.text_area("策略全文", value=source_saved.original_language, height=180, disabled=True, key="saved_strategy_full_text")
            if source_saved.strategy is not None:
                st.sidebar.button("载入并使用", key="load_source_saved", on_click=activate_saved_strategy, args=(source_saved.strategy, source_saved.original_language, source_saved.assumptions, source_saved.warnings, source_saved.source_clauses, source_saved.coverage), width="stretch")
            else:
                st.sidebar.button("载入后继续解释", key="load_source_draft", on_click=load_strategy_text, args=(source_saved.original_language,), width="stretch")
            with st.sidebar.expander("管理此策略"):
                renamed = st.text_input("新名称", value=source_saved.strategy_name, key="rename_saved_strategy")
                if st.button("重命名", key="rename_saved_strategy_button"):
                    try:
                        library.rename(source_saved.strategy_id, renamed)
                        st.rerun()
                    except StrategyLibraryError as error:
                        st.error(error.message)
                confirm_delete = st.checkbox("我确认删除此策略", key="confirm_delete_saved_strategy")
                if st.button("删除此策略", key="delete_saved_strategy", disabled=not confirm_delete):
                    try:
                        library.delete(source_saved.strategy_id)
                        st.session_state.strategy_source = "新建策略"
                        st.rerun()
                    except StrategyLibraryError as error:
                        st.error(error.message)
        else:
            st.sidebar.info("还没有已保存策略。")
    except StrategyLibraryError as error:
        st.sidebar.error(f"无法读取策略库：{error.message}")


def interpret_and_render(request: StrategyInterpretationRequest, source_text: str) -> None:
    service = StrategyInterpreterService(
        BedrockStrategyModelClient(model=model, region=region),
        monitor=st.session_state.interpreter_monitor,
    )
    try:
        with st.spinner("正在解释策略…"):
            result = service.interpret(request)
        if result.status == "parsed":
            st.success("策略已解释完成。请在“解释审阅”中逐条核对原文与 DSL 映射。")
            st.session_state["last_parsed_strategy"] = result
            st.session_state["last_strategy_text"] = source_text
            st.session_state["last_review_request_id"] = st.session_state.interpreter_monitor.events[-1].request_id
            st.session_state.pop("pending_clarification", None)
            st.session_state.pop("pending_clarification_text", None)
        else:
            st.warning("需要补充澄清信息。请在下方逐项回答后重新解释。")
            st.caption("尚未确认的表达：" + "、".join(result.ambiguous_terms))
            st.session_state["pending_clarification"] = result
            st.session_state["pending_clarification_text"] = source_text
    except AlphaAgentError as error:
        st.error(f"{error.code}: {error.message}")
        if error.code == "model_client_failure":
            st.info("这是模型服务或模型配置问题，不表示策略本身已被 DSL 拒绝。请查看下方详情中的 provider_message，并确认侧边栏模型对当前 API 项目可用。")
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

last_result = st.session_state.get("last_parsed_strategy")
matching_result = last_result if st.session_state.get("last_strategy_text") == strategy_text else None
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
    st.subheader("研究设置")
    st.caption("日内触发采用当日 Low 触及的保守近似，不是分时精确撮合。")
    initial_capital = st.number_input("初始资金", min_value=1_000.0, value=10_000.0, step=1_000.0, key="dsl_initial_capital")
    cost_bps = st.number_input("单边费用（bps）", min_value=0.0, value=0.0, step=0.5, key="dsl_cost_bps")
    if st.button("运行当前组回测", type="primary", key="run_dsl_backtest"):
        if matching_result.strategy.schema_version != "0.2":
            st.error("当前仅可回测 v0.2 时序策略；静态 v0.1 通用执行器将在下一步接入。")
        else:
            try:
                timed_strategy = TimedStrategyDefinition.model_validate(matching_result.strategy.model_dump(mode="json"))
                params = compile_timed_strategy(timed_strategy, approximate_intraday_with_daily_low=True)
                params["cost_bps"] = cost_bps
                paths = backtest_input_files(Path(data_path))
                kpis, trades = collect_results(paths, params, initial_capital, True, load_ohlc=backtest_load_ohlc, run_backtest=legacy_run_backtest, progress_label="DSL 分组回测")
                st.session_state["dsl_backtest_kpis"] = kpis
                st.session_state["dsl_backtest_trades"] = trades
                st.session_state["dsl_backtest_strategy_text"] = strategy_text
            except AlphaAgentError as error:
                st.error(f"{error.code}: {error.message}")
            except Exception as error:
                st.error(f"回测失败：{error}")

    if st.session_state.get("dsl_backtest_strategy_text") == strategy_text:
        st.subheader("回测结果（日线 Low 近似）")
        kpis = st.session_state.get("dsl_backtest_kpis", pd.DataFrame())
        if not kpis.empty:
            active = kpis[kpis["n_trades"] > 0]
            limit = 15.0
            gauges = st.columns(3)
            gauges[0].html(kpi_gauge_html("sharpe", active["sharpe_ratio"].median() if len(active) else None, max_drawdown_limit=limit))
            gauges[1].html(kpi_gauge_html("annualized_return", active["annualized_return_pct"].median() if len(active) else None, max_drawdown_limit=limit))
            gauges[2].html(kpi_gauge_html("drawdown", active["max_drawdown_pct"].median() if len(active) else None, max_drawdown_limit=limit))
            gross_profit, gross_loss = active["gross_profit"].fillna(0).sum(), active["gross_loss"].fillna(0).sum()
            metrics = st.columns(7)
            for column, label, value in zip(metrics, ["标的数", "总交易数", "有交易标的", "平均标的胜率", "合并盈利因子", "中位标的累计收益", "回撤阈值"], [len(kpis), int(kpis["n_trades"].sum()), len(active), f"{active['win_rate_pct'].mean():.2f}%" if len(active) else "不适用", f"{gross_profit / gross_loss:.3f}" if gross_loss else "不适用", f"{active['cumulative_return_pct'].median():.2f}%" if len(active) else "不适用", "-15.00%"]): column.metric(label, value)
            st.subheader("Top 100 标的")
            options = {"盈利因子（高→低）": "profit_factor", "累计收益率（高→低）": "cumulative_return_pct", "年化收益率（高→低）": "annualized_return_pct", "夏普比率（高→低）": "sharpe_ratio", "胜率（高→低）": "win_rate_pct", "最大回撤（低→高）": "max_drawdown_pct"}
            rank_label = st.selectbox("排序指标", list(options), key="dsl_rank_metric")
            min_trades = st.number_input("最少交易次数", min_value=0, value=1, step=1, key="dsl_min_trades")
            metric = options[rank_label]
            ranked = kpis[(kpis["n_trades"] >= min_trades) & kpis[metric].notna()].sort_values(metric, ascending=metric == "max_drawdown_pct").head(100)
            display = ranked[["标的", "n_trades", "sharpe_ratio", "annualized_return_pct", "max_drawdown_pct", "win_rate_pct", "cumulative_return_pct", "payoff_ratio", "profit_factor", "avg_return_pct", "avg_days_held"]].rename(columns={"n_trades":"交易次数", "sharpe_ratio":"夏普比率", "annualized_return_pct":"年化收益率 (%)", "max_drawdown_pct":"最大回撤 (%)", "win_rate_pct":"胜率 (%)", "cumulative_return_pct":"累计收益率 (%)", "payoff_ratio":"盈亏比", "profit_factor":"盈利因子", "avg_return_pct":"平均单笔收益 (%)", "avg_days_held":"平均持仓天数"})
            st.caption("绿色行符合回撤阈值；红色行超过阈值。选择行后可查看该标的交易。")
            selection = st.dataframe(style_by_drawdown(display, "最大回撤 (%)", -limit, integer_columns=("交易次数",)), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="dsl_top100_table")
            rows = selection.selection.rows if selection else []
            if rows:
                selected_symbol = display.iloc[rows[0]]["标的"]
                symbol_trades = st.session_state.get("dsl_backtest_trades", pd.DataFrame())
                symbol_trades = symbol_trades[symbol_trades["symbol"] == selected_symbol].reset_index(drop=True)
                if not symbol_trades.empty:
                    st.subheader(f"{selected_symbol}：交易 K 线")
                    trade_index = st.selectbox("选择交易段", list(range(len(symbol_trades))), format_func=lambda index: f"第 {index + 1} 笔：t0 {symbol_trades.iloc[index].signal}｜入场 {symbol_trades.iloc[index].entry}｜出场 {symbol_trades.iloc[index].exit}｜{symbol_trades.iloc[index].ret_pct:+.2f}%", key="dsl_chart_trade")
                    trade = symbol_trades.iloc[trade_index]
                    source = kpis.loc[kpis["标的"] == selected_symbol, "源文件"].iloc[0]
                    chart = backtest_load_ohlc(Path(source)).copy()
                    for window in (5, 10, 20): chart[f"MA{window}"] = chart["Close"].rolling(window).mean()
                    signal, entry, exit_ = (pd.Timestamp(trade[column]) for column in ("signal", "entry", "exit"))
                    start = max(0, chart.index[chart["Date"] == signal][0] - 15); end = min(len(chart), chart.index[chart["Date"] == exit_][0] + 11)
                    chart = chart.iloc[start:end].copy().reset_index(drop=True); chart["Index"] = range(len(chart))
                    markers = []
                    for date, label in ((signal, "t0 基准点"), (entry, "入场"), (exit_, "出场")):
                        positions = chart.index[chart["Date"].dt.normalize() == date.normalize()]
                        if len(positions):
                            position = int(positions[0])
                            markers.append({"Index": position, "Price": float(chart.loc[position, "High"]), "Label": label})
                    chart["Date"] = chart["Date"].dt.strftime("%Y-%m-%d")
                    x = {"field":"Index","type":"quantitative","axis":{"title":"交易日（日期见悬停）"}}
                    spec = {"vconcat":[{"height":360,"layer":[{"mark":"rule","encoding":{"x":x,"y":{"field":"Low","type":"quantitative","scale":{"zero":False}},"y2":{"field":"High"}}},{"mark":{"type":"bar","size":8},"encoding":{"x":x,"y":{"field":"Open","type":"quantitative","scale":{"zero":False}},"y2":{"field":"Close"},"color":{"condition":{"test":"datum.Close >= datum.Open","value":"#198754"},"value":"#d62728"}}},{"transform":[{"fold":["MA5","MA10","MA20"],"as":["MA","Value"]}],"mark":{"type":"line"},"encoding":{"x":x,"y":{"field":"Value","type":"quantitative","scale":{"zero":False}},"color":{"field":"MA","type":"nominal"}}},{"data":{"values":markers},"mark":{"type":"text","dy":-12},"encoding":{"x":{"field":"Index","type":"quantitative"},"y":{"field":"Price","type":"quantitative"},"text":{"field":"Label"}}}]},{"height":100,"mark":"bar","encoding":{"x":x,"y":{"field":"Volume","type":"quantitative"},"color":{"condition":{"test":"datum.Close >= datum.Open","value":"#198754"},"value":"#d62728"}}}]}
                    st.vega_lite_chart(chart, spec, width="stretch", key=f"dsl_chart_{selected_symbol}_{trade_index}")
        else:
            st.warning("该分组没有生成已平仓交易。")
    else:
        st.info("在左侧点击“运行当前组回测”后，这里会显示 KPI 概览、Top 100 和交易 K 线。")
    backtest_tab.__exit__(None, None, None)
else:
    st.info("在左侧选择或输入策略，完成解释后即可查看策略与 DSL，并运行当前分组回测。")

st.sidebar.__enter__()
if strategy_source == "新建策略":
    st.subheader("保存当前策略")
    strategy_name = st.text_input("策略名称", value=(matching_result.strategy.strategy_name if matching_result is not None and matching_result.strategy.strategy_name else "未命名策略"), key="saved_strategy_name")
    if matching_result is None:
        st.info("当前文本尚未成功解析；可以保存为待解释策略草稿。")
    if st.button("保存策略", type="secondary"):
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
            st.success(f"已保存跨组策略：{saved.strategy_name}")
        except StrategyLibraryError as error:
            st.error(f"{error.code}: {error.message}")

st.sidebar.__exit__(None, None, None)
