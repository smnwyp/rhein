# Natural-Language Alpha Research Agent

This is an incremental, hands-on AI-agent-engineering project. Sprint 1 implements one user-facing **Strategy Interpreter Agent**: it turns a natural-language trading rule into a constrained Strategy DSL v0.1, or returns precise clarification questions. It does not execute trades, calculate KPIs, or generate executable strategy code.

## Scope and architecture

The interpreter agent owns semantic interpretation and clarification because both require language judgment. It returns exactly one structured variant: `parsed` (a validated DSL plus explicit assumptions/warnings) or `clarification_required` (questions, original ambiguous terms, and an optional partial DSL).

The future Research Workflow is deliberately deterministic Python: it will validate data, calculate indicators/KPIs, simulate execution, apply costs, retain reproducibility, and enforce state transitions. LLMs must not calculate those values. Arbitrary Python generation and `eval` are rejected: the strategy is data, with explicit operands and conditions, not executable text.

The pre-existing `rhein/` application remains the deterministic execution and
KPI layer. A deliberately narrow adapter can compile a supported v0.2 timed
strategy subset into that existing daily engine. DSL v0.3 adds a separate,
deterministic daily-close executor for the supported stateful C-point rules.
Unsupported clauses are rejected rather than dropped. The LLM is never in the
backtest or KPI path.

## DSL v0.1

`StrategyDefinition` pins `schema_version: "0.1"`, `frequency: "1d"`, `direction: "long_only"`, and `position_mode: "fully_invested_or_flat"`. It supports one symbol, one entry tree, and one exit tree.

- Market fields: `open`, `high`, `low`, `close`, `volume`
- Indicators: SMA, EMA, RSI, rolling return, rolling mean volume
- Conditions: persistent comparisons (`greater_than`, `less_than`, `greater_than_or_equal`, `less_than_or_equal`), cross events (`cross_above`, `cross_below`), and nested `and` / `or` groups
- Operands: current market fields, calculated indicators, finite scalars, and explicit scaled operands

For example, `volume > 1.5 × rolling_mean(volume, 20)` is a comparison whose right side is a `scaled_operand`; it is never flattened into text. A cross condition only permits two time-series operands, so it cannot be confused with `close > SMA(20)`.

Missing RSI windows are clarified by default. A future product policy can explicitly set `default_rsi_window`, at which point the interpreter must record that as an assumption.

DSL v0.2 additionally represents anchored, relative-day strategies: `t0` is an anchor day and `t1`, `t2`, and so on are explicit integer offsets. It supports anchor/entry-price references, date-ranged exits, and forced closes. Intraday triggers are represented only with an explicit `intraday_ohlcv` data requirement; a future daily-only backtest must reject them rather than fabricate an intraday path. Shorts, leverage, options, and portfolio allocation remain unsupported and must be rejected or clarified.

DSL v0.3 adds the stateful daily-close primitives required by the C-point
strategy: ordered close extrema inside a fixed lookback window, an anchored
running maximum of volume, explicit doji/large-bearish candle formulas, and a
multiple of the entry price. Its ordered-extrema ties are explicitly resolved as
an explicit **earliest** or **latest** occurrence. A v0.3 definition must also state whether an open
position is force-closed at the end of the sample and whether a later C point
may open another trade. If either policy is missing, the interpreter asks before
calling the provider.

## Model and validation boundary

`StrategyInterpreterService` takes a typed `StrategyInterpretationRequest` and delegates to the provider-independent `StrategyModelClient` protocol. Provider adapters return JSON-compatible data only. The service then performs:

1. Pydantic schema/discriminated-union parsing.
2. Deterministic semantic validation, including operand compatibility, RSI ranges, non-empty groups, cross operand types, and indicator field constraints.
3. Typed errors for client failure, malformed model output, invalid schema, unsupported features, and semantic violations.

`FakeModelClient` queues structured responses or exceptions, so tests require no real external model.

### Source coverage guardrail

Before calling the model, the interpreter deterministically assigns the original strategy's reviewable clauses stable IDs (`C01`, `C02`, …). The model must return one coverage record per ID, stating whether that clause is mapped, an explicit assumption, requires clarification, or is unsupported, along with real DSL paths for mapped clauses. The service rejects a `parsed` response if any clause is absent, unresolved, or points to a non-existent DSL path. This is deliberately an auditability control, not an assertion that natural-language equivalence has been formally proved.

The local **解释审阅** tab displays the source-to-DSL ledger and a labelled, synthetic t0-relative candle timeline. It is a reading aid only: it never uses market data and never feeds the backtest.

The page deliberately presents this human-readable mapping rather than raw DSL
JSON. It is still possible to inspect the DSL through code or saved strategy
records, but the user-facing review question is: “Was every original condition
captured at the intended path?”

The supported v0.3 stateful primitives are intentionally narrow. An unmodelled
stateful construct is returned as a clarification or typed unsupported-feature
result; it is never silently removed. The deterministic executor likewise
rejects a v0.3 feature it does not implement rather than approximating it.

## Monitoring the interpreter

Pass `InMemoryInterpreterMonitor` to the service to record one privacy-preserving event per attempt: outcome, latency, clarification/warning counts, error code, symbol presence, and hashes—not raw strategy text. `summary()` reports parse/clarification/failure rates, latency, and failure-code distribution.

Operational telemetry cannot prove interpretation quality. For that, attach reviewed examples through `record_evaluation(InterpretationEvaluation(...))`; the summary then reports reviewed-sample accuracy. This separates observed system health from evidence-backed quality evaluation and can later be replaced by a database or telemetry adapter without changing the interpreter.

## Development

Python 3.12+ is required.

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit/test_dsl.py tests/integration/test_interpreter.py
```

For a live local test, copy `.env.example` to `.env`, set `AWS_BEARER_TOKEN_BEDROCK`, and start the existing app with `streamlit run app.py`. The interpreter page reads the Bedrock key only from `.env`; it never displays or persists it. Open **Strategy Interpreter** in the sidebar, choose a group/symbol, enter a natural-language rule, answer any targeted clarification, then use **解释审阅** to inspect the original-condition-to-DSL mapping. `BEDROCK_MODEL` and `AWS_REGION` optionally override the infrastructure defaults. The Bedrock adapter uses an 8192-token output ceiling and reports provider truncation explicitly; the page uses session-local, hashed monitoring events.

## Future sprints

Sprint 2 adds only the deterministic backtest workflow. Sprint 3 adds a hand-written research tool loop; Sprint 4 adds the evidence-based reviewer; Sprint 5 adds approval-gated immutable experiments; Sprint 6 adds state and memory; Sprint 7 may introduce LangGraph; Sprint 8 adds formal evaluation and production hardening.
