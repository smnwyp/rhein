# Codebase architecture

## System boundary

The repository currently has two deliberately different layers:

- `alpha_agent/` owns natural-language strategy interpretation, its constrained
  and versioned DSL, clarification, source-coverage auditing, and the limited
  bridge into the existing backtest implementation.
- `rhein/` owns deterministic OHLC data loading, the existing daily backtest,
  KPI calculation, result tables, and shared Streamlit visual components.

An LLM may interpret text and identify ambiguity. It must not calculate KPIs,
simulate a trade, decide execution order, or execute generated Python. Those
operations remain deterministic Python code.

## Entrypoints

- `app.py` — Streamlit Cloud entrypoint; delegates to `rhein.ui_app`.
- `app.py` is the sole root-level application entrypoint for Streamlit Cloud.
- `pages/1_Strategy_Interpreter.py` — local Streamlit page for creating/loading
  a strategy, answering clarifications, reviewing source-to-DSL mapping, and
  running the currently supported v0.2/v0.3 strategy subset against the
  selected group.
- CLI commands live in `scripts/` and can be invoked with `python -m scripts.<name>`.

## Application package: `rhein/`

- `backtest.py` — OHLC loading, strategy execution, KPIs, reporting, and CLI implementation.
- `strategy.py` — atomic-condition IDs and parameter parsing.
- `paths.py` — repository-relative data and report paths.
- `ui_app.py` — Streamlit composition and interactive workflow.

## Interpreter package: `src/alpha_agent/`

- `domain/` — strict, immutable Pydantic v2 models for operands, conditions,
  strategy definitions, interpretation variants, and relative-day v0.2/v0.3
  rules.
- `model/` — provider-independent `StrategyModelClient` implementations. The
  Bedrock client reads its bearer token and configuration from `.env`; the fake
  client is used by tests.
- `parser/service.py` — the single interpreter orchestration boundary.
- `domain/semantic_inventory.py` and `parser/inventory.py` — the strict,
  source-grounded semantic intermediate representation. It records symbols,
  atomic requirements, closure dimensions, and required capabilities before a
  complex strategy is lowered into the recursive executable DSL.
- `parser/completeness.py` — deterministic clause segmentation, targeted
  preflight clarification guards, and source-coverage validation.
- `parser/review.py` and `parser/mock_timeline.py` — deterministic Chinese DSL
  descriptions and a synthetic, labelled t0 timeline for review. Neither uses
  market data or drives a backtest.
- `research/legacy_adapter.py` — an intentionally narrow compiler from a
  supported v0.2 subset into the pre-existing `rhein.backtest` parameters. It
  raises a typed unsupported-feature error instead of dropping a DSL clause.
- `research/v03_engine.py` — deterministic daily-close execution for the
  supported v0.3 C-point/stateful subset. It evaluates ordered extrema, anchor
  running-volume maxima/ranks, persistent post-trigger states, candlestick
  definitions, lifecycle policy, and costs; it never invokes a model.
- `strategy_library.py` — repository-local saved strategy records, including
  original text, DSL, explicit assumptions, source clauses, and coverage data.

## Strategy interpretation request flow

```text
Natural-language strategy
  → deterministic C01…Cn clause segmentation
  → deterministic ambiguity guards (when applicable)
  → compact Semantic Inventory model call
  → deterministic inventory coverage / closure / capability gate
  → clarification or typed unsupported result when not compile-eligible
  → provider-independent model client
  → Pydantic schema parsing
  → semantic DSL validation
  → source-coverage validation
  → parsed DSL or structured clarification request
  → Streamlit source-to-DSL review
```

The coverage contract is a hard gate. Every C clause must occur exactly once in
the model result. A `parsed` result may contain only `mapped` or `assumption`
entries, and each such entry must reference a real DSL path. `clarification_required`
and `unsupported` entries cannot be silently promoted to a parsed strategy.

Coverage establishes traceability, not formal proof that two natural-language
phrases have identical financial meaning. The review UI therefore presents the
original wording, C ID, disposition, DSL path, and a compact explanation for
user verification.

The inventory is intentionally not executable DSL and does not calculate any
financial value. It is a bounded agentic planning step: the model identifies
source-grounded semantic components and whether vocabulary, parameters,
temporal relations, state, execution, data, lifecycle, and capability are
closed. This prevents clearly unsupported complex strategies from consuming a
second, much larger recursive-DSL generation request.

## Clarification and execution safeguards

- Strategy interpretation and clarification are one responsibility; a
  clarification is an output state, not a second agent.
- The service asks before choosing an ambiguous low-price field, an unspecified
  t0 reference price, or an ambiguous indicator-change basis. User answers are
  sent back through the same interpreter request.
- Before any provider call, the service asks for the two lifecycle choices that
  are necessary for a full deterministic v0.3 run: treatment of an open
  position at sample end, and whether a completed trade may be followed by a
  new C-point entry. The ordered A→B extrema drawdown, anchor-to-current
  expanding volume maximum, and doji/large-bearish candle formulas are now
  explicit v0.3 primitives. Their tie rule is `earliest` or `latest`, never an implicit
  implementation accident.
- A DSL explicitly declares `frequency: "1d"`. Intraday triggers declare an
  `intraday_ohlcv` requirement; daily OHLC cannot establish the exact intraday
  price path.
- The current legacy adapter permits a documented daily-Low approximation only
  where requested by the UI. It refuses a rolling-low constraint defined on
  Close rather than silently executing it as Low.
- The adapter also rejects unsupported condition trees or timing rules rather
  than producing a partial backtest.
- The current research policy executes every backtest order at the final daily
  close. If source prose asks for a close-minus-minutes price or cumulative
  minute volume, the interpreter records an explicit daily-close proxy
  assumption and uses final daily Close/Volume. Suspension, price-limit, and
  no-fill handling are surfaced as a warning: daily OHLCV assumes the close
  order filled and never fabricates exchange-state data.
- v0.3 supports a persistent state that activates once and remains active until
  position exit, plus a tied top-two running-volume rank from t0. Exit rules may
  require or forbid such a state; the daily executor evaluates a state
  transition before same-day state-gated exits.

## Bedrock and observability

The Bedrock adapter uses a tiny native JSON-schema envelope because Bedrock
cannot compile the recursive DSL directly. Its embedded interpretation is
immediately parsed by the same local strict validators before it can become a
strategy.

For complex strategies, the adapter first calls a separate compact
`SemanticInventory` transport. Its strict compact-output contract keeps normal
responses small, while its 8192-token ceiling prevents a legitimate complex
inventory from being truncated solely by an artificial 4096-token cap. Only an
inventory marked `compile_eligible` proceeds to the existing full DSL call.
Transient Bedrock connection and read-timeout failures receive one bounded
retry; provider 4xx errors and local validation failures never retry.

The output ceiling is 8192 tokens because a v0.3 DSL plus C-clause coverage can
legitimately exceed 4096. A `max_tokens` stop is reported as a truncation, never
misrepresented as a strategy schema failure. Raw provider payloads are not
shown in the UI diagnostic.

`InMemoryInterpreterMonitor` stores only per-session hashes, lengths, outcomes,
latency, counts, and error codes. It does not retain the raw natural-language
strategy. Operational success is not an accuracy claim; source coverage and
user review address a different, semantic question.

## Operational scripts: `scripts/`

- Data acquisition: `download_nasdaq.py`
- Feature groups: `build_nasdaq_feature_groups.py`, `organize_nasdaq_groups.py`
- Research and scans: `scan_all_group_domains.py`, `scan_high_volatility_group.py`,
  `analyze_group_statistics.py`, `analyze_group_symbol_quality.py`

Run the canonical module form, for example:

```bash
.venv/bin/python -m scripts.scan_all_group_domains
```

All internal imports should target `rhein.*`; command-line behavior belongs in `scripts/`.
