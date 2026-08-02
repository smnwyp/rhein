# Complex Strategy Semantic Inventory v0.1

## Purpose and status

This is a manually authored, source-grounded reference interpretation of the
complex strategy supplied in the user conversation on 2026-08-02. It is a
**golden design artifact**, not an executable strategy, a backtest result, or
an instruction to silently approximate unavailable data.

It has four purposes:

1. demonstrate the distinction between document provenance, strategy semantics,
   and executable DSL;
2. define the smallest semantic concepts required by this strategy;
3. identify which requirements are closed, need user clarification, or require
   a new DSL/engine capability;
4. become a future test fixture for Semantic Inventory extraction, review, and
   compilation.

The source strategy deliberately permits a non-causal research assumption:
final daily information determines a signal while an order is priced ten
minutes before the close. This document preserves that assumption and flags it;
it does not treat the resulting backtest as live-trading-reproducible.

## 1. Normalized source spans

These are provenance anchors. They record *where the user said something*;
they do not yet decide what that wording means financially.

| Span | Source content, compactly preserved | Structural role |
|---|---|---|
| S01 | Daily frequency; forward-adjusted prices; simple MAs calculated from actual daily closes. | Data definition |
| S02 | Signals use final daily Close/Open/Volume; fills use the latest price ten minutes before close. | Observation and execution timing |
| S03 | Accept the bias of confirming after close while filling at close minus ten minutes. | Explicit research assumption |
| S04 | No fees/slippage; one position per stock; only a fully closed position may seek a new C point. | Costs and lifecycle |
| S05 | MA60 is strictly increasing for ten consecutive comparisons. | Entry condition |
| S06 | Close > MA5 and MA5 > MA10. | Entry conditions |
| S07 | MA5 crosses above MA20 on the candidate day. | Entry event |
| S08 | Inclusive 15-day minimum Close and no more than 15% gain from it. | Entry constraint |
| S09 | Attempt a close-minus-ten-minute entry; if it cannot fill, abandon C and do not defer it. | Entry execution/failure policy |
| S10 | During holding days 1–10, Close < MA5 exits. | Early exit |
| S11 | From day 11, while not high state, Close < MA5 exits. | Normal-state exit |
| S12 | From day 11, return from C >15% enters high state; it persists until exit. | Persistent-state transition |
| S13 | In high state, Close < MA10, doji, large bearish candle, or top-two C-to-current volume exits. | High-state exits |
| S14 | Prefer cumulative volume through close minus ten minutes; otherwise scale historical full-day volume by configurable 96%. | Volume data fallback |
| S15 | No maximum holding period; MA60 no longer affects an open position; failed exits do not defer the order and are re-evaluated next day. | Lifecycle and exit-failure policy |

## 2. Strategy symbol table

The table is the strategy-specific instance layer. It is not a list of market
tickers; it defines the names used within this strategy.

| Symbol | Semantic role | Definition / use | Source |
|---|---|---|---|
| `t` | trading day | Current daily observation/evaluation day. | S01–S15 |
| `C` | signal anchor | Day on which all entry signal conditions hold. It anchors holding-day and return calculations. | S05–S12 |
| `Close[t]` | market observation | Final daily close used for signal and exit predicates. | S01–S03 |
| `Open[t]` | market observation | Daily open used by candlestick predicates. | S02, S13 |
| `Volume[t]` | market observation | Daily/cumulative volume, depending on the explicitly selected data policy. | S02, S13–S14 |
| `MA5/10/20/60[t]` | derived series | Simple moving averages of the declared close series. | S01, S05–S07 |
| `MinClose15[t]` | derived series | Minimum `Close` in the inclusive `[t-14, t]` window. | S08 |
| `position` | position entity | At most one open position per stock. | S04 |
| `position_state` | persistent state | One of `FLAT`, `OPEN_NORMAL`, `OPEN_HIGH`. | S04, S10–S15 |
| `holding_day` | relative time | Trading-day offset from C for an open position. C is day 0. | S10–S12 |
| `HighState` | persistent Boolean state | Enters after the high-return condition and stays true until position exit. | S12–S13 |
| `signal_close[C]` | anchor value | Final daily close on C, used as the explicitly stated reference for the 15% high-state trigger. | S12 |
| `entry_fill_price` | execution value | Latest available price ten minutes before close, not necessarily `signal_close[C]`. | S02–S03, S09 |
| `entry_order` / `exit_order` | order attempt | Attempt made at close minus ten minutes; can fill or fail. | S09, S15 |

## 3. Position state machine

```text
                          all C signal predicates true
             ┌─────────────────────────────────────────────────┐
             │                                                 ▼
          [FLAT] -- entry order fills at close-10m --> [OPEN_NORMAL]
             │                                        │        │
             │ entry order fails                      │        │ holding_day >= 11
             └── abandon this C; keep scanning        │        │ AND Close / Close[C] - 1 > 15%
                                                      │        ▼
       Close < MA5 in the applicable normal phase     │   [OPEN_HIGH]
                                                      │        │
                                                      ▼        │ Close < MA10
                                             exit order attempt │ doji
                                                      │          │ large bearish
                                                      │          │ top-two volume since C
                                                      ▼          ▼
                                                   [FLAT] <- successful exit
                                                      ▲
                                                      │
                               failed exit: retain existing OPEN_* state,
                               do not carry the old order, re-evaluate tomorrow
```

State ordering required by the source:

1. Evaluate a candidate C only while `position_state = FLAT`.
2. An entry fill creates `OPEN_NORMAL`; an entry failure does not create a
   position and does not defer C.
3. On holding day 11 or later, evaluate the high-state transition before
   same-day high-state-gated exits: the source says "from the day it enters".
4. `OPEN_HIGH` is persistent until a successful exit; a later price decline
   does not return the position to `OPEN_NORMAL`.
5. A failed exit leaves the position in its prior state. The next day is a new
   rule evaluation, not a deferred execution of yesterday's order.

## 4. Atomic semantic items

An item is the smallest independently reviewable business requirement: a
predicate, state transition, execution instruction, data policy, or lifecycle
rule. It is deliberately *not* final DSL JSON.

| ID | Source | Category | Pseudo-DSL / semantic requirement | Dependencies | Required capability | Disposition |
|---|---|---|---|---|---|---|
| I-DATA-01 | S01 | data policy | `daily_bar(frequency=1d, adjustment=forward_adjusted)` | — | daily adjusted OHLCV | resolved |
| I-DATA-02 | S01 | definition | `SMA(close_series=declared_actual_close, windows=[5,10,20,60])` | I-DATA-01 | SMA | resolved |
| I-EXEC-01 | S02–S03 | execution policy | `signal_known_at=final_daily_close; order_attempt_at=close_minus_10m` | I-DATA-01 | knowledge/execution-time model | policy-gated |
| I-EXEC-02 | S03 | execution policy | `allow_noncausal_execution_assumption=true` | I-EXEC-01 | research-policy acknowledgement | policy-gated |
| I-LIFE-01 | S04 | lifecycle policy | `max_open_positions_per_symbol=1; reentry_after_successful_exit=true` | position | single-position lifecycle | resolved |
| I-ENTRY-01 | S05 | anchor condition | `strictly_increasing(MA60[t-10..t], comparisons=10)` | I-DATA-02 | monotonic indicator sequence | extension required |
| I-ENTRY-02 | S06 | anchor condition | `Close[t] > MA5[t]` | I-DATA-02 | comparison | resolved |
| I-ENTRY-03 | S06 | anchor condition | `MA5[t] > MA10[t]` | I-DATA-02 | comparison | resolved |
| I-ENTRY-04 | S07 | anchor condition | `cross_above(MA5, MA20, at=t)` | I-DATA-02 | cross event | resolved |
| I-ENTRY-05 | S08 | anchor constraint | `MinClose15[t] = min(Close[t-14..t])` | I-DATA-01 | inclusive rolling minimum on close | extension / formalization required |
| I-ENTRY-06 | S08 | anchor constraint | `Close[t] / MinClose15[t] - 1 <= 0.15` | I-ENTRY-05 | anchored relative return | resolved after I-ENTRY-05 |
| I-ENTRY-07 | S05–S08 | state transition | `FLAT AND all(I-ENTRY-01..06) => candidate C` | I-LIFE-01, I-ENTRY-01..06 | state-gated anchor | extension required |
| I-ENTRY-08 | S09 | entry rule | `attempt_buy(at=close_minus_10m); if no_fill => abandon(C)` | I-EXEC-01, I-ENTRY-07 | intraday execution + entry no-fill | extension/data required |
| I-EXIT-01 | S10 | exit rule | `OPEN_NORMAL AND holding_day in [1,10] AND Close < MA5 => attempt_exit` | C, I-DATA-02 | relative-day state-gated exit | extension required |
| I-EXIT-02 | S11 | exit rule | `OPEN_NORMAL AND holding_day >= 11 AND Close < MA5 => attempt_exit` | C, I-DATA-02 | state-gated exit | extension required |
| I-STATE-01 | S12 | state transition | `OPEN_NORMAL AND holding_day >= 11 AND Close / Close[C] - 1 > .15 => OPEN_HIGH` | C, signal_close[C] | persistent position state | extension required |
| I-STATE-02 | S12 | state transition | `OPEN_HIGH persists until successful position exit` | I-STATE-01 | persistent state lifecycle | extension required |
| I-EXIT-03 | S13 | exit rule | `OPEN_HIGH AND Close < MA10 => attempt_exit` | I-STATE-01, I-DATA-02 | state-gated exit | extension required |
| I-EXIT-04 | S13 | exit rule | `OPEN_HIGH AND (doji(.005) OR large_bearish(.02)) => attempt_exit` | I-STATE-01 | explicit candle pattern | resolved semantically |
| I-EXIT-05 | S13 | exit rule | `OPEN_HIGH AND rank_desc(volume[C..t], ties=include) <= 2 => attempt_exit` | C, I-VOL-01 | anchored top-two rank with ties | extension required |
| I-VOL-01 | S14 | data policy | `volume_at(close_minus_10m); historical_fallback=full_day_volume * configured(0.96)` | I-DATA-01 | intraday cumulative volume + calibrated fallback | extension/data required |
| I-LIFE-02 | S14–S15 | order failure policy | `exit no_fill => retain_state; no carried order; re-evaluate all active rules next day` | position_state | no-fill / exchange availability | extension/data required |
| I-LIFE-03 | S15 | lifecycle policy | `no_max_holding; MA60 condition is entry-only` | I-ENTRY-01 | lifecycle scope | resolved |

`policy-gated` means the user intent is explicit but a research policy must
acknowledge the implication before any result may be presented as a backtest.

## 5. Dependency graph

```text
I-DATA-01 ─┬─ I-DATA-02 ──────┬─ I-ENTRY-01..04
           │                  ├─ I-EXIT-01..04
           │                  └─ I-STATE-01
           └─ I-VOL-01 ────────── I-EXIT-05

I-ENTRY-05 ─ I-ENTRY-06 ─┐
I-ENTRY-01..04 ──────────┼─ I-ENTRY-07 (creates C candidate)
I-LIFE-01 ───────────────┘

I-ENTRY-07 + I-EXEC-01 + I-ENTRY-08
  └─ successful fill creates OPEN_NORMAL and binds C
       ├─ I-EXIT-01 / I-EXIT-02
       └─ I-STATE-01 ─ I-STATE-02 ─┬─ I-EXIT-03
                                    ├─ I-EXIT-04
                                    └─ I-EXIT-05

Any exit attempt + I-LIFE-02
  ├─ fill     → FLAT → I-LIFE-01 permits a later C
  └─ no fill  → preserve OPEN_NORMAL or OPEN_HIGH
```

The graph is the future repair boundary. For example, a defect in I-EXIT-05
does not require the whole source document; it requires its source span, `C`,
the volume data policy, `OPEN_HIGH`, and the exit-order/fill policy.

## 6. Closure and capability report

| Closure dimension | Status | Evidence and consequence |
|---|---|---|
| Vocabulary | clarification required | "highest or second-highest volume, ties trigger" needs an exact rank/tie definition. |
| Parameters | closed with configuration | 15%, 10 days, 15 days, MA windows, candle thresholds, and 96% default are explicit. The permitted range/change-control policy for 96% is a product configuration decision, not a missing trading parameter. |
| Temporal | closed but policy-gated | Signal uses final close while order prices at close minus ten minutes. The source explicitly accepts this non-causal assumption. |
| State | closed at semantic level | Flat, normal holding, high holding, transition, persistence, and exit are stated. Current executable DSL needs persistent state support. |
| Execution | unsupported by current daily-close research mode | Exact close-minus-ten-minute price and no-fill behavior require intraday observations and order availability. |
| Data | unsupported / incomplete for faithful run | The strategy needs minute price, cumulative minute volume, and suspension/limit/fill availability. The 96% fallback addresses only part of volume history. |
| Lifecycle | closed | Single position, no maximum holding, successful-exit re-entry, and failed-exit behavior are explicit. |
| Capability | unsupported for full fidelity | Monotonic MA sequence, persistent state, state-gated exits, top-two tied volume rank, intraday execution, and no-fill handling must be represented and executed. |

### Explicit research-policy finding

```text
Finding: non_causal_execution_assumption
Source: S02–S03
Meaning: the final daily bar determines the signal, yet the strategy chooses a
price observable ten minutes before that close.
User stance: explicitly accepted.
System behavior: require an explicit research-policy acknowledgement; annotate
any result. Do not describe it as causally executable live performance.
```

## 7. Clarification candidates and non-clarification findings

### Blocking user clarification

| ID | Question | Why it matters |
|---|---|---|
| Q-VOL-RANK-01 | “最高或次高成交量；并列也触发”中的“次高”是指前两条交易日的排序名次，还是前两个不同的成交量水平？当多日并列最高时，所有并列日是否都触发？ | These definitions produce materially different exit dates. |
| Q-ADJUST-01 | “前复权”与“实际日收盘价”应如何协同：均线、C 日参考价、收益率使用前复权序列；收盘前十分钟原始成交价如何与该序列对齐？ | A mixed adjusted/raw price basis can distort the 15% state trigger and reported returns. |

### Explicit intent; no user clarification needed

| Finding | Treatment |
|---|---|
| 收盘后确认信号、收盘前十分钟成交 | Keep as an explicit non-causal policy assumption, not a clarification. |
| 买入无法成交即放弃 C | A clear entry no-fill policy. |
| 卖出无法成交不顺延，下一日重新判断 | A clear exit no-fill policy. |
| 无最长持仓 | A clear lifecycle choice. |

### Clear user intent, but currently a capability/data gap

| Finding | Treatment |
|---|---|
| 停牌、涨跌停或其他不可成交原因 | Need exchange/status and fill-availability data; do not fabricate it from daily OHLCV. |
| 分钟级累计量、收盘前十分钟价格 | Need intraday data and deterministic execution model. |

## 8. Lowering boundary: semantic plan to DSL

Lowering means converting a semantic item into a more concrete, executable DSL
node. It must never silently replace a required capability with an approximation.

### Direct or near-direct lowering candidates

| Semantic item | Candidate executable representation |
|---|---|
| I-ENTRY-02 / I-ENTRY-03 | Current comparison condition nodes. |
| I-ENTRY-04 | Current `cross_above` node. |
| I-ENTRY-06 | Comparison between Close and a typed rolling-min-derived reference, once I-ENTRY-05 is formalized. |
| I-EXIT-04 | Current explicit `doji(.005)` and `large_bearish(.02)` pattern nodes. |
| I-LIFE-01 / I-LIFE-03 | Lifecycle policy plus entry-only condition scoping. |

### Required semantic/DSL extensions

| Needed primitive | Serves | Notes |
|---|---|---|
| `monotonic_indicator_sequence` | I-ENTRY-01 | Encodes strict MA60 increase over a declared number of comparisons without LLM-expanded repeated inequalities. |
| `rolling_extremum` | I-ENTRY-05 | Generic inclusive minimum/maximum with field, lookback, and tie semantics. |
| `position_state` / `persistent_state_transition` | I-STATE-01/02 | Allows a state to change once and remain active until position exit. |
| `state_gated_exit` | I-EXIT-02..05 | Runs an exit only in an explicit position state. |
| `anchored_rank` | I-EXIT-05 | Defines top-k ranking, anchor, rank convention, and tie behavior. |
| `execution_time` + `knowledge_time` | I-EXEC-01/02 | Makes non-causal assumptions visible and policy controlled. |
| `fill_policy` / `order_availability` | I-ENTRY-08, I-LIFE-02 | Separates signal, order attempt, fill, no-fill, and next-day behavior. |
| `data_fallback_policy` | I-VOL-01 | Versioned, explicit calibration such as the 96% volume proxy. |

### Compilation decision for this strategy

```text
Do not lower the full strategy to the current daily-close DSL/engine.

It is compile-eligible only after:
1. Q-VOL-RANK-01 and Q-ADJUST-01 are answered;
2. the required primitives above exist in the Semantic Model and DSL;
3. a research policy explicitly accepts the non-causal execution assumption;
4. the selected data source can provide, or an approved proxy policy can
   explicitly replace, the required intraday and order-availability data.
```

Until then, the correct interpreter outcome is a mixed result: structured
clarification for unresolved meaning and typed unsupported/capability findings
for clear requirements that cannot yet be executed faithfully.
