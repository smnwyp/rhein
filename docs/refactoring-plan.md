# Rhein 模块化重构与回归保障计划

状态：计划阶段。未经明确确认，不开始移动生产模块。

## 目标与边界

目标是将 `rhein/backtest.py`、`rhein/strategy.py`、`rhein/ui_app.py` 拆成职责单一、可独立测试的模块，同时保持策略规则、参数默认值、metadata 格式、CLI 参数、UI 文案与所有用户路径的行为不变。

本轮不改变任何交易条件、KPI 公式、70/30 验证口径、搜索排序、数据文件格式或 UI 产品需求。业务逻辑变化必须与纯重构分开提交。

## 目标目录

```text
rhein/
  domain/
    strategy_config.py      # 默认参数、条件 ID、参数校验与兼容规范化
    strategy_text.py        # 参数摘要与自然语言策略说明（唯一来源）
    validation.py           # 每标的 70/30 时间切分与测试期 t0 边界
  data/
    ohlc.py                 # OHLCV 读取、yfinance 格式兼容、数据校验
    discovery.py            # manifest 优先的价格文件发现
    presets.py              # metadata 版本、搜索预设、saved combo 读写
  engine/
    indicators.py           # SMA、RSI、成交量均线、日收益等指标
    eligibility.py          # t0 与 tN 的条件判定
    exits.py                # 早期止损、趋势出场、强制平仓执行
    runner.py               # 单标的交易循环；维持 run_backtest 公共接口
    kpis.py                 # 单标的与跨标的 KPI
  reporting/
    markdown.py             # CLI Markdown 报告
    scan_results.py         # 搜索结果与样本/测试结果汇总
  ui/
    controls.py             # 参数控件、toggle 依赖、session state 同步
    summaries.py            # 单组 KPI、Top 100、样本/测试双行汇总
    charts.py               # K 线、成交量、SMA、t0/入场/出场文字标记
    group_views.py          # 分组预设、保存组合、分组组合表与分组分析
    styles.py               # 数字格式与回撤颜色
    app.py                  # 薄页面编排入口
```

`rhein/backtest.py` 将在整个重构后仍作为兼容层，继续导出当前的 `load_ohlc`、`input_files`、`run_backtest`、`calculate_kpis` 等函数。外部 import 路径、脚本调用和旧测试不需要改。

## 已识别的重复与拆分重点

- `params_to_text()` 和 `strategy_narrative()` 在 `strategy.py`、`ui_app.py` 重复；收敛到 `domain/strategy_text.py`，防止策略说明与真实规则脱节。
- `backtest.py` 同时承担引擎、数据、KPI、报告、CLI 与 sweep；按上方目录逐层迁移。
- `ui_app.py` 同时承担数据持久化、业务聚合、状态同步和渲染；优先抽离无 Streamlit 副作用的逻辑，再拆 UI 区块。

## 必须回归的用户功能矩阵

以下每一项都会有自动化测试；涉及浏览器选择、表格行选择或图表视觉层的项目还会有 AppTest/spec 断言和一次本地手工验收。

| 功能模块 | 必须验证的行为 |
|---|---|
| 条件 toggle | 14 个原子条件可独立 on/off；关闭后相关输入灰化；共享输入仅在所有依赖关闭时灰化；EX-05 依赖 EX-04；修改任意项切回“自定义参数”。 |
| 分组选择 | 切组清除旧回测/旧预设状态，不白屏；manifest 限定标的，`* 2.csv` 不重复纳入；九个成熟组均出现。 |
| 参数版本与 combo | 切换 v4/v5/v6/v7 能将参数、toggle、数值控件完整回填；metadata 旧字段仍可读；切回自定义不再被 rerun 覆盖。 |
| 保存 combo | 仅分组目录可保存；名称为空/重复报错；保存后可加载；加载后参数与条件 ID 完全一致；不修改其他组的 `saved_combos.json`。 |
| 当前 combo 回测 | 固定仓位默认、费用默认 0；单标的 KPI、Top 100、交易明细、下载 CSV 和错误处理保持不变。 |
| 参数扫描 | 当前范围解析、tN 对应 tN+1/tN+2 早期止损日、排序与回撤红绿提示保持不变。 |
| 样本/测试验证 | 每标的按日期 70/30；参数只按样本指标选择；测试只计入测试期 t0；训练历史只用于指标预热；样本/测试双行汇总显示收益、胜率、盈亏比、交易数、回撤。 |
| 分组结果页 | 当前激活版本的 result files 正确读取；v6 的“条件筛选/完整复验”和旧版第一/二阶段兼容；样本/测试列正确展示。 |
| K 线 | 选中标的与交易后显示；含 t0 前 15 根 K 线、成交量、SMA5/10/20、紧凑交易日轴、t0…标签，以及“基准点/入场点/出场点”文字。 |
| CLI/报告 | 原 CLI 参数、profile 覆盖、sweep、Markdown/CSV/JSON 报告字段和文件产出保持兼容。 |

## 测试策略

### 1. 先建立行为冻结测试

在移动代码前增加交易级快照测试。对固定 OHLC fixture 与真实 NVDA/TSLA 小样本，精确比较：

- 每笔 `signal`、`entry`、`exit`、`entry_px`、`exit_px`、`reason`、`ret_pct`；
- KPI（交易数、胜率、盈亏比、盈利因子、累计收益、最大回撤）；
- 日内止损、收盘止损、强制平仓、日内保护、关闭条件、测试期边界；
- 70/30 测试中任一交易的 t0 都不得早于测试开始日。

这些基线会写成 fixture/JSON；重构后逐字段比较，而不是只比较最终收益。

### 2. 模块单元测试

- 参数规范化、旧 metadata 兼容、条件 ID、toggle 依赖、数据发现、CSV 格式、KPI 和 validation 边界。
- 所有纯函数不依赖 Streamlit，确保能快速测试。

### 3. UI 集成测试

使用 Streamlit `AppTest` 覆盖：启动、切换组、选择 v7、加载组合、运行回测、显示双行样本/测试表、Top 100、保存/加载组合。

对 K 线使用 Vega-Lite spec 断言：必须包含蜡烛、成交量、SMA5/10/20 与三个中文节点文字数据。必要时补一次本地浏览器手工验收。

### 4. 重构后端到端验收

1. `pytest` 全绿；
2. NVDA、TSLA、一个高流动性组、一个低流动性组的新旧交易明细逐行 diff 为零；
3. v7 样本/测试 metadata 的参数、训练指标、测试指标与重构前一致；
4. 本地启动应用，完成：选组 → 选版本 → 加载 combo → 运行 → 看双行汇总 → 选标的 → 看 K 线；
5. 仅当以上全部通过，才合并最后的兼容层清理。

## 执行顺序与提交纪律

1. 补测试与快照；不移动生产代码。
2. 抽 `domain/` 公共逻辑，旧文件改为调用新模块。
3. 抽 `data/` 和 `engine/`；每一小步运行交易级 diff。
4. 抽 `reporting/` 和 CLI。
5. 抽 `ui/` 的无副作用逻辑、控件、汇总、图表与各 tab。
6. 完整回归与手工验收。

每一步单独 commit；重构 commit 不包含策略规则或数据结果更新。任一交易级 diff、UI 路径或测试失败时停止迁移并回退该步骤，而不是继续在错误基础上重构。
