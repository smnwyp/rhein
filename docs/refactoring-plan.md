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
    indicators.py           # MA、RSI、成交量均线、日收益等指标
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
    charts.py               # K 线、成交量、MA、t0/入场/出场文字标记
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
| K 线 | 选中标的与交易后显示；含 t0 前 15 根 K 线、成交量、MA5/10/20、紧凑交易日轴、t0…标签，以及“基准点/入场点/出场点”文字。 |
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

### 3. UI 集成与真实用户行为测试

`AppTest` 与 Cypress 的职责不同，二者都需要：

- **Streamlit AppTest（Python）**：快速验证脚本启动、控件/Session State、回测结果数据和页面异常；适合在每个小型重构 commit 中运行。
- **Cypress（真实浏览器 E2E）**：从用户角度点击和等待页面更新，能发现白屏、Streamlit rerun、下拉控件失效、表格选择失效及 Vega-Lite 图表未显示等问题；适合作为阶段验收与 CI 的关键路径测试。

Cypress 将固化以下用户旅程：打开应用 → 选择分组 → 选择 v7 预设 → 验证参数回填与样本/测试双行汇总 → 手动切换 condition toggle → 运行当前组合 → 选 Top 100 标的 → 选交易段 → 断言 K 线、成交量、MA5/10/20 与“基准点/入场点/出场点”文字存在 → 保存组合 → 切换到自定义 → 加载已保存组合并验证完整回填。

对 K 线还保留 Vega-Lite spec 单元断言，作为 Cypress 视觉/DOM 断言之前的快速故障定位层。

### 4. 重构后端到端验收

1. `pytest` 全绿；
2. NVDA、TSLA、一个高流动性组、一个低流动性组的新旧交易明细逐行 diff 为零；
3. v7 样本/测试 metadata 的参数、训练指标、测试指标与重构前一致；
4. 本地启动应用，完成：选组 → 选版本 → 加载 combo → 运行 → 看双行汇总 → 选标的 → 看 K 线；
5. 仅当以上全部通过，才合并最后的兼容层清理。

## 分阶段执行计划与验收门槛

### 阶段 0：建立可重复的测试环境

**改动**：加入 pytest 开发依赖、Cypress 项目配置、固定测试数据 fixture、启动 Streamlit 的 E2E 脚本；不移动生产代码。

**验收**：`pytest`、AppTest、Cypress 冒烟测试均可在干净环境运行；Cypress 能打开本地 app 并识别五个 tab。

**停止条件**：任何测试需要人工预置状态、依赖未固定端口或不能重复运行，则先修测试基础设施。

### 阶段 1：行为冻结（characterization tests）

**改动**：为引擎建立交易级 JSON 快照，为 metadata/预设建立兼容 fixture；补完整的 AppTest 和 Cypress 用户旅程，但不移动生产代码。

**验收**：上述功能矩阵全部有对应测试；NVDA、TSLA 和两类分组的交易明细与 KPI 被锁定；Cypress 完整旅程通过。

**停止条件**：任何现有行为无法被确定性复现时，先记录并修正测试 fixture，不进入重构。

### 阶段 2：抽取领域公共逻辑

**改动**：创建 `domain/strategy_config.py`、`strategy_text.py`、`validation.py`；删除 UI 与 strategy 中重复的策略文案实现，但保留原函数作为转发兼容层。

**验收**：参数、条件 ID、自然语言策略文本、70/30 边界快照全部逐字/逐字段一致；AppTest/Cypress 通过。

**停止条件**：任何预设回填、toggle 文案或策略说明变化，即回退本阶段并修正适配层。

### 阶段 3：拆分数据与回测引擎

**改动**：先迁移 CSV/manifest 发现逻辑到 `data/`，再迁移指标、资格判定、出场和 KPI 到 `engine/`；`rhein/backtest.py` 只作为稳定的兼容 facade。

**验收**：每次子迁移后执行交易级 diff；所有交易日期、价格、原因、收益与 KPI 完全相同；CLI 旧命令不变。

**停止条件**：交易级 diff 非空，即停止后续迁移，不接受“最终收益接近”的替代标准。

### 阶段 4：拆分报告、扫描与 metadata 服务

**改动**：迁移 Markdown/CSV/JSON 报告、搜索结果聚合、分组预设和 saved combo 服务。

**验收**：v1 至 v7 metadata 都能读取；保存/加载 combo 的 Cypress 旅程通过；旧报告字段与文件命名保持一致。

**停止条件**：任何历史 metadata 无法加载、任何组的 result files 未显示或保存组合串组，立即回退。

### 阶段 5：拆分 Streamlit UI

**改动**：按 `controls → summaries → charts → group_views` 顺序抽取；`ui_app.py` 最终只保留页面编排与依赖注入。

**验收**：每抽一个 UI 模块都跑 AppTest；完成一个 tab 跑对应 Cypress 路径。K 线模块须额外验证文字标记、MA、成交量和紧凑 x 轴。

**停止条件**：出现白屏、控件回填不完整、切组卡住、图表缺层或 session state 泄漏，先修复再继续。

### 阶段 6：最终回归与交付

**改动**：只清理已废弃的内部适配代码与重复 import；不改变公共 API。

**验收**：完整 pytest、AppTest、Cypress、交易级 diff、样本/测试 metadata 验证全部通过；本地人工走完关键路径。

**停止条件**：任何失败均保留兼容层，不进行“为了代码整洁”而破坏旧调用的清理。

每个阶段至少一个独立 commit；重构 commit 不包含策略规则或数据结果更新。仅当该阶段验收全部通过，才进入下一阶段。
