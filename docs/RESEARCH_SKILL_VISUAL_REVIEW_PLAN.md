# Statistical + Visual Research Skill 复盘 v1

## Summary

首版是一个完整的 `research skill`，由两条并列证据线组成：

1. **Statistical analysis**：对该 saved run 的全部交易做可复算统计与 DSL 原生指标分层；
2. **Visual review**：从统计发现中抽取 60 笔代表交易看图，解释形态、核对假设与发现遗漏变量。

首个 skill 是 `dd5/v1`（DD5：入场质量与过早退出）：围绕“哪些入场让利润滚动、哪些入场很快触发退出，以及 DD/MMACD 是否仍放行小波动”复盘一条已保存回测。视觉观察不能替代统计结论；统计关联也不能自动被表述为因果关系或未来收益保证。

```text
saved backtest + research skill + 用户研究重点
  → 全量确定性证据
  → statistical analysis 与 DSL 原生指标分层
  → skill 定义的分层图表样本
  → GPT 逐图观察
  → skill 定义的综合与实验卡
  → 保存的 research report
  →（未来）受控 agent 申请策略变更与新回测
```

## Research Skill Contract

- 新增版本化 `ResearchSkillDefinition`，包含：
  - `skill_id`、版本、研究目的、允许的证据类型；
  - 图表采样策略、逐图观察 schema、综合问题与输出限制；
  - 实验卡规则：只能引用已有 DSL 路径，不得发明参数或执行代码。
- UI 提供内置 skill `入场质量与过早退出 v1`，并让用户输入本次“研究重点”，例如“检查 MMACD DD 是否仍允许小波动入场”。用户输入是 skill 的研究上下文，不是可执行 prompt。
- 该 skill 的固定流程：
  - 对全量交易生成 `StatisticalEvidence`：收益率、交易盈亏、持仓天数、退出原因、时间段和标的分布；
  - 对 `早退亏损`（`days_held ≤ 2` 且 `ret_pct ≤ 0`）与 `趋势赢家`（正收益前四分位且持有至少 5 天）做 cohort 对照；
  - 按退出原因、标的和三个连续时间段交叉分析，样本不足的 cell 明确标为证据不足；
  - 从原始 OHLC 和当前 DSL 重放每笔入场/出场审计，提取策略实际声明的指标数值、条件路径、约束状态与触发规则；
  - 默认抽取 60 张图：最高收益率 15 笔、最低收益率 15 笔、负收益且最快退出 15 笔、其余按退出原因与时间分层抽取 15 笔；去重后补足；
  - 要求逐图观察入场位置、DD/MMACD 状态、入场后延续/失速、出场触发形态与不确定性；
  - 输出“事实 / 视觉观察 / 待验证假设”三类结论，并最多给出 3 张实验卡。
- 以后新增 skill（例如“赢家出场质量”“市场阶段适配”）只替换其研究协议和采样策略，复用同一套证据、视觉与报告基础设施。

## Visual Evidence and Models

- 使用现有 `matplotlib` 生成标准交易复盘 PNG：K 线、成交量、MMACD/DD、均线、t0、入场、出场及确定性 trigger 审计。
- `gpt-5.4-mini-2026-03-17` 分批读取样本图，返回逐笔结构化观察；`gpt-5.4-2026-03-05` 只调用一次，按 skill 的综合规则写最终报告。两者均支持图片输入和 structured outputs。
- DD/MMACD 的统计主口径只使用当前 DSL 明确声明的指标和参数；图上固定 `(10,24,8)` MMACD/DD 仅作为额外视觉观察，明确标注“展示指标，不等同于策略规则”。
- 视觉模型不计算 KPI、不预测价格、不主张因果；每个观察必须关联 trade ID 和图像 checksum。
- 最终模型只读 `StatisticalEvidence`、逐图结构化观察和用户研究重点；所有 KPI 与 cohort 统计继续由 Python 计算。

## Report, UX, and Future Agent Boundary

- 保存独立的 `SavedResearchReport`，按 immutable `run_id` 关联：skill 版本、研究重点、全量统计证据、样本清单、图 checksum、模型 snapshots、实际 token/费用、逐图观察、结论与实验卡。
- 仅在已载入 saved DSL v0.3 run 时显示“启动复盘”；报告内可跳转到现有交易 K 线和入场/出场审计。
- 默认 60 图报告预计约 $0.15–0.25；全量扫描是明确升级动作，需展示费用上限并确认。图像和逐图结果按 checksum 缓存。
- 为后续 agent 定义输入边界 `ExperimentCard → DraftStrategyChange`：agent 只能把实验卡转成待验证草案，必须经 DSL/schema/semantic validation、用户确认和确定性回测，才可产生新结果。

## Test Plan and Assumptions

- 测试全量交易统计、cohort 定义、时间/标的分层、DSL 指标重放、展示指标与 DSL 指标的隔离。
- 测试 skill 驱动的样本选择、全量统计无未来函数、图像稳定性、视觉 schema 校验、证据引用、实验卡单变量约束与报告历史删除级联。
- 集成测试覆盖 saved v0.3 run、缓存命中不重跑模型、错误模型输出、无样本/无交易、以及报告到图表的正确跳转。
- 新增 `OPENAI_API_KEY` 作为部署密钥；保留现有 Bedrock 策略解释功能不变。
- 首版不使用 Kronos、不训练/微调模型、不自动执行策略变更；回测 insight 永不描述为未来收益保证。
