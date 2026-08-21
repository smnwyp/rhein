# Research Skill v1 实施计划

## Summary

在已保存的 DSL v0.3 回测结果页新增可恢复的“统计 + 看图”复盘。所有交易统计、DSL 条件重放和样本选择由 Python 确定性完成；GPT 只做逐图观察和受证据约束的报告综合。

## 核心实现

- 新增固定内置 skill：`dd5/v1`（DD5：入场质量与过早退出）；“研究重点”作为普通文本上下文，不能改变流程、执行代码或生成参数。
- 仅允许对当前已载入、且策略 fingerprint 完全匹配的 saved v0.3 run 启动复盘；否则明确提示先载入对应回测。
- 建立 `ResearchSkillDefinition`、`StatisticalEvidence`、`VisualObservation`、`ExperimentCard`、`SavedResearchReport` 等 Pydantic 接口；报告状态为 `preparing → mapping → synthesizing → completed/failed`。
- 对全部交易确定性计算收益、盈亏、持仓天数、退出原因、标的与三个时间段的分布；按“早退亏损”和“趋势赢家”定义 cohort，并重放每笔交易的 DSL 入/出场审计，提取真实条件路径与指标值。缺失源数据必须记录覆盖率，不补造证据。
- 固定抽样 60 笔：收益最高 15、收益最低 15、负收益且最快退出 15、其余按退出原因和时间分层 15；去重后按确定性顺序补足。
- 新建纯 Matplotlib 复盘图渲染器：每笔图包含价格上下文、入场放大、出场放大、K 线/成交量/均线、t0/入场/出场、事件点和审计摘要。固定 MMACD/DD 仅标注为“展示指标”；统计证据只使用 DSL 实际声明的指标。
- 新增独立 OpenAI research client：
  - `gpt-5.4-mini-2026-03-17` 每批 10 图输出严格结构化观察；
  - `gpt-5.4-2026-03-05` 一次性综合统计证据与观察，输出事实、视觉观察、待验证假设及最多 3 张实验卡。
- 实验卡只能引用已存在的 DSL 路径，描述单变量、变动方向、验证指标和反证条件；不生成可执行策略代码、不编造参数值、不自动回测或修改策略。

## 可恢复执行、成本与存储

- 以 `run_id + strategy_fingerprint + skill_version + research_focus` 作为输入指纹；相同已完成报告直接复用，相同未完成报告继续执行。
- 每完成确定性证据、每个视觉批次、最终综合后立即原子持久化。中断后从已验证的批次继续，避免重复模型调用。
- 报告历史沿用“索引 + gzip artifact”模式，独立于回测历史；保存证据、观察、模型快照、token/费用记录和图 checksum，但不将 PNG 图像提交到产品数据。查看报告时按保存的交易定位信息重新渲染图。
- 启动前显示固定 60 图的预估成本与调用上限；模型调用设置输出 token 上限、最多两次短暂重试。没有 `OPENAI_API_KEY`、源数据不可读或模型输出未通过 schema 校验时停止在可恢复状态并展示原因。
- 在现有回测结果页增加“启动/继续复盘”、报告列表、进度与“大尺寸复盘报告” overlay；每条事实、图形观察、假设和实验卡都保存可点击的代表交易引用，点击后回到该笔交易的现有 K 线与 DSL 审计。首版不提供全量视觉扫描或后台 worker。

## 测试

- 先为 skill contract、cohort 边界、分层统计、样本去重与稳定排序编写单元测试。
- 测试全量 DSL 审计特征与展示 MMACD/DD 的隔离、源数据缺失覆盖率、图 checksum 稳定性和交易定位。
- 用假 OpenAI Responses client 测试图像批处理、严格 schema、断点恢复、缓存命中、失败重试和费用记录；不得发起真实网络请求。
- 集成测试覆盖仅限匹配 saved v0.3 run、报告保存/加载/删除、报告跳转现有 K 线、零交易与所有图不可用等情形。

## Assumptions

- 复盘持续在当前 Streamlit 请求中执行并显示进度；采用批次持久化恢复，不引入外部任务队列。
- 金融 KPI、分位数、cohort 比较与费用计算全部由 Python 完成；模型不计算 KPI、不预测价格、不声称因果或未来收益保证。
- 后续 research agent 只能消费 `ExperimentCard → DraftStrategyChange`，再经 DSL/schema/semantic validation、用户确认和确定性回测。
