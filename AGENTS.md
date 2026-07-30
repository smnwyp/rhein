# 协作约定

## Project Rules

- Use agents only where semantic ambiguity or evidence-based judgment requires them.
- Strategy interpretation and clarification are one agent responsibility.
- Research execution is a deterministic workflow.
- Never let the LLM calculate financial KPIs.
- Never generate arbitrary executable strategy code or use eval for strategy execution.
- Preserve strategy semantics explicitly and treat ambiguity as a first-class result.
- Separate schema validation from semantic validation.
- Prefer deterministic workflows over unnecessary agent autonomy.
- Add tests before expanding DSL functionality.
- Do not introduce LangChain or LangGraph, or later sprints, before explicitly requested.
- Do not silently invent trading parameters; assumptions must be explicit and machine-readable.
- Backtest performance must never be described as guaranteed future performance.

## Git 提交与推送

- 用户说出暗号 **“bibu”** 时，才执行 `git commit` 和 `git push`。
- 未出现暗号时，如任务需要提交，只执行 `git commit`，不要推送远端。

## 重构质量门槛

- 每个重构阶段必须先完成该阶段计划中的测试，并且全部通过，才能进入下一阶段。
- 任一测试、交易级快照 diff、UI 集成测试或用户行为测试失败时，停止后续迁移，先修复失败原因；不得带着失败继续重构。

## 持续执行约束

- 只要当前用户目标存在已知且可执行的待办，必须持续推进直至全部完成；不得将部分完成、阶段性进展或待环境验证的项目当作最终交付而停止。
- 只有所有计划阶段、对应测试与验收项均完成，或出现无法由代理自行消除的外部阻塞时，才能结束本轮工作；外部阻塞必须明确说明具体原因与已完成的替代验证。
