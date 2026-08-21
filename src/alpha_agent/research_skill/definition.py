"""The versioned, built-in research protocol.  It is not user programmable."""
from __future__ import annotations

from alpha_agent.domain.indicators import DSLModel


class ResearchSkillDefinition(DSLModel):
    skill_id: str
    version: str
    display_name: str
    purpose: str
    allowed_evidence_types: tuple[str, ...]
    visual_sample_target: int
    visual_batch_size: int
    observation_model: str
    synthesis_model: str


DD5 = ResearchSkillDefinition(
    skill_id="dd5",
    version="v2",
    display_name="DD5 · 入场质量与过早退出",
    purpose="复盘哪些入场延续为趋势赢家、哪些在短期内以亏损退出。",
    allowed_evidence_types=("saved_trade_ledger", "dsl_replay_audit", "source_ohlcv", "display_indicator"),
    visual_sample_target=60,
    visual_batch_size=10,
    observation_model="gpt-5.4-mini-2026-03-17",
    synthesis_model="gpt-5.4-2026-03-05",
)

# The UI must select from this closed registry.  A user's research focus is
# context for the selected protocol, never a user-programmable prompt or an
# executable strategy definition.
RESEARCH_SKILLS: tuple[ResearchSkillDefinition, ...] = (DD5,)
