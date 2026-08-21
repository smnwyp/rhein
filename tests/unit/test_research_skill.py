from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd

from alpha_agent.domain.sequence import TimedStrategyDefinition
from alpha_agent.research_skill.audit import attach_replay_audits
from alpha_agent.research_skill.evidence import build_statistical_evidence, select_visual_samples
from alpha_agent.research_skill.definition import DD5, RESEARCH_SKILLS
from alpha_agent.research_skill.history import JsonResearchReportHistory
from alpha_agent.research_skill.client import ResearchModelFailure
from alpha_agent.research_skill.models import ResearchSynthesis, SavedResearchReport
from alpha_agent.research_skill.service import ResearchSkillService, _synthesis_evidence


def _trade(index: int, *, ret: float, days: int, reason: str = "v03:exit") -> dict[str, object]:
    return {
        "symbol": f"S{index % 3}", "signal": f"2024-01-{index + 1:02d}",
        "entry": f"2024-01-{index + 2:02d}", "exit": f"2024-02-{index + 1:02d}",
        "ret_pct": ret, "days_held": days, "reason": reason,
    }


def test_cohorts_and_visual_sample_selection_are_deterministic_and_unique() -> None:
    trades = [
        _trade(index, ret=float(index - 30), days=1 + index % 9, reason="v03:stop" if index % 2 else "v03:trend")
        for index in range(60)
    ]
    evidence = build_statistical_evidence(trades)

    assert evidence.cohorts["early_loss"].trade_count > 0
    assert evidence.cohorts["trend_winner"].trade_count > 0

    first = select_visual_samples(evidence, target_count=60)
    second = select_visual_samples(evidence, target_count=60)
    assert first == second
    assert len({sample.trade_id for sample in first}) == 60
    assert {sample.bucket for sample in first}.issuperset({"highest_return", "lowest_return", "fast_negative_exit"})


def test_dd5_is_the_first_registered_research_skill() -> None:
    assert RESEARCH_SKILLS == (DD5,)
    assert DD5.skill_id == "dd5"
    assert DD5.display_name.startswith("DD5")
    assert DD5.version == "v2"


def test_replay_audit_reports_progress_and_loads_each_source_once() -> None:
    evidence = build_statistical_evidence([
        _trade(0, ret=-1.0, days=1),
        _trade(1, ret=2.0, days=2),
        _trade(2, ret=3.0, days=3),
    ])
    updates: list[tuple[int, int, str]] = []
    load_count = 0

    def load_once(path: Path) -> pd.DataFrame:
        nonlocal load_count
        load_count += 1
        return pd.DataFrame()

    attach_replay_audits(
        evidence,
        strategy=cast(TimedStrategyDefinition, object()),
        source_paths={"S0": Path("one.csv"), "S1": Path("one.csv"), "S2": Path("one.csv")},
        load_ohlc=load_once,
        progress=lambda completed, total, symbol: updates.append((completed, total, symbol)),
    )

    assert load_count == 1
    assert updates[0] == (0, 3, "S0")
    assert updates[-1] == (3, 3, "S2")


def test_synthesis_evidence_excludes_full_audits_and_keeps_selected_trade_context() -> None:
    evidence = build_statistical_evidence([
        _trade(0, ret=-1.0, days=1),
        _trade(1, ret=2.0, days=6),
    ])
    report = SavedResearchReport.new(
        run_id=uuid4(), strategy_id=uuid4(), strategy_fingerprint="fingerprint",
        skill_id="dd5", skill_version="v1", research_focus="check entries",
    ).evolve(evidence=evidence, samples=select_visual_samples(evidence, target_count=1))

    compact = _synthesis_evidence(report)

    assert "trades" not in compact
    assert len(compact["selected_trade_context"]) == 1
    assert compact["selected_trade_context"][0]["trade_id"] == report.samples[0].trade_id
    json.dumps(compact, ensure_ascii=False)


def test_research_history_uses_a_small_index_and_gzip_artifact(tmp_path) -> None:
    report = SavedResearchReport.new(
        run_id=uuid4(), strategy_id=uuid4(), strategy_fingerprint="fingerprint",
        skill_id="entry_quality_and_premature_exit", skill_version="v1", research_focus="check entries",
    )
    history = JsonResearchReportHistory(tmp_path / "saved_research_reports.json")
    history.save(report)

    summaries = history.list_for_run(report.run_id)
    assert len(summaries) == 1
    assert summaries[0].report_id == report.report_id
    assert (tmp_path / "saved_research_report_artifacts" / f"{report.report_id}.json.gz").exists()
    assert history.load(report.report_id) == report


def test_legacy_string_findings_remain_readable_without_trade_references() -> None:
    synthesis = ResearchSynthesis.model_validate({
        "facts": ["旧版事实"],
        "visual_observations": ["旧版图形观察"],
        "hypotheses_to_verify": ["旧版假设"],
        "experiment_cards": [],
    })

    assert synthesis.facts[0].statement == "旧版事实"
    assert synthesis.facts[0].evidence_trade_ids == []
    assert synthesis.facts[0].finding_id == "legacy-facts-1"


def test_new_synthesis_rejects_unlinked_finding() -> None:
    evidence = build_statistical_evidence([_trade(0, ret=-1.0, days=1)])
    report = SavedResearchReport.new(
        run_id=uuid4(), strategy_id=uuid4(), strategy_fingerprint="fingerprint",
        skill_id="dd5", skill_version="v2", research_focus="check entries",
    ).evolve(evidence=evidence, samples=select_visual_samples(evidence, target_count=1))
    synthesis = ResearchSynthesis.model_validate({
        "facts": [{"finding_id": "fact-1", "statement": "未关联交易", "evidence_trade_ids": []}],
        "visual_observations": [],
        "hypotheses_to_verify": [],
        "experiment_cards": [],
    })

    import pytest
    with pytest.raises(ResearchModelFailure):
        ResearchSkillService._validate_synthesis(synthesis, report)
