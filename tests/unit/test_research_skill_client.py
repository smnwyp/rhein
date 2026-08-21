from __future__ import annotations

import json
from types import SimpleNamespace

from alpha_agent.research_skill.charts import RenderedTradeChart
from alpha_agent.research_skill.client import OpenAIResearchClient
from alpha_agent.research_skill.models import ResearchSynthesis


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"observations": [{
                "trade_id": "t1", "chart_checksum": "checksum", "entry_context": "entry context",
                "post_entry_path": "path", "exit_context": "exit", "display_indicator_observation": "display only",
                "uncertainty": "one chart is not causal evidence",
            }]}),
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )


def test_research_visual_client_uses_strict_structured_output_without_network() -> None:
    responses = _Responses()
    client = OpenAIResearchClient(
        client=SimpleNamespace(responses=responses), observation_model="observation", synthesis_model="synthesis",
    )
    observations, usage = client.observe_batch(
        [RenderedTradeChart(trade_id="t1", png_bytes=b"png", checksum="checksum")], research_focus="inspect entries",
    )

    assert observations[0].trade_id == "t1"
    assert usage.input_tokens == 12
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["text"]["format"]["strict"] is True


def test_research_synthesis_schema_requires_all_root_output_categories() -> None:
    schema = ResearchSynthesis.model_json_schema()

    assert set(schema["required"]) == {
        "facts", "visual_observations", "hypotheses_to_verify", "experiment_cards",
    }
    finding = schema["$defs"]["ResearchFinding"]
    assert set(finding["required"]) == {"finding_id", "statement", "evidence_trade_ids"}
