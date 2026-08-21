"""A narrow OpenAI Responses adapter dedicated to research review."""
from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from typing import Any

from alpha_agent.errors import AlphaAgentError
from alpha_agent.domain.indicators import DSLModel

from .charts import RenderedTradeChart
from .models import ResearchSynthesis, TokenUsage, VisualObservation


class ResearchModelFailure(AlphaAgentError):
    code = "research_model_failure"


class VisualObservationBatch(DSLModel):
    observations: list[VisualObservation]


def _usage(response: object) -> TokenUsage:
    raw = getattr(response, "usage", None)
    return TokenUsage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        # Pricing is deliberately not inferred from an undocumented rate.
        cost_usd=None,
    )


class OpenAIResearchClient:
    """Calls only structured, non-executable research endpoints."""
    def __init__(self, *, api_key: str | None = None, client: Any | None = None, observation_model: str, synthesis_model: str) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ResearchModelFailure("OpenAI client is not installed") from error
            client = OpenAI(api_key=api_key)
        self._client = client
        self._observation_model = observation_model
        self._synthesis_model = synthesis_model

    def observe_batch(self, charts: Sequence[RenderedTradeChart], *, research_focus: str) -> tuple[list[VisualObservation], TokenUsage]:
        content: list[dict[str, str]] = [{
            "type": "input_text",
            "text": (
                "Observe each labelled trade image. Do not calculate returns, predict prices, claim causality, "
                "or propose parameters. DD/MMACD are display-only. Return exactly one observation per image, "
                "using its trade_id and checksum. The user's research focus is handled only during final synthesis."
            ),
        }]
        for chart in charts:
            content.append({"type": "input_text", "text": f"trade_id={chart.trade_id}; checksum={chart.checksum}"})
            content.append({"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(chart.png_bytes).decode("ascii")})
        batch, usage = self._structured(
            model=self._observation_model, name="trade_visual_observations", schema=VisualObservationBatch,
            input_=[{"role": "user", "content": content}], max_output_tokens=4_000,
        )
        return batch.observations, usage

    def synthesize(self, *, evidence: dict[str, object], observations: list[VisualObservation], research_focus: str, allowed_dsl_paths: list[str]) -> tuple[ResearchSynthesis, TokenUsage]:
        instruction = {
            "research_focus": research_focus,
            "evidence": evidence,
            "visual_observations": [item.model_dump(mode="json") for item in observations],
            "allowed_dsl_paths": allowed_dsl_paths,
            "constraints": [
                "Facts must only restate supplied deterministic evidence.",
                "Do not calculate KPIs or predict future price/performance.",
                "Separate visual observations from hypotheses.",
                "Every fact, visual observation, and hypothesis must include a stable finding_id, a statement, and one or more evidence_trade_ids drawn only from selected_trade_context.",
                "Return at most three non-executable, single-variable experiment cards.",
                "Every card must use an allowed DSL path and existing evidence trade IDs; never include a numeric parameter value or code.",
            ],
        }
        try:
            serialized_instruction = json.dumps(instruction, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ResearchModelFailure(
                "research synthesis input could not be serialized",
                details={"exception_type": type(error).__name__},
            ) from error
        synthesis, usage = self._structured(
            model=self._synthesis_model, name="research_synthesis", schema=ResearchSynthesis,
            input_=serialized_instruction, max_output_tokens=4_000,
        )
        return synthesis, usage

    def _structured(self, *, model: str, name: str, schema: type[DSLModel], input_: object, max_output_tokens: int) -> tuple[Any, TokenUsage]:
        payload = schema.model_json_schema()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.responses.create(
                    model=model, input=input_, store=False, max_output_tokens=max_output_tokens,
                    text={"format": {"type": "json_schema", "name": name, "schema": payload, "strict": True}},
                )
                return schema.model_validate_json(response.output_text), _usage(response)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                last_error = error
            except Exception as error:
                last_error = error
            if attempt < 2:
                time.sleep(.25 * (attempt + 1))
        assert last_error is not None
        raise ResearchModelFailure(
            "OpenAI research response failed schema validation after retries",
            details={"exception_type": type(last_error).__name__, "provider_message": str(last_error), "model": model},
        ) from last_error
