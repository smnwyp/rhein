"""Top-level executable-independent strategy definition."""

from typing import Literal

from pydantic import Field

from alpha_agent.domain.conditions import Condition
from alpha_agent.domain.indicators import DSLModel


class StrategyDefinition(DSLModel):
    schema_version: Literal["0.1"] = "0.1"
    name: str | None = None
    asset: str = Field(min_length=1)
    frequency: Literal["daily"] = "daily"
    direction: Literal["long_only"] = "long_only"
    position_sizing: Literal["fully_invested_or_flat"] = "fully_invested_or_flat"
    entry: Condition
    exit: Condition
