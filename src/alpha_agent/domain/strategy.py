"""Top-level, execution-independent Strategy DSL v0.1 contract."""
from typing import Annotated, Literal
from pydantic import Field
from alpha_agent.domain.conditions import Condition
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.sequence import TimedStrategyDefinition
class StrategyDefinition(DSLModel):
    schema_version: Literal["0.1"]
    strategy_name: str | None = Field(default=None, min_length=1)
    symbol: str = Field(min_length=1)
    frequency: Literal["1d"]
    direction: Literal["long_only"]
    position_mode: Literal["fully_invested_or_flat"]
    entry_condition: Condition
    exit_condition: Condition


AnyStrategyDefinition = Annotated[StrategyDefinition | TimedStrategyDefinition, Field(discriminator="schema_version")]
