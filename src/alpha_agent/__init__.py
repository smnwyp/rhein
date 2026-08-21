"""Natural-language alpha research agent, Sprint 1.

Keep package import light: deterministic research utilities must not import a
network-provider adapter just because Python resolves ``alpha_agent.*``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_agent.domain.interpretation import StrategyInterpretationResult
    from alpha_agent.domain.strategy import StrategyDefinition
    from alpha_agent.parser.service import StrategyInterpreterService

__all__ = ["StrategyDefinition", "StrategyInterpretationResult", "StrategyInterpreterService"]


def __getattr__(name: str):
    if name == "StrategyDefinition":
        from alpha_agent.domain.strategy import StrategyDefinition
        return StrategyDefinition
    if name == "StrategyInterpretationResult":
        from alpha_agent.domain.interpretation import StrategyInterpretationResult
        return StrategyInterpretationResult
    if name == "StrategyInterpreterService":
        from alpha_agent.parser.service import StrategyInterpreterService
        return StrategyInterpreterService
    raise AttributeError(name)
