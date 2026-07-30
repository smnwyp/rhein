"""Compatibility aliases for early prototype imports."""
from alpha_agent.domain.interpretation import ClarificationRequired, InterpretationNote as ParserNote, ParsedStrategy, StrategyInterpretationResult
ParserResult = StrategyInterpretationResult
__all__ = ["ClarificationRequired", "ParsedStrategy", "ParserNote", "ParserResult"]
