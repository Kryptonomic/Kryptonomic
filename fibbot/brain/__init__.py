"""Claude brain: the research director that drives the lab through tools.

Importing this package does not import `anthropic`; the SDK is loaded on first use.
"""
from .client import Brain, BrainUnavailable
from .cycles import research_cycle, review_cycle, review_signals
from .tools import build_tools, market_context

__all__ = ["Brain", "BrainUnavailable", "build_tools", "market_context", "research_cycle",
           "review_cycle", "review_signals"]
