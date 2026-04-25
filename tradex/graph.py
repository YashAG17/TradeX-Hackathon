"""
Legacy prototype module.

This file previously contained an early LangGraph orchestration experiment with
agent names and interfaces that no longer match the active `tradex` runtime.
The current production path is:

- `tradex.env.MarketEnv`
- `tradex.overseer.Overseer`
- `tradex.train` (PPO loop)
- `tradex.compare` / `tradex.compare_generalization` (benchmarking)

The legacy graph code is intentionally disabled to avoid stale imports and
confusion. Keep this module as a marker for future graph-based experiments.
"""

# Explicit marker that this module is legacy and intentionally inactive.
IS_LEGACY_GRAPH_PROTOTYPE = True