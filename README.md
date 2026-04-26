# TradeX — Multi-Agent AMM Governance Knowledge Base

TradeX is a market-surveillance simulation and training stack for AMM-style environments.  
It combines:

- a working PPO governance pipeline in `tradex/`
- an OpenEnv-compatible surveillance environment in `meverse/`
- interactive analysis surfaces (Dashboard + Playground)

This README is the main "important database" for how the system works, how to run it, and how to navigate the project.

## What TradeX Simulates

TradeX models governance under adversarial DeFi pressure:

- spoofing
- pump-and-dump behavior
- burst manipulation
- front-running style timing attacks
- MEV-like extraction behavior

## Agent Ecosystem

TradeX includes strategically coupled agents:

- **NormalTrader** -> mean-reversion / value trader
- **ManipulatorBot** -> spoof / pump-dump adversary
- **ArbitrageAgent** -> price-correcting stabilizer
- **LiquidityProvider** -> passive market maker
- **Overseer** -> governance controller (policy under training/evaluation)

Agents are not independent. One trade shifts AMM price/liquidity and changes incentives for every other participant.

## Governance Actions

Overseer response space:

- `ALLOW`
- `MONITOR`
- `FLAG`
- `BLOCK`

In PPO training, these are mapped to concrete allow/block-target decisions while keeping the broader action interface.

## Learning Loop

![TradeX governance learning flow](docs/assets/tradex-governance-mindmap.svg)

```text
Reset episode -> generate observations -> policy acts -> environment updates AMM state
-> reward from detection + market health -> trajectory logging -> optimization -> next episode
```

## Implementation Status

- **Implemented now (production path):** PPO training + evaluation (`tradex/`)
- **Planned/extended path:** TRL/Unsloth/GRPO-style LLM governance fine-tuning

## Quick Start

```bash
pip install -r requirements.txt
python -m tradex.train --episodes 1000
python -m tradex.compare_generalization
python app.py
```

## Unified Pipeline (Train + Compare in One Run)

`inference.py` is now a unified runner for:

1. `tradex.train`
2. `tradex.compare_all`
3. combined final output

Run:

```bash
python inference.py --train-episodes 1000 --compare-episodes 100
```

Outputs:

- `outputs/final_combined_output.json`
- `outputs/final_benchmark.csv`

## Dashboard (Operational View)

`dashboard.py` provides a Gradio dashboard for episode execution and analysis.

Main tabs:

- **Episode Runner** -> run one full episode with selected task/policy/seed
- **Policy Comparison** -> compare policies on same task/seed
- **Telemetry Viewer** -> upload telemetry JSONL and replay rewards
- **About** -> scoring/tasks/system info

Key visual outputs include:

- AMM final state gauges (price, confidence, health, volatility)
- reward timeline + cumulative reward
- action distribution by ground truth
- signal heatmap (burst/pattern/suspicion/manipulation/frequency/slippage)
- AMM state evolution
- grade radar chart
- confusion matrix
- step-by-step episode table

Use `docs/Dashboard.md` for full chart-level documentation.

## Playground (Manual Step-Through)

The Playground is OpenEnv web UI integration for manual interaction:

- choose task
- select action (`ALLOW`/`FLAG`/`BLOCK`/`MONITOR`)
- step the environment
- inspect full observation fields
- grade episode manually

Behavior by mode:

- **HF Space mode** -> tabbed app with Playground + Dashboard
- **OpenEnv local mode** -> served via FastAPI/OpenEnv interface
- **Standalone dashboard mode** -> dashboard only

Use `docs/playground.md` for detailed flow and metadata-driven behavior.

## Tasks and Difficulty

- `burst_detection` (easy)
- `pattern_manipulation_detection` (medium)
- `full_market_surveillance` (hard)

## Core Evaluation Framing

TradeX benchmarks are designed to compare:

- heuristic baseline
- always-allow/random sanity baselines
- PPO-trained overseer (current)
- TRL-style overseer variants (future path)

Primary reference command:

```bash
python -m tradex.compare_generalization
```

## Repository Map

- `tradex/env.py` -> AMM market environment (PPO path)
- `tradex/agents.py` -> strategic agent behaviors
- `tradex/overseer.py` -> overseer model + observation encoding
- `tradex/train.py` -> PPO training pipeline
- `tradex/compare.py` -> core evaluation routines
- `tradex/compare_all.py` -> multi-policy benchmark table output
- `tradex/compare_generalization.py` -> unseen-seed benchmark wrapper
- `tradex/reward.py` -> reward shaping logic
- `tradex/utils.py` -> logs and plots
- `meverse/server/meverse_environment.py` -> OpenEnv surveillance environment
- `dashboard.py` -> interactive dashboard
- `app.py` -> app composition (HF Space / OpenEnv / dashboard modes)
- `inference.py` -> unified train+compare runner with combined output

## Documentation Index

- `tradex/README.md` -> PPO-focused module-level overview
- `docs/hf-mini-blog.md` -> blog-ready narrative and governance flow graphic
- `docs/Dashboard.md` -> deep dashboard technical documentation
- `docs/playground.md` -> Playground/OpenEnv integration details
