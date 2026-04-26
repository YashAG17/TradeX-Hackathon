# TradeX — Multi-Agent AMM Governance Knowledge Base

TradeX is a market-surveillance and anomaly-detection system for DeFi AMMs, built as an OpenEnv-compatible reinforcement learning gym where agents learn directly through interaction. AMM markets routinely face front-running, sandwich attacks, coordinated manipulation, and MEV-style extraction; while major infrastructure players such as Uniswap, Flashbots, and other DeFi teams use private order flow, private relays, auction systems, and protocol-level mitigations, these defenses are often protocol-specific and less adaptive to emerging attack patterns. TradeX addresses that gap by enabling dynamic detection of unseen bots, novel exploit strategies, and manipulative agents, while providing a practical foundation for research, benchmarking, and deployment-oriented market integrity tooling.

It combines:

- a working PPO governance pipeline in `tradex/`
- an OpenEnv-compatible surveillance environment in `meverse/`
- reproducible evaluation and reporting workflows for benchmark analysis

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

## Evaluation Workflow

The practical workflow for this repository is script-first:

1. Train PPO policy with `tradex.train`.
2. Validate with `tradex.compare_generalization` and `tradex.compare_all`.
3. Run unified benchmark packaging via `inference.py`.
4. Review exported artifacts in `outputs/` and `plots/`.

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

## Training Results (How to Read the Graphs)

### 1) Reward vs Episodes (RL objective trend)

![Reward vs Episodes](docs/assets/reward-vs-episodes.png)

- **What this shows:** per-episode reward (light line) and rolling average (dark line).
- **How to read it:** higher rolling average means the policy is improving expected governance decisions over time.
- **Your current signal:** rolling reward is positive and generally stable with noise, which is expected in adversarial multi-agent training.

### 2) Precision vs Recall Tracking (detection quality trade-off)

![Precision vs Recall Tracking](docs/assets/precision-recall-tracking.png)

- **What this shows:** precision (green) vs recall (orange) during training.
- **How to read it:** high precision with low recall means conservative blocking (fewer false positives but more missed attacks).
- **Your current signal:** precision is strong while recall is modest, suggesting the overseer is cautious and can be tuned for better attack coverage.

### 3) Detection Capability (True Positives vs False Positives)

![Detection Capability](docs/assets/detection-capability.png)

- **What this shows:** smoothed true positives (green area) against false positives (red line).
- **How to read it:** ideal behavior is high green with low red most of the time.
- **Your current signal:** true-positive activity dominates many windows, but red spikes indicate periodic overblocking risk.

### 4) Final Price Error vs Episode (market stability)

![Final Price Error](docs/assets/final-price-error.png)

- **What this shows:** absolute deviation from target AMM price at episode end, plus rolling average.
- **How to read it:** lower values mean better market stability preservation under interventions.
- **Your current signal:** rolling error remains low overall, with occasional spikes during harder attack dynamics.

### Practical interpretation for judges

- The policy demonstrates **consistent positive reward learning**.
- It maintains **strong precision** (low false-alarm tendency).
- It achieves **meaningful threat interception**, though recall still has room to improve.
- It keeps **market stability largely intact** with manageable outlier episodes.

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
- `inference.py` -> unified train+compare runner with combined output

## Documentation Index

- `docs/hf-mini-blog.md` -> blog-ready narrative and governance flow graphic

