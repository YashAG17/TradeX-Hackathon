# TradeX (`tradex/`) — Working PPO Governance Pipeline

## Overview

`tradex/` is the primary working pipeline in this repository. It implements a multi-agent AMM market simulation where agents react strategically to one another, and an Overseer policy is trained with PPO to improve governance outcomes.

The environment captures adversarial DeFi-style behavior including spoofing, pump-and-dump patterns, burst manipulation, front-running style timing, and MEV-like extraction pressure.

## Agent Roles

- **NormalTrader** -> mean-reversion / value trader
- **ManipulatorBot** -> spoof / pump-dump adversary
- **ArbitrageAgent** -> price-correcting stabilizer
- **LiquidityProvider** -> passive market maker
- **Overseer Agent** -> governance controller

Agents are coupled through shared AMM state. They do not act independently; each trade changes price/liquidity and affects downstream behavior of all participants.

## Multi-Agent Strategic Interaction

Strategic interactions create emergent, partially observable signals. The Overseer must infer intent and choose a governance response:

- `ALLOW`
- `MONITOR`
- `FLAG`
- `BLOCK`

In the current PPO implementation, the concrete action map focuses on allow/block-target decisions while preserving compatibility with broader governance messaging.

## Current PPO Training Loop

- Environment rollouts generated
- Overseer policy acts on observations
- Rewards based on detection accuracy, false positives, and market stability
- PPO updates weights
- Best checkpoints benchmarked

## Future TRL Training Loop

- Same environment can emit text observations
- LLM Overseer can be fine-tuned with TRL
- Rewards from environment can optimize policy

Current state is technically honest:
- PPO training and evaluation are implemented and working today
- TRL/Unsloth LoRA/GRPO represent the forward integration path, not a completed replacement

## Files

| File | Purpose |
|------|---------|
| `env.py` | AMM market environment |
| `agents.py` | Agent behavior models |
| `overseer.py` | Overseer neural policy and observation encoding |
| `train.py` | PPO training pipeline |
| `compare.py` | Benchmark runner |
| `compare_generalization.py` | Unseen-seed generalization benchmark wrapper |
| `reward.py` | Reward shaping logic |
| `utils.py` | Plotting and episode logs |
| `graph.py` | Legacy prototype module (not used by the current PPO runtime path) |

## Run

```bash
pip install -r requirements.txt
python -m tradex.train --episodes 1000
python -m tradex.compare_generalization
```