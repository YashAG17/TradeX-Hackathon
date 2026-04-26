# TradeX on Hugging Face: PPO Today, TRL Pathway Next

TradeX is a multi-agent AMM governance environment where autonomous trading agents interact strategically and an Overseer learns intervention behavior from reward feedback.

The current production path in this repository is a working PPO pipeline in `tradex/`. The project is also designed to transition toward Hugging Face TRL workflows for an LLM-style Overseer without misrepresenting what is already implemented.

## Why this matters

AMM ecosystems face adversarial behavior that can look legitimate in isolation:

- spoofing
- pump and dump
- burst manipulation
- front-running style timing attacks
- MEV-like extraction behavior

TradeX models these pressures as a governance problem under partial observability.

## Agent ecosystem

TradeX uses these roles:

- **NormalTrader** -> mean-reversion / value trader
- **ManipulatorBot** -> spoof / pump-dump adversary
- **ArbitrageAgent** -> price-correcting stabilizer
- **LiquidityProvider** -> passive market maker
- **Overseer Agent** -> governance controller

These agents are strategically coupled. One agent's trade changes AMM price and liquidity, which in turn reshapes incentives for every other agent and the Overseer.

## Multi-Agent Strategic Interaction

The Overseer observes combined market signals and must infer intent, then choose:

- `ALLOW`
- `MONITOR`
- `FLAG`
- `BLOCK`

In the current PPO stack, policy actions are implemented as allow/block-target controls, while the broader governance action space is maintained as the long-term interface.

## Current PPO Training Loop

- Environment rollouts generated
- Overseer policy acts on observations
- Rewards based on detection accuracy, false positives, and market stability
- PPO updates weights
- Best checkpoints benchmarked

`python -m tradex.compare_generalization` is currently runnable and provides benchmark output over unseen seeds.

## Future TRL Training Loop

- Same environment can emit text observations
- LLM Overseer can be fine-tuned with TRL
- Rewards from environment can optimize policy

Planned extension points include:
- Hugging Face TRL policy optimization
- Unsloth LoRA for efficient fine-tuning
- prompt optimization and GRPO-style iteration

## Governance learning flow

![TradeX governance learning flow](assets/tradex-governance-mindmap.svg)

```text
Agents trade against each other
        ↓
AMM price / liquidity changes
        ↓
Other agents react strategically
        ↓
Market state evolves
        ↓
Overseer observes combined signals
        ↓
Chooses ALLOW / MONITOR / FLAG / BLOCK
        ↓
Environment returns reward
        ↓
PPO updates policy now
TRL updates policy later
        ↓
Smarter future governance
```

## Benchmark framing

TradeX evaluation should compare:

- Heuristic baseline
- Static prompted model (future)
- PPO-trained Overseer (current)
- TRL-trained Overseer (future)

This keeps experimentation honest: strong current PPO baseline, clear roadmap toward Hugging Face-native LLM governance training.
