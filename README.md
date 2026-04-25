# TradeX — Multi-Agent AMM Governance

TradeX is a multi-agent AMM market simulation where autonomous agents interact strategically while an Overseer learns governance actions from reward feedback. The current implementation is a working PPO pipeline that trains on market dynamics generated in `tradex/`.

The project focuses on governance under DeFi-style adversarial pressure, including:
- spoofing
- pump-and-dump behavior
- burst manipulation
- front-running style timing attacks
- MEV-like extraction behavior

## Agent Roles

TradeX models the following roles:

- **NormalTrader** -> mean-reversion / value trader
- **ManipulatorBot** -> spoof / pump-dump adversary
- **ArbitrageAgent** -> price-correcting stabilizer
- **LiquidityProvider** -> passive market maker
- **Overseer Agent** -> governance controller

Agents do **not** act independently. They react to each other's trades, incentives, price movement, and liquidity:
- Manipulator creates artificial moves
- Arbitrage exploits and corrects inefficiency
- NormalTrader reacts to value and momentum
- Liquidity behavior changes depth and slippage
- Overseer observes resulting multi-agent dynamics

## Multi-Agent Strategic Interaction

Interactions generate emergent, partially observable market signals. The Overseer must infer malicious intent from behavior and market state, then choose governance actions:

- `ALLOW`
- `MONITOR`
- `FLAG`
- `BLOCK`

## Current PPO Training Loop

- Environment rollouts are generated from the AMM market simulator
- Overseer policy acts on observations
- Rewards are computed from detection accuracy, false positives, and market stability
- PPO updates policy weights
- Best checkpoints are benchmarked in evaluation scripts

## Future TRL Training Loop

- The same environment can emit text observations for language-model control
- An LLM-based Overseer can be fine-tuned with Hugging Face TRL
- Environment rewards can optimize policy behavior over episodes

Current status:
- **Implemented now:** PPO training and evaluation pipeline
- **Planned/prototype path:** TRL integration, Unsloth LoRA fine-tuning, and prompt optimization/GRPO

## Governance Learning Flow

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

## Benchmarking

TradeX benchmarking is structured to compare:

- Heuristic baseline
- Static prompted model (future)
- PPO-trained Overseer (current)
- TRL-trained Overseer (future)

`python -m tradex.compare_generalization` currently runs and benchmarks PPO-based behavior on unseen seeds.

## Codebase Structure

- `tradex/env.py` -> AMM market environment
- `tradex/agents.py` -> agent behaviors
- `tradex/overseer.py` -> Overseer model/policy
- `tradex/train.py` -> PPO training pipeline (working)
- `tradex/compare.py` and `tradex/compare_generalization.py` -> evaluation benchmarks
- `tradex/reward.py` -> reward logic
- `tradex/utils.py` -> plots and logging
- `app.py` -> Gradio dashboard using `tradex` modules
- `tradex/graph.py` -> legacy LangGraph prototype (kept for reference, not part of current runtime pipeline)

## Run

```bash
pip install -r requirements.txt
python -m tradex.train --episodes 1000
python -m tradex.compare_generalization
python app.py
```
