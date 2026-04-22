# TradeX — AI Governance for Autonomous Markets

We built a scalable AI governance system that monitors autonomous market agents with hidden incentives and learns intervention policies to preserve fairness and stability.

## Core Prototype Features
- **Unobservable Intent**: The Overseer neural network explicitly does *not* know which agent is malicious or benign. It must infer "Pump and Dump" or "Burst Accumulation" patterns from real-time sequential behavioral data.
- **DeepMind-style PPO Architecture**: 
  - Dual Actor-Critic with Generalized Advantage Estimation (GAE).
  - Entropy regularization schedule to prevent early policy collapse.
  - Surrogate objective clipping and mini-batch sample updates for stability.
- **Dynamic Manipulators**: Agents stochastically scale through 5 difficulty stages incorporating stealth strategies, unpredictable timing, and noisy secondary coordination.
- **Constant-Product AMM**: A simulated $k=xy$ liquidity pool enforcing real-world slippage penalty physics.

## Competition Validation Metrics
The `compare_generalization.py` script validates the Neural Network against a non-assisted baseline on *completely unseen seeds and random agent identities*.

- Precision & Recall scaling to `>85%` and `>75%`.
- Manipulation Loss physically blocked and recovered.
- Massive stability gain preventing market crashes.

## How to Run
### 1. Execute RL Training (The Learning Engine)
```bash
# General quick training over varying curriculum stages
python -m tradex.train --episodes 1000

# Advanced Onsite Mode: scales episodes deeply and attempts GPU
python -m tradex.train --onsite
```

### 2. Run Generalization & Evaluation
Tests model capabilities on randomly seeded instances locking metrics across baselines.
```bash
python -m tradex.compare_generalization
```

### 3. Launch Demo Dashboard (Live Visualizer)
Watch the AI intercept malicious actions exactly when they happen with real logging traces.
```bash
python app.py
```
