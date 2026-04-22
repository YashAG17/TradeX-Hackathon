# TradeX — AI Governance for Autonomous Markets

TradeX is an advanced multi-agent, constant-product AMM simulation governed by a PyTorch Reinforcement Learning Overseer.

**Objective:** The AI Overseer monitors the market without knowing the "true intent" of any agent. It must infer malicious manipulation from behavior alone (volume spikes, price impact, burst frequency) and learn when to issue blocks to maintain market stability.

## Features Added for Hackathon Build
- **Constant Product AMM** Market Engine with slippage & price impact logic
- **Behavioral inference only** — 100% of hidden role leakages removed
- **Reward Shaping** normalized strictly between [0.0, 1.0]
- **Agents Evolve Curriculum** scaling from obvious attacks to colluding & stealth behavior over episodes
- **PPO/Actor-Critic** style training logic
- **Hugging Face Space** Gradio Dashboard built directly inside `app.py`
- **Plotting Pipeline** for auto-generated learning curves inside `plots/`

## Project Structure
```
tradex/
├── agents.py       # Base Agent, NormalTrader, ManipulatorBot, ArbitrageAgent, LiquidityProvider
├── env.py          # MarketEnv (AMM step, reset, hidden intent processing)
├── reward.py       # Normalized bound reward compute
├── overseer.py     # Pytorch Model (Actor, Critic)
├── train.py        # Main RL training loop with GPU/onsite support
├── compare.py      # Benchmark WITHOUT vs WITH overseer
└── utils.py        # Charting and CSV/JSON log exporting mechanisms

app.py              # Gradio Hugging Face Spaces UI
```

## How to Run Locally

### 1. Training (Local CPU/GPU)
```bash
python -m tradex.train
```
_Add `--onsite` for long runs & GPU acceleration + `--verbose` for stepwise traces._

### 2. Compare Performance
Benchmark your trained agent (`policy.pth`) against the system with No Overseer.
```bash
python -m tradex.compare
```

### 3. Hugging Face Dashboard UI
Launch the interactive Web UI.
```bash
python app.py
```
