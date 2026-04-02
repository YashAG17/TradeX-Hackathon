# MEVerse

MEVerse is an OpenEnv environment for training and evaluating agents on MEV-aware execution in a simulated Uniswap V3 pool. The agent trades ETH/USDC, manages concentrated liquidity, and reacts to mempool-visible attacks such as JIT liquidity and sandwiching.

## What It Models

- Real-world task: DeFi execution and liquidity management under adversarial market conditions.
- Environment type: OpenEnv FastAPI server with typed `Action` and `Observation` models.
- Tasks:
  - `easy`: passive market, low volatility
  - `medium`: JIT-liquidity adversary
  - `hard`: adaptive adversary with higher volatility

## Action Space

`MeverseAction` accepts:

- `swap_exact_in`
- `split_swap`
- `add_liquidity`
- `remove_liquidity`
- `range_order`
- `jit_liquidity`
- `hold`
- `close_episode`

Each action uses a `params` object with the fields required for that action, for example:

```json
{"action_type":"swap_exact_in","params":{"amount_in":1.0,"zero_for_one":true,"use_private_rpc":false}}
```

## Observation Space

`MeverseObservation` returns:

- Pool state: `current_tick`, `current_price`, `sqrt_price`, `active_liquidity`
- Local tick window: `tick_distribution`
- Agent portfolio: `agent_token0`, `agent_token1`, `agent_positions`
- Adversary context: `mempool`, `last_mev_loss`
- Episode context: `step_num`, `max_steps`, `task_name`
- Metadata: `episode_id`, `available_tasks`, `last_action_error`

## Reward and Grading

- Dense reward includes swap quality, LP fee capture, MEV losses, and end-of-episode portfolio change.
- `grade()` returns normalized component scores and a deterministic `final_score` in `[0.0, 1.0]`.

## Local Usage

Validate the environment:

```bash
cd /home/casp1an/Code/TradeX/meverse
openenv validate
```

Run the server locally:

```bash
cd /home/casp1an/Code/TradeX/meverse
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run inference from the repo root:

```bash
python inference.py
```

## Inference Configuration

The baseline inference script uses the OpenAI client and reads configuration from environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` or `MEVERSE_BASE_URL`
- `MEVERSE_TASK`

The root-level [`inference.py`](/home/casp1an/Code/TradeX/inference.py) emits the required `[START]`, `[STEP]`, and `[END]` logs for benchmark evaluation.

## Project Layout

```text
meverse/
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── app.py
    └── meverse_environment.py
```
