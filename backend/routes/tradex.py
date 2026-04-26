"""TradeX dashboard endpoints (mirrors app.py)."""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException

from tradex.compare import run_evaluation
from tradex.env import MarketEnv
from tradex.overseer import Overseer, action_map, encode_observation

from ..schemas import (
    TradexAgentTrade,
    TradexBenchmarkRow,
    TradexCompareRequest,
    TradexCompareResponse,
    TradexEpisodeRequest,
    TradexEpisodeResponse,
    TradexStep,
)

router = APIRouter()

# Lazy-load the overseer once per process so repeated requests do not
# re-initialize the network. The original app.py loaded it at module import.
_OVERSEER: Optional[Overseer] = None


def _get_overseer() -> Overseer:
    global _OVERSEER
    if _OVERSEER is None:
        policy = Overseer()
        if os.path.exists("models/best_model.pth"):
            try:
                policy.load_state_dict(
                    torch.load("models/best_model.pth", map_location="cpu", weights_only=True)
                )
            except Exception:  # noqa: BLE001 - matches app.py behavior
                pass
        policy.eval()
        _OVERSEER = policy
    return _OVERSEER


@router.post("/run-episode", response_model=TradexEpisodeResponse)
def post_run_episode(req: TradexEpisodeRequest) -> TradexEpisodeResponse:
    if not (1 <= req.stage <= 5):
        raise HTTPException(status_code=400, detail="stage must be between 1 and 5")

    env = MarketEnv()
    obs = env.reset(stage=int(req.stage), seed=int(req.seed))

    policy = _get_overseer() if req.use_overseer else None

    correct_blocks = 0
    missed_attacks = 0
    false_positives = 0
    allow_count = 0
    max_threat = 0.0
    steps_out: List[TradexStep] = []

    done = False
    while not done:
        if req.use_overseer and policy is not None:
            obs_vec = encode_observation(obs)
            action_idx, _, _, _, probs = policy.select_action(obs_vec, deterministic=True)
            action_str = action_map[action_idx]
            if action_str == "ALLOW":
                allow_count += 1
            confidence: Optional[float] = float(probs[action_idx]) * 100.0
        else:
            action_str = "ALLOW"
            confidence = None
            allow_count += 1

        old_price = float(env.price)
        obs, reward, done, info = env.step(action_str)
        correct_blocks += int(info.get("correct_detect", 0) or 0)
        missed_attacks += int(info.get("missed_attack", 0) or 0)
        false_positives += int(info.get("false_positive", 0) or 0)

        threat = float(info.get("threat_score", 0.0))
        if threat > max_threat:
            max_threat = threat

        # Build per-agent trade records (executed + blocked intent).
        agent_types = info.get("agent_types", {})
        trades_map: dict = {}
        for trade in info.get("executed_trades", []):
            trades_map[trade["agent"]] = (trade["action"], float(trade["size"]))

        blocked_id = -1
        if action_str.startswith("BLOCK_"):
            try:
                blocked_id = int(action_str.split("_")[1])
            except (ValueError, IndexError):
                blocked_id = -1
            for trade in info.get("intended_trades", []):
                if trade["agent"] == blocked_id:
                    trades_map[trade["agent"]] = (f"BLOCKED_{trade['action']}", float(trade["size"]))

        trade_records: List[TradexAgentTrade] = []
        for agent_id, agent_type in agent_types.items():
            agent_id = int(agent_id)
            if agent_id in trades_map:
                action, size = trades_map[agent_id]
            else:
                action, size = "HOLD", 0.0
            trade_records.append(
                TradexAgentTrade(
                    agent_id=agent_id,
                    agent_type=str(agent_type),
                    action=action,
                    size=round(size, 2),
                )
            )

        decision = action_str
        if action_str.startswith("BLOCK_") and blocked_id != -1 and blocked_id in agent_types:
            decision = f"BLOCK_{agent_types[blocked_id]}"

        steps_out.append(
            TradexStep(
                step=int(env.timestep),
                price_before=round(old_price, 2),
                price_after=round(float(env.price), 2),
                threat_score=round(threat, 3),
                threat_reasons=list(info.get("threat_reasons", []) or []),
                decision=decision,
                confidence=round(confidence, 1) if confidence is not None else None,
                reward=round(float(reward), 3),
                trades=trade_records,
            )
        )

    intervention_rate = (
        ((env.max_steps - allow_count) / env.max_steps) * 100.0 if req.use_overseer else 0.0
    )

    tp, fp, fn = correct_blocks, false_positives, missed_attacks
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    stability_gain = 100.0 - (abs((float(env.price) - 100.0) / 100.0) * 100.0)

    threat_level = (
        "CRITICAL" if max_threat > 0.85 else "ELEVATED" if max_threat > 0.5 else "SAFE"
    )

    return TradexEpisodeResponse(
        threat_level=threat_level,
        max_threat=round(max_threat, 3),
        intervention_rate=round(intervention_rate, 1),
        correct_blocks=correct_blocks,
        missed_attacks=missed_attacks,
        false_positives=false_positives,
        precision=round(precision, 1),
        recall=round(recall, 1),
        stability_gain=round(stability_gain, 1),
        final_price=round(float(env.price), 2),
        steps=steps_out,
    )


@router.post("/compare", response_model=TradexCompareResponse)
def post_compare(req: TradexCompareRequest) -> TradexCompareResponse:
    n = int(req.num_episodes)
    no_overseer = run_evaluation(num_episodes=n, use_overseer=False)
    det = run_evaluation(num_episodes=n, use_overseer=True, deterministic=True)
    stoch = run_evaluation(num_episodes=n, use_overseer=True, deterministic=False)
    rule = run_evaluation(num_episodes=n, use_overseer=True, pure_rule_based=True)

    rows = [
        ("Heuristic Baseline", no_overseer),
        ("PPO Overseer (Det)", det),
        ("PPO Overseer (Stoch)", stoch),
        ("Rule Hybrid", rule),
    ]
    benchmark_rows = [
        TradexBenchmarkRow(
            policy=name,
            avg_reward=round(float(r["avg_reward"]), 2),
            price_error=round(float(r["avg_final_price_error"]), 2),
            precision=round(float(r["precision"]), 2),
            recall=round(float(r["recall"]), 2),
            f1=round(float(r["f1_score"]), 2),
            intervention_rate=round(float(r["intervention_rate"]), 2),
        )
        for name, r in rows
    ]

    best_reward = max(benchmark_rows, key=lambda x: x.avg_reward)
    best_f1 = max(benchmark_rows, key=lambda x: x.f1)
    summary = (
        f"Completed benchmark on {n} episodes.\n"
        f"Best average reward: {best_reward.policy} ({best_reward.avg_reward:.2f}).\n"
        f"Best F1 score: {best_f1.policy} ({best_f1.f1:.2f})."
    )

    return TradexCompareResponse(rows=benchmark_rows, summary=summary)
