import re
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are the TradeX Overseer.\n"
    "Your job is to protect AMM markets from manipulation while minimizing false positives.\n"
    "Choose one action:\n"
    "ALLOW\n"
    "MONITOR\n"
    "FLAG\n"
    "BLOCK"
)

VALID_TEXT_ACTIONS = ("ALLOW", "MONITOR", "FLAG", "BLOCK")


def _bucketize(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Low"
    if value >= high:
        return "High"
    return "Medium"


def _estimate_suspicious_agent(obs: Dict) -> int:
    stats: List[Dict] = obs.get("stats", [])
    if not stats:
        return 0

    scores = []
    for idx, s in enumerate(stats):
        # Heuristic suspicion estimate from existing environment features.
        score = (
            0.6 * float(s.get("burst_score", 0.0))
            + 0.8 * float(s.get("dump_score", 0.0))
            + 0.4 * float(s.get("pump_score", 0.0))
            + 0.2 * float(s.get("avg_size", 0.0))
            - 0.1 * float(s.get("blocked", 0.0))
        )
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return int(scores[0][0])


def observation_to_prompt(obs: Dict) -> str:
    price = float(obs.get("price", 100.0))
    volatility = float(obs.get("volatility", 0.0))
    liquidity = float(obs.get("liquidity", 0.0))
    threat = float(obs.get("threat_score", 0.0))
    momentum = float(obs.get("momentum", 0.0))

    volatility_bucket = _bucketize(volatility, 0.8, 2.0)
    liquidity_bucket = _bucketize(liquidity, 120000.0, 220000.0)
    burst_active = "Yes" if threat > 0.55 else "No"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Market State:\n"
        f"Price: {price:.2f}\n"
        f"Momentum: {momentum:.3f}\n"
        f"Volatility: {volatility_bucket} ({volatility:.3f})\n"
        f"Liquidity: {liquidity_bucket} ({liquidity:.2f})\n"
        f"Threat Score: {threat:.2f}\n"
        f"Recent Burst Activity: {burst_active}\n"
        f"Step: {int(obs.get('timestep', 0))}\n"
        f"Stage: {int(obs.get('stage', 1))}\n\n"
        "Choose governance action:\n"
        "ALLOW / MONITOR / FLAG / BLOCK\n\n"
        "Action:"
    )


def parse_model_action(text: str) -> str:
    if not text:
        return "ALLOW"

    upper = text.strip().upper()
    for action in VALID_TEXT_ACTIONS:
        if re.search(rf"\b{action}\b", upper):
            return action
    return "ALLOW"


def text_action_to_env_action(text_action: str, obs: Dict) -> str:
    action = parse_model_action(text_action)
    if action in ("ALLOW", "MONITOR"):
        return "ALLOW"
    if action == "FLAG":
        # Conservative map: FLAG records caution but avoids hard intervention.
        return "ALLOW"

    target_id = _estimate_suspicious_agent(obs)
    return f"BLOCK_{target_id}"
