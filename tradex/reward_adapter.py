from typing import Dict


def to_trl_reward(env_reward: float, info: Dict, text_action: str) -> float:
    reward = float(env_reward)

    # Reuse existing reward as primary signal.
    # Add small shaping to improve language-policy stability.
    reward += 0.10 * float(info.get("correct_detect", 0.0))
    reward -= 0.05 * float(info.get("false_positive", 0.0))
    reward -= 0.05 * float(info.get("missed_attack", 0.0))
    reward -= 0.03 * float(info.get("price_error", 0.0))

    # Slight regularization against overly aggressive textual actions.
    if text_action == "BLOCK" and not info.get("is_attack_active", False):
        reward -= 0.05

    return reward
