"""Pure-Python MEVerse episode runner factored out of dashboard.py.

The original `run_full_episode` in `dashboard.py` interleaves env stepping with
Plotly figure construction. The frontend wants the data, so we split: this
module produces a structured `EpisodeResult`, and `routes/meverse.py` decides
which slices to send as raw JSON vs as Plotly figures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from meverse.baseline_policy import choose_surveillance_action
from meverse.models import SurveillanceAction
from meverse.server.meverse_environment import MarketSurveillanceEnvironment
from meverse.tasks import list_task_names, task_definition

VALID_POLICIES = ["Heuristic", "Always Allow", "Random"]
ACTIONS_ORDER = ["ALLOW", "FLAG", "BLOCK", "MONITOR"]
LABELS_ORDER = ["normal", "suspicious"]
SIGNAL_NAMES = [
    "Burst",
    "Pattern",
    "Suspicion",
    "Manipulation",
    "Frequency",
    "Slippage",
]


@dataclass
class StepRow:
    step: int
    action: str
    label: str
    reward: float
    burst: float
    pattern: float
    suspicion: float
    manipulation: float


@dataclass
class EpisodeResult:
    task_name: str
    task_title: str
    task_difficulty: str
    policy: str
    seed: Optional[int]

    actions: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    amm_prices: List[float] = field(default_factory=list)
    amm_liquidity: List[float] = field(default_factory=list)
    amm_bot_conf: List[float] = field(default_factory=list)
    amm_volatility: List[float] = field(default_factory=list)
    amm_health: List[float] = field(default_factory=list)

    signal_matrix: List[List[float]] = field(default_factory=list)
    step_rows: List[StepRow] = field(default_factory=list)
    grade: Dict[str, Any] = field(default_factory=dict)


def _signal_row(obs) -> List[float]:
    return [
        obs.burst_indicator,
        obs.pattern_indicator,
        obs.suspiciousness_score,
        obs.manipulation_score,
        obs.trade_frequency / 10.0,
        obs.recent_slippage_impact * 10.0,
    ]


def _select_action(policy: str, obs, rng: random.Random) -> str:
    if policy == "Heuristic":
        return choose_surveillance_action(obs)
    if policy == "Random":
        return rng.choice(ACTIONS_ORDER)
    return "ALLOW"


def validate_inputs(task: str, policy: str) -> Optional[str]:
    if task not in list_task_names():
        return f"Invalid task. Choose from: {', '.join(list_task_names())}"
    if policy not in VALID_POLICIES:
        return f"Invalid policy. Choose from: {', '.join(VALID_POLICIES)}"
    return None


def run_episode(
    task: str,
    policy: str,
    seed: Optional[int],
) -> EpisodeResult:
    """Run a complete MEVerse surveillance episode and capture all telemetry."""

    err = validate_inputs(task, policy)
    if err:
        raise ValueError(err)

    effective_seed = seed if seed not in (None, 0) else None
    task_def = task_definition(task)

    env = MarketSurveillanceEnvironment(
        task=task,
        eval_mode=(effective_seed is not None),
        demo_mode=(effective_seed is None),
    )
    obs = env.reset(task=task, seed=effective_seed)
    rng = random.Random(effective_seed)

    result = EpisodeResult(
        task_name=task,
        task_title=task_def.title,
        task_difficulty=task_def.difficulty,
        policy=policy,
        seed=seed,
    )

    snap = env.debug_snapshot()
    result.amm_prices.append(snap["amm_state"]["price"])
    result.amm_liquidity.append(snap["amm_state"]["liquidity"])
    result.amm_bot_conf.append(snap["amm_state"]["bot_confidence"])
    result.amm_volatility.append(snap["amm_state"]["volatility"])
    result.amm_health.append(snap["amm_state"]["health_index"])
    result.signal_matrix.append(_signal_row(obs))

    step_count = 0
    while not obs.done:
        action = _select_action(policy, obs, rng)

        pre_snap = env.debug_snapshot()
        label = pre_snap["current_step"]["label"] if pre_snap["current_step"] else "normal"

        obs = env.step(SurveillanceAction(action_type=action))
        step_count += 1
        reward = float(obs.reward or 0.0)

        result.actions.append(action)
        result.labels.append(label)
        result.rewards.append(reward)

        post_snap = env.debug_snapshot()
        result.amm_prices.append(post_snap["amm_state"]["price"])
        result.amm_liquidity.append(post_snap["amm_state"]["liquidity"])
        result.amm_bot_conf.append(post_snap["amm_state"]["bot_confidence"])
        result.amm_volatility.append(post_snap["amm_state"]["volatility"])
        result.amm_health.append(post_snap["amm_state"]["health_index"])

        if not obs.done:
            result.signal_matrix.append(_signal_row(obs))

        result.step_rows.append(
            StepRow(
                step=step_count,
                action=action,
                label=label,
                reward=reward,
                burst=obs.burst_indicator,
                pattern=obs.pattern_indicator,
                suspicion=obs.suspiciousness_score,
                manipulation=obs.manipulation_score,
            )
        )

    result.grade = env.grade()
    return result


def run_compare(task: str, seed: int) -> List[Dict[str, float]]:
    """Run the three baseline policies on the same seed and return raw scores."""

    if task not in list_task_names():
        raise ValueError(f"Invalid task. Choose from: {', '.join(list_task_names())}")

    out: List[Dict[str, float]] = []
    rng = random.Random(seed)
    for policy in VALID_POLICIES:
        env = MarketSurveillanceEnvironment(task=task, eval_mode=True, demo_mode=False)
        obs = env.reset(task=task, seed=seed)
        rewards: List[float] = []
        while not obs.done:
            action = _select_action(policy, obs, rng)
            obs = env.step(SurveillanceAction(action_type=action))
            rewards.append(float(obs.reward or 0.0))
        grade = env.grade()
        out.append(
            {
                "policy": policy,
                "score": float(grade["score"]),
                "detection": float(grade["detection_score"]),
                "fp": float(grade["false_positive_score"]),
                "fn": float(grade["false_negative_score"]),
                "health": float(grade["health_score"]),
                "overblock": float(grade["overblocking_score"]),
                "total_reward": float(sum(rewards)),
            }
        )
    return out
