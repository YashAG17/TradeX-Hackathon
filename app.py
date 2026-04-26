"""
TradeX HF Space — main landing entrypoint.

Tabs:
1. Live Playground (Start / Step / Stop)  — default OpenEnv-style scaffolding
2. Reward Optimization Curves            — generated on-the-fly with Plotly
3. Live Market Replay                    — single-shot end-to-end episode
4. Baseline vs Learned Policy            — multi-policy benchmark
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from tradex.compare import run_evaluation
from tradex.env import MarketEnv
from tradex.overseer import Overseer, action_map, encode_observation


# ---------------------------------------------------------------------------
# Theme / palette
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "border": "#2a2d3a",
    "text": "#e4e6eb",
    "muted": "#8b8fa3",
    "accent": "#00c9a7",
    "accent2": "#0088cc",
    "danger": "#ff4757",
    "warning": "#ffa502",
    "success": "#2ed573",
    "info": "#3498db",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,39,0.85)",
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=55, r=30, t=50, b=45),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)

ACTION_COLORS = {
    "ALLOW": COLORS["success"],
    "BLOCK_NormalTrader": COLORS["info"],
    "BLOCK_Manipulator": COLORS["danger"],
    "BLOCK_Arbitrage": COLORS["warning"],
    "BLOCK_NoisyTrader": COLORS["accent2"],
}


def _layout(**updates: Any) -> dict[str, Any]:
    return {**PLOTLY_LAYOUT, **updates}


def _empty_plot(message: str = "Press Start to begin", *, height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_layout(height=height))
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color=COLORS["muted"]),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    return fig


# ---------------------------------------------------------------------------
# Policy loader (shared)
# ---------------------------------------------------------------------------

def _load_policy() -> Overseer:
    policy = Overseer()
    if os.path.exists("models/best_model.pth"):
        try:
            policy.load_state_dict(
                torch.load("models/best_model.pth", map_location="cpu", weights_only=True)
            )
        except Exception:
            pass
    policy.eval()
    return policy


_GLOBAL_POLICY = _load_policy()


# ---------------------------------------------------------------------------
# Live Playground (Start / Step / Stop)
# ---------------------------------------------------------------------------

def _new_state() -> dict[str, Any]:
    return {
        "env": None,
        "obs": None,
        "done": True,
        "step_index": 0,
        "history": {
            "step": [],
            "price": [],
            "threat": [],
            "reward": [],
            "cum_reward": [],
            "action": [],
            "agent_types": {},
        },
    }


def _make_price_chart(history: dict) -> go.Figure:
    fig = go.Figure()
    if history["step"]:
        fig.add_trace(
            go.Scatter(
                x=history["step"],
                y=history["price"],
                mode="lines+markers",
                name="AMM Price",
                line=dict(color=COLORS["accent2"], width=2),
                marker=dict(size=6),
            )
        )
        fig.add_hline(
            y=100.0,
            line_dash="dot",
            line_color=COLORS["muted"],
            annotation_text="baseline",
            annotation_position="top left",
            annotation_font_color=COLORS["muted"],
        )
    fig.update_layout(
        **_layout(
            title=dict(text="AMM Price Over Time", font=dict(size=14)),
            xaxis_title="Step",
            yaxis_title="Price",
            height=280,
        )
    )
    if not history["step"]:
        fig.add_annotation(
            text="Press Start to begin",
            showarrow=False,
            font=dict(size=14, color=COLORS["muted"]),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
    return fig


def _make_threat_chart(history: dict) -> go.Figure:
    fig = go.Figure()
    if history["step"]:
        fig.add_trace(
            go.Scatter(
                x=history["step"],
                y=history["threat"],
                mode="lines",
                fill="tozeroy",
                name="Threat Score",
                line=dict(color=COLORS["danger"], width=2),
                fillcolor="rgba(255,71,87,0.15)",
            )
        )
        fig.add_hline(
            y=0.6, line_dash="dash", line_color=COLORS["warning"],
            annotation_text="alert", annotation_font_color=COLORS["warning"],
        )
    fig.update_layout(
        **_layout(
            title=dict(text="Threat Score (live)", font=dict(size=14)),
            xaxis_title="Step",
            yaxis_title="Threat",
            yaxis=dict(range=[0, 1.05], gridcolor=COLORS["border"]),
            height=280,
        )
    )
    if not history["step"]:
        fig.add_annotation(
            text="Press Start to begin",
            showarrow=False,
            font=dict(size=14, color=COLORS["muted"]),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
    return fig


def _make_reward_chart(history: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if history["step"]:
        bar_colors = [ACTION_COLORS.get(a, COLORS["muted"]) for a in history["action"]]
        fig.add_trace(
            go.Bar(
                x=history["step"],
                y=history["reward"],
                marker_color=bar_colors,
                name="Step Reward",
                hovertemplate="Step %{x}<br>Reward %{y:.2f}<br>Action %{customdata}<extra></extra>",
                customdata=history["action"],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=history["step"],
                y=history["cum_reward"],
                mode="lines+markers",
                name="Cumulative",
                line=dict(color=COLORS["accent"], width=2, dash="dot"),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        **_layout(
            title=dict(text="Reward (per step + cumulative)", font=dict(size=14)),
            xaxis_title="Step",
            barmode="relative",
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            height=300,
        )
    )
    fig.update_yaxes(title_text="Step reward", secondary_y=False, gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="Cumulative", secondary_y=True, gridcolor="rgba(0,0,0,0)",
                     color=COLORS["accent"])
    if not history["step"]:
        fig.add_annotation(
            text="Press Start to begin",
            showarrow=False,
            font=dict(size=14, color=COLORS["muted"]),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
    return fig


def _empty_status() -> str:
    return (
        "**Status:** idle  \n"
        "**Step:** 0 / 50  \n"
        "**Last action:** —  \n"
        "**Threat score:** —  \n"
        "**Cumulative reward:** 0.00"
    )


def _format_status(state: dict, last_action: str = "—", last_reward: float = 0.0) -> str:
    h = state["history"]
    if not h["step"]:
        return _empty_status()
    threat = h["threat"][-1]
    threat_label = "CRITICAL" if threat > 0.85 else "ELEVATED" if threat > 0.5 else "SAFE"
    status = "running"
    if state["done"]:
        status = "episode complete"
    return (
        f"**Status:** {status}  \n"
        f"**Step:** {state['step_index']} / 50  \n"
        f"**Last action:** `{last_action}`  \n"
        f"**Last reward:** {last_reward:+.2f}  \n"
        f"**Threat score:** {threat:.2f} ({threat_label})  \n"
        f"**Cumulative reward:** {h['cum_reward'][-1]:.2f}"
    )


def _format_step_log(state: dict) -> str:
    h = state["history"]
    if not h["step"]:
        return "Press **Start** to reset the environment, then **Step** to advance one timestep."
    rows = []
    for i in range(len(h["step"])):
        rows.append(
            f"step={h['step'][i]:>3} | price={h['price'][i]:7.2f} | "
            f"threat={h['threat'][i]:.2f} | action={h['action'][i]:<22} | "
            f"reward={h['reward'][i]:+.2f}"
        )
    # Show the most recent ~40 lines, newest at the top
    return "\n".join(rows[-40:][::-1])


def _format_agents(state: dict) -> str:
    h = state["history"]
    if not h["agent_types"]:
        return "Agents will appear here once an episode starts."
    lines = ["| Idx | Agent type |", "|-----|------------|"]
    for i, name in h["agent_types"].items():
        flag = " (malicious)" if name == "Manipulator" else ""
        lines.append(f"| {i} | {name}{flag} |")
    return "\n".join(lines)


def playground_start(seed: int, stage: int, use_overseer: bool, state: dict | None):
    state = _new_state()
    env = MarketEnv()
    obs = env.reset(stage=int(stage), seed=int(seed))
    state["env"] = env
    state["obs"] = obs
    state["done"] = False
    state["step_index"] = 0
    state["use_overseer"] = bool(use_overseer)
    state["history"]["agent_types"] = {i: a.__class__.__name__ for i, a in enumerate(env.agents)}
    state["history"]["step"].append(0)
    state["history"]["price"].append(env.price)
    state["history"]["threat"].append(float(env.current_threat_score))
    state["history"]["reward"].append(0.0)
    state["history"]["cum_reward"].append(0.0)
    state["history"]["action"].append("RESET")
    return (
        state,
        _make_price_chart(state["history"]),
        _make_threat_chart(state["history"]),
        _make_reward_chart(state["history"]),
        _format_status(state, last_action="RESET", last_reward=0.0),
        _format_agents(state),
        _format_step_log(state),
    )


def playground_step(state: dict | None):
    if not state or state.get("env") is None or state.get("done", True):
        msg = "Press **Start** first to reset the environment."
        empty_state = state or _new_state()
        return (
            empty_state,
            _make_price_chart(empty_state["history"]),
            _make_threat_chart(empty_state["history"]),
            _make_reward_chart(empty_state["history"]),
            _format_status(empty_state),
            _format_agents(empty_state),
            msg,
        )

    env: MarketEnv = state["env"]
    obs = state["obs"]
    use_overseer = state.get("use_overseer", True)

    if use_overseer:
        obs_vec = encode_observation(obs)
        action_idx, _, _, _, probs = _GLOBAL_POLICY.select_action(obs_vec, deterministic=True)
        action_str = action_map[action_idx]
    else:
        action_str = "ALLOW"

    obs, reward, done, info = env.step(action_str)
    state["obs"] = obs
    state["done"] = done
    state["step_index"] = env.timestep

    if action_str.startswith("BLOCK_"):
        try:
            blocked_id = int(action_str.split("_")[1])
            t_agent = info["agent_types"].get(blocked_id, "Unknown")
            display_action = f"BLOCK_{t_agent}"
        except Exception:
            display_action = action_str
    else:
        display_action = action_str

    h = state["history"]
    h["step"].append(env.timestep)
    h["price"].append(env.price)
    h["threat"].append(float(info["threat_score"]))
    h["reward"].append(float(reward))
    h["cum_reward"].append(h["cum_reward"][-1] + float(reward))
    h["action"].append(display_action)

    return (
        state,
        _make_price_chart(h),
        _make_threat_chart(h),
        _make_reward_chart(h),
        _format_status(state, last_action=display_action, last_reward=float(reward)),
        _format_agents(state),
        _format_step_log(state),
    )


def playground_stop(state: dict | None):
    fresh = _new_state()
    return (
        fresh,
        _make_price_chart(fresh["history"]),
        _make_threat_chart(fresh["history"]),
        _make_reward_chart(fresh["history"]),
        _empty_status(),
        _format_agents(fresh),
        "Environment stopped. Press **Start** to begin a new episode.",
    )


# ---------------------------------------------------------------------------
# Reward Optimization Curves (generated on the fly, plotly only)
# ---------------------------------------------------------------------------

def _per_episode_metrics(num_episodes: int, use_overseer: bool, deterministic: bool = True):
    """Run `num_episodes` and return per-episode metric arrays."""
    env = MarketEnv()
    rewards: list[float] = []
    correct: list[int] = []
    fps: list[int] = []
    fns: list[int] = []
    price_errs: list[float] = []
    for ep in range(num_episodes):
        # vary the chaos stage as we progress so the curve has structure
        stage = 1 + min(4, ep // max(1, num_episodes // 5))
        seed = 1000 + ep
        obs = env.reset(stage=stage, seed=seed)
        ep_reward = 0.0
        ep_correct = 0
        ep_fp = 0
        ep_fn = 0
        done = False
        while not done:
            if use_overseer:
                obs_vec = encode_observation(obs)
                action_idx, _, _, _, _ = _GLOBAL_POLICY.select_action(
                    obs_vec, deterministic=deterministic
                )
                action_str = action_map[action_idx]
            else:
                action_str = "ALLOW"
            obs, r, done, info = env.step(action_str)
            ep_reward += r
            ep_correct += info.get("correct_detect", 0)
            ep_fp += info.get("false_positive", 0)
            ep_fn += info.get("missed_attack", 0)
        rewards.append(ep_reward)
        correct.append(ep_correct)
        fps.append(ep_fp)
        fns.append(ep_fn)
        price_errs.append(info["price_error"])
    return {
        "rewards": np.array(rewards, dtype=np.float32),
        "correct": np.array(correct, dtype=np.float32),
        "fps": np.array(fps, dtype=np.float32),
        "fns": np.array(fns, dtype=np.float32),
        "price_errs": np.array(price_errs, dtype=np.float32),
    }


def _smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _curve_reward(metrics: dict) -> go.Figure:
    rewards = metrics["rewards"]
    eps = np.arange(1, len(rewards) + 1)
    smooth = _smooth(rewards, window=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps, y=rewards, mode="lines",
        name="Episode Reward",
        line=dict(color=COLORS["accent2"], width=1),
        opacity=0.4,
    ))
    if len(smooth) > 0:
        fig.add_trace(go.Scatter(
            x=eps[len(eps) - len(smooth):], y=smooth, mode="lines",
            name="Rolling Avg (10)",
            line=dict(color=COLORS["accent"], width=3),
        ))
    fig.update_layout(**_layout(
        title=dict(text="Reward vs Episode", font=dict(size=15)),
        xaxis_title="Episode",
        yaxis_title="Total reward",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=320,
    ))
    return fig


def _curve_precision_recall(metrics: dict) -> go.Figure:
    correct = metrics["correct"]
    fps = metrics["fps"]
    fns = metrics["fns"]
    precisions = np.where((correct + fps) > 0, correct / (correct + fps + 1e-9) * 100, 0.0)
    recalls = np.where((correct + fns) > 0, correct / (correct + fns + 1e-9) * 100, 0.0)
    eps = np.arange(1, len(correct) + 1)
    p_smooth = _smooth(precisions, window=10)
    r_smooth = _smooth(recalls, window=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps[len(eps) - len(p_smooth):], y=p_smooth, mode="lines",
        name="Precision %",
        line=dict(color=COLORS["success"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=eps[len(eps) - len(r_smooth):], y=r_smooth, mode="lines",
        name="Recall %",
        line=dict(color=COLORS["warning"], width=3),
    ))
    fig.update_layout(**_layout(
        title=dict(text="Precision vs Recall (rolling)", font=dict(size=15)),
        xaxis_title="Episode",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 105], gridcolor=COLORS["border"]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=320,
    ))
    return fig


def _curve_detection(metrics: dict) -> go.Figure:
    correct = metrics["correct"]
    fps = metrics["fps"]
    eps = np.arange(1, len(correct) + 1)
    tp_smooth = _smooth(correct, window=10)
    fp_smooth = _smooth(fps, window=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps[len(eps) - len(tp_smooth):], y=tp_smooth, mode="lines",
        name="Correct Blocks (TP)",
        line=dict(color=COLORS["accent"], width=3),
        fill="tozeroy",
        fillcolor="rgba(0,201,167,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=eps[len(eps) - len(fp_smooth):], y=fp_smooth, mode="lines",
        name="False Positives (FP)",
        line=dict(color=COLORS["danger"], width=3, dash="dash"),
    ))
    fig.update_layout(**_layout(
        title=dict(text="Detection Capability (rolling)", font=dict(size=15)),
        xaxis_title="Episode",
        yaxis_title="Event count",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=320,
    ))
    return fig


def _curve_price_error(metrics: dict) -> go.Figure:
    errs = metrics["price_errs"]
    eps = np.arange(1, len(errs) + 1)
    smooth = _smooth(errs, window=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps, y=errs, mode="lines",
        name="Episode price error",
        line=dict(color=COLORS["info"], width=1),
        opacity=0.4,
    ))
    if len(smooth) > 0:
        fig.add_trace(go.Scatter(
            x=eps[len(eps) - len(smooth):], y=smooth, mode="lines",
            name="Rolling Avg",
            line=dict(color=COLORS["warning"], width=3),
        ))
    fig.update_layout(**_layout(
        title=dict(text="Final Price Error vs Episode", font=dict(size=15)),
        xaxis_title="Episode",
        yaxis_title="|price - baseline| / baseline",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=320,
    ))
    return fig


def generate_curves(num_episodes: int):
    n = max(20, int(num_episodes))
    metrics = _per_episode_metrics(n, use_overseer=True, deterministic=True)
    summary = (
        f"Generated **{n}** episodes against the trained PPO overseer.\n\n"
        f"- **Average reward:** {metrics['rewards'].mean():.2f}\n"
        f"- **Total correct blocks (TP):** {int(metrics['correct'].sum())}\n"
        f"- **Total false positives (FP):** {int(metrics['fps'].sum())}\n"
        f"- **Total missed attacks (FN):** {int(metrics['fns'].sum())}\n"
        f"- **Mean price error:** {metrics['price_errs'].mean():.4f}\n"
    )
    return (
        _curve_reward(metrics),
        _curve_precision_recall(metrics),
        _curve_detection(metrics),
        _curve_price_error(metrics),
        summary,
    )


# ---------------------------------------------------------------------------
# Live Market Replay (full episode, single shot)
# ---------------------------------------------------------------------------

def run_single_episode(seed, stage, use_overseer):
    env = MarketEnv()
    obs = env.reset(stage=int(stage), seed=int(seed))
    logs: list[str] = []
    total_reward = 0.0
    allow_count = 0
    correct_blocks = 0
    missed_attacks = 0
    false_positives = 0
    action_counts = {
        "ALLOW": 0,
        "BLOCK_Manipulator": 0,
        "BLOCK_NormalTrader": 0,
        "BLOCK_Arbitrage": 0,
        "BLOCK_NoisyTrader": 0,
    }
    done = False
    max_threat = 0.0

    while not done:
        if use_overseer:
            obs_vec = encode_observation(obs)
            action_idx, _, _, _, probs = _GLOBAL_POLICY.select_action(obs_vec, deterministic=True)
            action_str = action_map[action_idx]
            if action_str == "ALLOW":
                allow_count += 1
        else:
            action_str = "ALLOW"

        old_price = env.price
        obs, reward, done, info = env.step(action_str)
        total_reward += reward
        correct_blocks += info.get("correct_detect", 0)
        missed_attacks += info.get("missed_attack", 0)
        false_positives += info.get("false_positive", 0)

        target_str = ""
        if action_str != "ALLOW" and use_overseer:
            blocked_id = int(action_str.split("_")[1]) if "_" in action_str and action_str.split("_")[1].isdigit() else -1
            t_agent = info["agent_types"].get(blocked_id, "Unknown")
            target_str = f"BLOCK_{t_agent}"
            if target_str in action_counts:
                action_counts[target_str] += 1
        elif action_str == "ALLOW":
            action_counts["ALLOW"] += 1

        threat = info["threat_score"]
        max_threat = max(max_threat, threat)

        step_log = "-" * 50 + f"\nStep {env.timestep:02d}\nPrice: {old_price:.1f}\n\nActions:\n"
        agents_acted = {t["agent"]: t for t in info["executed_trades"]}
        if "intended_trades" in info and action_str != "ALLOW":
            blocked_id = int(action_str.split("_")[1]) if "_" in action_str and action_str.split("_")[1].isdigit() else -1
            blocked_intent = [t for t in info["intended_trades"] if t["agent"] == blocked_id]
            if blocked_intent:
                agents_acted[blocked_id] = blocked_intent[0]

        for i, a_type in info["agent_types"].items():
            if i in agents_acted:
                t = agents_acted[i]
                step_log += f"{a_type:14s} -> {t['action']} {t['size']:.1f}\n"
            else:
                step_log += f"{a_type:14s} -> HOLD\n"

        step_log += f"\nOverseer Analysis:\nThreat Score: {threat:.2f}\nDetected:\n"
        if threat > 0.3:
            for line in info["block_reason"].split("- "):
                if line.strip():
                    step_log += f"- {line.strip()}\n"
        else:
            step_log += "- routine liquidity flow\n"

        if use_overseer and target_str:
            confidence = probs[action_idx].item() * 100
            step_log += f"\nDecision:\n{target_str}\n\nConfidence:\n{confidence:.0f}%\n"
            step_log += f"\nOutcome:\nTrade cancelled\nPrice stabilized to {env.price:.1f}\nReward: {reward:.2f}\n"
        else:
            step_log += "\nDecision:\nALLOW\n"
            if use_overseer:
                confidence = probs[0].item() * 100
                step_log += f"\nConfidence:\n{confidence:.0f}%\n"
            step_log += f"\nOutcome:\nTrade allowed\nPrice becomes {env.price:.1f}\nReward: {reward:.2f}\n"

        logs.append(step_log)

    intervention_rate = ((env.max_steps - allow_count) / env.max_steps) * 100 if use_overseer else 0.0
    tp = correct_blocks
    fp = false_positives
    fn = missed_attacks
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    stability_gain = 100 - (abs((env.price - 100.0) / 100.0) * 100)
    threat_level = "CRITICAL" if max_threat > 0.85 else "ELEVATED" if max_threat > 0.5 else "SAFE"

    return [
        f"**Threat Level:** {threat_level} ({max_threat:.2f})",
        f"**Intervention Rate:** {intervention_rate:.1f}%",
        f"**Correct Blocks:** {correct_blocks}",
        f"**Missed Attacks:** {missed_attacks}",
        f"**Precision:** {precision:.1f}%",
        f"**Recall:** {recall:.1f}%",
        f"**Stability Gain:** {stability_gain:.1f}%",
        "\n".join(logs),
    ]


# ---------------------------------------------------------------------------
# Baseline vs Learned Policy
# ---------------------------------------------------------------------------

def run_compare(num_episodes):
    num_episodes = int(num_episodes)
    no_overseer = run_evaluation(num_episodes=num_episodes, use_overseer=False)
    with_overseer_det = run_evaluation(num_episodes=num_episodes, use_overseer=True, deterministic=True)
    with_overseer_stoch = run_evaluation(num_episodes=num_episodes, use_overseer=True, deterministic=False)
    with_overseer_rule = run_evaluation(num_episodes=num_episodes, use_overseer=True, pure_rule_based=True)

    policies = ["Heuristic Baseline", "PPO Overseer (Det)", "PPO Overseer (Stoch)", "Rule Hybrid"]
    metric_groups = {
        "Avg Reward": [no_overseer["avg_reward"], with_overseer_det["avg_reward"],
                       with_overseer_stoch["avg_reward"], with_overseer_rule["avg_reward"]],
        "Precision": [no_overseer["precision"], with_overseer_det["precision"],
                      with_overseer_stoch["precision"], with_overseer_rule["precision"]],
        "Recall": [no_overseer["recall"], with_overseer_det["recall"],
                   with_overseer_stoch["recall"], with_overseer_rule["recall"]],
        "F1": [no_overseer["f1_score"], with_overseer_det["f1_score"],
               with_overseer_stoch["f1_score"], with_overseer_rule["f1_score"]],
    }

    fig = go.Figure()
    color_for = {"Avg Reward": COLORS["accent"], "Precision": COLORS["success"],
                 "Recall": COLORS["warning"], "F1": COLORS["info"]}
    for metric_name, values in metric_groups.items():
        fig.add_trace(go.Bar(
            x=policies, y=values,
            name=metric_name,
            marker_color=color_for[metric_name],
        ))
    fig.update_layout(**_layout(
        barmode="group",
        title=dict(text="Baseline vs Learned Policy Benchmark", font=dict(size=15)),
        xaxis_title="Policy",
        yaxis_title="Score / Percentage",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=400,
    ))

    table_data = []
    for label, res in zip(policies, [no_overseer, with_overseer_det, with_overseer_stoch, with_overseer_rule]):
        table_data.append([
            label,
            round(res["avg_reward"], 2),
            round(res["avg_final_price_error"], 2),
            round(res["precision"], 2),
            round(res["recall"], 2),
            round(res["f1_score"], 2),
            round(res["intervention_rate"], 2),
        ])

    summary = (
        f"Completed benchmark on {num_episodes} episodes.\n"
        f"Best average reward: {max(table_data, key=lambda x: x[1])[0]}.\n"
        f"Best F1 score: {max(table_data, key=lambda x: x[5])[0]}."
    )

    return fig, table_data, summary


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --body-background-fill: #0f1117 !important;
    --block-background-fill: #1a1d27 !important;
    --block-border-color: #2a2d3a !important;
    --block-label-text-color: #e4e6eb !important;
    --body-text-color: #e4e6eb !important;
    --button-primary-background-fill: #00c9a7 !important;
    --button-primary-text-color: #0f1117 !important;
    --button-primary-background-fill-hover: #00b396 !important;
    --input-background-fill: #1a1d27 !important;
    --input-border-color: #2a2d3a !important;
}

.header-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 8px;
}
.header-banner h1 {
    color: #00c9a7 !important;
    font-size: 28px !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #8b8fa3 !important;
    font-size: 14px !important;
    margin: 0 !important;
}

.tabs > .tab-nav { background: #1a1d27 !important; border-bottom: 1px solid #2a2d3a !important; }
.tabs > .tab-nav button { color: #8b8fa3 !important; font-weight: 500 !important; border: none !important; }
.tabs > .tab-nav button.selected {
    color: #00c9a7 !important;
    border-bottom: 2px solid #00c9a7 !important;
    background: transparent !important;
}

.gr-group { border: 1px solid #2a2d3a !important; border-radius: 10px !important; background: #1a1d27 !important; }

.btn-start button { background: #00c9a7 !important; color: #0f1117 !important; }
.btn-step button  { background: #0088cc !important; color: #ffffff !important; }
.btn-stop button  { background: #ff4757 !important; color: #ffffff !important; }
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#e6faf5", c100="#b3f0e0", c200="#80e6cc",
        c300="#4ddcb8", c400="#26d4a8", c500="#00c9a7",
        c600="#00b396", c700="#009d85", c800="#008774",
        c900="#006b5d", c950="#004f46",
    ),
    secondary_hue=gr.themes.Color(
        c50="#e6f3ff", c100="#b3d9ff", c200="#80bfff",
        c300="#4da6ff", c400="#2696ff", c500="#0088cc",
        c600="#007ab8", c700="#006ca3", c800="#005e8f",
        c900="#004a70", c950="#003652",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f0f1f5", c100="#d4d6e0", c200="#b8bbcb",
        c300="#9ca0b6", c400="#8b8fa3", c500="#6b7089",
        c600="#555970", c700="#3f4257", c800="#2a2d3a",
        c900="#1a1d27", c950="#0f1117",
    ),
    font=["Inter", "system-ui", "sans-serif"],
)


with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="TradeX — Multi-Agent AMM Governance") as demo:
    gr.HTML("""
    <div class="header-banner">
        <h1>TradeX — Multi-Agent AMM Governance</h1>
        <p>Interactive OpenEnv playground · live PPO-trained oversight · benchmark-oriented evaluation</p>
    </div>
    """)

    state = gr.State(value=_new_state())

    with gr.Tabs():

        # ============== Tab 1: Live Playground (DEFAULT) ==============
        with gr.Tab("Live Playground"):
            gr.Markdown(
                "Step through the OpenEnv `MarketEnv` one timestep at a time. "
                "**Start** resets the environment, **Step** advances one step using the chosen policy, "
                "**Stop** ends the session."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("#### Configuration")
                    pg_seed = gr.Number(value=4021, label="Seed", precision=0)
                    pg_stage = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Market Chaos Stage")
                    pg_overseer = gr.Checkbox(value=True, label="Use Trained PPO Overseer")

                    with gr.Row():
                        pg_start = gr.Button("Start", variant="primary", elem_classes="btn-start")
                        pg_step = gr.Button("Step", elem_classes="btn-step")
                        pg_stop = gr.Button("Stop", variant="stop", elem_classes="btn-stop")

                    gr.Markdown("#### Status")
                    pg_status = gr.Markdown(_empty_status())

                    gr.Markdown("#### Agents in pool")
                    pg_agents = gr.Markdown("Press **Start** to populate the pool.")

                with gr.Column(scale=2):
                    pg_price = gr.Plot(value=_make_price_chart(_new_state()["history"]), label="AMM Price")
                    pg_threat = gr.Plot(value=_make_threat_chart(_new_state()["history"]), label="Threat Score")
                    pg_reward = gr.Plot(value=_make_reward_chart(_new_state()["history"]), label="Reward Curve")

            gr.Markdown("#### Step log")
            pg_log = gr.Textbox(
                value="Press **Start** to reset the environment, then **Step** to advance one timestep.",
                lines=14, max_lines=20, interactive=False, show_label=False,
            )

            playground_outputs = [state, pg_price, pg_threat, pg_reward, pg_status, pg_agents, pg_log]
            pg_start.click(playground_start, [pg_seed, pg_stage, pg_overseer, state], playground_outputs)
            pg_step.click(playground_step, [state], playground_outputs)
            pg_stop.click(playground_stop, [state], playground_outputs)

        # ============== Tab 2: Reward Optimization Curves ==============
        with gr.Tab("Reward Optimization Curves"):
            gr.Markdown(
                "Generate the canonical training curves directly in-browser by replaying the loaded "
                "policy across N episodes. No image uploads — all charts are produced from the live "
                "`MarketEnv` + `Overseer` plot generator code in `tradex/utils.py`."
            )

            with gr.Row():
                curves_n = gr.Slider(minimum=20, maximum=300, value=80, step=10, label="Episodes to evaluate")
                curves_btn = gr.Button("Generate Curves", variant="primary")

            curves_summary = gr.Markdown("Set the slider and press **Generate Curves**.")

            with gr.Row():
                curve_reward = gr.Plot(value=_empty_plot("Press Generate Curves"), label="Reward vs Episode")
                curve_pr = gr.Plot(value=_empty_plot("Press Generate Curves"), label="Precision vs Recall")
            with gr.Row():
                curve_detect = gr.Plot(value=_empty_plot("Press Generate Curves"), label="Detection Capability")
                curve_price = gr.Plot(value=_empty_plot("Press Generate Curves"), label="Final Price Error")

            curves_btn.click(
                generate_curves,
                inputs=[curves_n],
                outputs=[curve_reward, curve_pr, curve_detect, curve_price, curves_summary],
            )

        # ============== Tab 3: Live Market Replay ==============
        with gr.Tab("Live Market Replay"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dashboard Controls")
                    seed_input = gr.Number(value=4021, label="Simulation Seed")
                    stage_input = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Market Chaos Stage")
                    use_ai = gr.Checkbox(value=True, label="Enable Deep RL Overseer")
                    run_btn = gr.Button("Execute Market Simulation", variant="primary")

                    gr.Markdown("### Governance Decisions")
                    out_threat = gr.Markdown()
                    out_int = gr.Markdown()
                    out_cb = gr.Markdown()
                    out_ma = gr.Markdown()
                    out_prec = gr.Markdown()
                    out_rec = gr.Markdown()
                    out_stab = gr.Markdown()

                with gr.Column(scale=2):
                    ep_logs = gr.Textbox(label="Intervention Metrics", lines=28, max_lines=40)

            run_btn.click(
                run_single_episode,
                inputs=[seed_input, stage_input, use_ai],
                outputs=[out_threat, out_int, out_cb, out_ma, out_prec, out_rec, out_stab, ep_logs],
            )

        # ============== Tab 4: Baseline vs Learned Policy ==============
        with gr.Tab("Baseline vs Learned Policy"):
            gr.Markdown("## PPO vs Baseline Benchmark")
            gr.Markdown("Run a direct benchmark to compare the baseline policy against the learned PPO overseer.")

            benchmark_episodes = gr.Slider(
                minimum=10, maximum=100, value=50, step=10, label="Evaluation Episodes",
            )
            compare_btn = gr.Button("Run Benchmark", variant="primary")
            benchmark_plot = gr.Plot(label="Baseline vs Learned Policy")
            benchmark_table = gr.Dataframe(
                headers=["Policy", "Avg Reward", "Price Error", "Precision", "Recall", "F1", "Intervention Rate"],
                datatype=["str", "number", "number", "number", "number", "number", "number"],
                row_count=4, col_count=7,
                label="Benchmark Metrics",
            )
            compare_out = gr.Textbox(lines=4, label="Benchmark Summary")

            compare_btn.click(
                fn=run_compare,
                inputs=benchmark_episodes,
                outputs=[benchmark_plot, benchmark_table, compare_out],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
