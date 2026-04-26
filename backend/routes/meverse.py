"""MEVerse dashboard endpoints (mirrors dashboard.py)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from fastapi import APIRouter, File, HTTPException, UploadFile
from plotly.subplots import make_subplots

from meverse.tasks import list_task_names

from ..episode_runner import (
    ACTIONS_ORDER,
    LABELS_ORDER,
    SIGNAL_NAMES,
    EpisodeResult,
    run_compare,
    run_episode,
)
from ..schemas import (
    ActionLabelCounts,
    AmmSeries,
    ComparePoliciesRequest,
    ComparePoliciesResponse,
    ConfusionMatrix,
    EpisodeSummary,
    GradeBreakdown,
    PolicyResult,
    RewardTimeline,
    RunEpisodeRequest,
    RunEpisodeResponse,
    StepRecord,
    TelemetryResponse,
    TelemetryStep,
)

router = APIRouter()

# Color palette mirrored from dashboard.py:38 so the server-built Plotly figures
# match the frontend theme without the React side having to override layouts.
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
    plot_bgcolor="rgba(26,29,39,0.8)",
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)


def _layout(**updates: Any) -> Dict[str, Any]:
    return {**PLOTLY_LAYOUT, **updates}


# ---------------------------------------------------------------------------
# Plotly figure builders (server-side; serialized to JSON)
# ---------------------------------------------------------------------------


def _signal_heatmap(result: EpisodeResult) -> Dict[str, Any]:
    matrix = np.array(result.signal_matrix).T.tolist() if result.signal_matrix else []
    steps = list(range(len(result.signal_matrix)))
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=steps,
            y=SIGNAL_NAMES,
            colorscale=[
                [0.0, "#0f1117"],
                [0.25, "#0a3d5c"],
                [0.5, "#0088cc"],
                [0.75, "#ffa502"],
                [1.0, "#ff4757"],
            ],
            hovertemplate="Step %{x}<br>%{y}: %{z:.3f}<extra></extra>",
            colorbar=dict(title=dict(text="Intensity", font=dict(size=11))),
        )
    )
    fig.update_layout(
        **_layout(
            title=dict(text="Signal Heatmap Over Time", font=dict(size=14)),
            xaxis_title="Step",
            height=300,
        )
    )
    return fig.to_dict()


def _amm_chart(result: EpisodeResult) -> Dict[str, Any]:
    steps = list(range(len(result.amm_prices)))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=(
            "AMM Price & Liquidity",
            "Bot Confidence, Volatility & Health",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result.amm_prices,
            name="Price",
            line=dict(color=COLORS["accent2"], width=2),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result.amm_liquidity,
            name="Liquidity",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result.amm_bot_conf,
            name="Bot Confidence",
            line=dict(color=COLORS["danger"], width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result.amm_volatility,
            name="Volatility",
            line=dict(color=COLORS["warning"], width=2, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result.amm_health,
            name="Health Index",
            line=dict(color=COLORS["success"], width=2, dash="dashdot"),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False, gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="Liquidity", row=1, col=1, secondary_y=True, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="Value (0-1)", row=2, col=1, range=[-0.05, 1.05], gridcolor=COLORS["border"])
    fig.update_xaxes(title_text="Step", row=2, col=1, gridcolor=COLORS["border"])
    fig.update_xaxes(gridcolor=COLORS["border"], row=1, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=65, r=65, t=88, b=50),
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
        height=540,
    )
    return fig.to_dict()


def _grade_radar(grade: Dict[str, float]) -> Dict[str, Any]:
    categories = ["Detection", "False Positive", "False Negative", "Health", "Overblocking"]
    values = [
        grade["detection_score"],
        grade["false_positive_score"],
        grade["false_negative_score"],
        grade["health_score"],
        grade["overblocking_score"],
    ]
    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,200,167,0.15)",
            line=dict(color=COLORS["accent"], width=2),
            name="Score",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(26,29,39,0.8)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"]),
            angularaxis=dict(gridcolor=COLORS["border"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        title=dict(text="Grade Breakdown", font=dict(size=14)),
        margin=dict(l=60, r=60, t=50, b=40),
        height=360,
        showlegend=False,
    )
    return fig.to_dict()


def _confusion_fig(matrix: List[List[int]]) -> Dict[str, Any]:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=ACTIONS_ORDER,
            y=LABELS_ORDER,
            colorscale=[[0.0, "#1a1d27"], [0.5, "#0088cc"], [1.0, "#00c9a7"]],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            showscale=False,
        )
    )
    fig.update_layout(
        **_layout(
            title=dict(text="Action vs Ground Truth", font=dict(size=14)),
            xaxis_title="Action Taken",
            yaxis_title="True Label",
            height=260,
        )
    )
    return fig.to_dict()


def _amm_gauges(result: EpisodeResult) -> Dict[str, Any]:
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "indicator"}] * 4])
    final = (
        result.amm_prices[-1],
        result.amm_bot_conf[-1],
        result.amm_health[-1],
        result.amm_volatility[-1],
    )
    gauges = [
        ("Price", final[0], 50, 150, COLORS["accent2"]),
        ("Bot Conf", final[1], 0, 1, COLORS["danger"]),
        ("Health", final[2], 0, 1, COLORS["success"]),
        ("Volatility", final[3], 0, 0.5, COLORS["warning"]),
    ]
    for col, (title, value, lo, hi, color) in enumerate(gauges, 1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title=dict(text=title, font=dict(size=13)),
                number=dict(font=dict(size=18), valueformat=".3f"),
                gauge=dict(
                    axis=dict(range=[lo, hi]),
                    bar=dict(color=color),
                    bgcolor=COLORS["surface"],
                    borderwidth=0,
                    steps=[dict(range=[lo, hi], color=COLORS["border"])],
                ),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=20, r=20, t=30, b=10),
        height=180,
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


def _action_label_counts(result: EpisodeResult) -> ActionLabelCounts:
    normal = []
    suspicious = []
    for action in ACTIONS_ORDER:
        normal.append(
            sum(1 for a, l in zip(result.actions, result.labels) if a == action and l == "normal")
        )
        suspicious.append(
            sum(1 for a, l in zip(result.actions, result.labels) if a == action and l == "suspicious")
        )
    return ActionLabelCounts(actions=list(ACTIONS_ORDER), normal=normal, suspicious=suspicious)


def _confusion_matrix(result: EpisodeResult) -> ConfusionMatrix:
    matrix: List[List[int]] = []
    for label in LABELS_ORDER:
        row = [
            sum(1 for a, l in zip(result.actions, result.labels) if a == action and l == label)
            for action in ACTIONS_ORDER
        ]
        matrix.append(row)
    return ConfusionMatrix(actions=list(ACTIONS_ORDER), labels=list(LABELS_ORDER), matrix=matrix)


def _reward_timeline(result: EpisodeResult) -> RewardTimeline:
    cumulative: List[float] = []
    running = 0.0
    for r in result.rewards:
        running += r
        cumulative.append(round(running, 4))
    return RewardTimeline(
        steps=list(range(1, len(result.rewards) + 1)),
        rewards=[round(r, 4) for r in result.rewards],
        cumulative=cumulative,
        actions=list(result.actions),
        labels=list(result.labels),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/tasks")
def get_tasks() -> Dict[str, List[str]]:
    return {"tasks": list_task_names()}


@router.post("/run-episode", response_model=RunEpisodeResponse)
def post_run_episode(req: RunEpisodeRequest) -> RunEpisodeResponse:
    try:
        result = run_episode(req.task, req.policy, req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    grade = result.grade
    total_reward = float(sum(result.rewards))
    avg_reward = total_reward / max(len(result.rewards), 1)

    summary = EpisodeSummary(
        task=result.task_name,
        title=result.task_title,
        difficulty=result.task_difficulty,
        policy=result.policy,
        seed=result.seed,
        steps=len(result.actions),
        total_reward=round(total_reward, 4),
        avg_reward=round(avg_reward, 4),
        grade=GradeBreakdown(
            score=grade["score"],
            detection_score=grade["detection_score"],
            false_positive_score=grade["false_positive_score"],
            false_negative_score=grade["false_negative_score"],
            health_score=grade["health_score"],
            overblocking_score=grade["overblocking_score"],
        ),
    )

    confusion = _confusion_matrix(result)

    return RunEpisodeResponse(
        summary=summary,
        step_history=[StepRecord(**row.__dict__) for row in result.step_rows],
        amm_series=AmmSeries(
            prices=result.amm_prices,
            liquidity=result.amm_liquidity,
            bot_confidence=result.amm_bot_conf,
            volatility=result.amm_volatility,
            health=result.amm_health,
        ),
        signal_matrix=result.signal_matrix,
        signal_names=list(SIGNAL_NAMES),
        reward_timeline=_reward_timeline(result),
        action_label_counts=_action_label_counts(result),
        confusion=confusion,
        plotly={
            "amm_chart": _amm_chart(result),
            "signal_heatmap": _signal_heatmap(result),
            "grade_radar": _grade_radar(grade),
            "amm_gauges": _amm_gauges(result),
            "confusion": _confusion_fig(confusion.matrix),
        },
    )


@router.post("/compare-policies", response_model=ComparePoliciesResponse)
def post_compare_policies(req: ComparePoliciesRequest) -> ComparePoliciesResponse:
    seed = req.seed if req.seed not in (None, 0) else 42
    try:
        results = run_compare(req.task, seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ComparePoliciesResponse(
        task=req.task,
        seed=seed,
        results=[PolicyResult(**r) for r in results],
    )


@router.post("/telemetry", response_model=TelemetryResponse)
async def post_telemetry(file: UploadFile = File(...)) -> TelemetryResponse:
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Upload a .jsonl telemetry file.")
    try:
        raw = (await file.read()).decode("utf-8")
        lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse JSONL: {exc}") from exc

    step_events = [e for e in lines if e.get("event") == "step"]
    if not step_events:
        raise HTTPException(status_code=400, detail="No step events found in telemetry file.")

    start_event = next((e for e in lines if e.get("event") == "episode_start"), {})
    end_event = next((e for e in lines if e.get("event") == "episode_end"), {})

    steps = [
        TelemetryStep(
            step=idx + 1,
            action=str(s.get("action", "?")),
            reward=float(s.get("reward", 0.0)),
        )
        for idx, s in enumerate(step_events)
    ]

    task_name = (
        start_event.get("task")
        or step_events[0].get("decision_observation", {}).get("task_name")
        or "unknown"
    )
    final_score = None
    grade = end_event.get("grade") or {}
    if isinstance(grade, dict) and "score" in grade:
        try:
            final_score = float(grade["score"])
        except (TypeError, ValueError):
            final_score = None

    return TelemetryResponse(
        task=str(task_name),
        model=str(start_event.get("model", "unknown")),
        steps=steps,
        total_reward=round(sum(s.reward for s in steps), 4),
        final_score=final_score,
    )
