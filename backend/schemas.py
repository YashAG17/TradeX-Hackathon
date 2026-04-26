"""Pydantic request/response models for the dashboard API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# MEVerse - run episode
# ---------------------------------------------------------------------------


class RunEpisodeRequest(BaseModel):
    task: str = Field(..., description="Task name from /api/meverse/tasks")
    policy: str = Field(..., description="One of Heuristic, Always Allow, Random")
    seed: Optional[int] = Field(default=None, ge=0, le=999_999)


class StepRecord(BaseModel):
    step: int
    action: str
    label: str
    reward: float
    burst: float
    pattern: float
    suspicion: float
    manipulation: float


class AmmSeries(BaseModel):
    prices: List[float]
    liquidity: List[float]
    bot_confidence: List[float]
    volatility: List[float]
    health: List[float]


class GradeBreakdown(BaseModel):
    score: float
    detection_score: float
    false_positive_score: float
    false_negative_score: float
    health_score: float
    overblocking_score: float


class EpisodeSummary(BaseModel):
    task: str
    title: str
    difficulty: str
    policy: str
    seed: Optional[int]
    steps: int
    total_reward: float
    avg_reward: float
    grade: GradeBreakdown


class ActionLabelCounts(BaseModel):
    actions: List[str]
    normal: List[int]
    suspicious: List[int]


class RewardTimeline(BaseModel):
    steps: List[int]
    rewards: List[float]
    cumulative: List[float]
    actions: List[str]
    labels: List[str]


class ConfusionMatrix(BaseModel):
    actions: List[str]
    labels: List[str]
    matrix: List[List[int]]


class RunEpisodeResponse(BaseModel):
    summary: EpisodeSummary
    step_history: List[StepRecord]
    amm_series: AmmSeries
    signal_matrix: List[List[float]]
    signal_names: List[str]
    reward_timeline: RewardTimeline
    action_label_counts: ActionLabelCounts
    confusion: ConfusionMatrix
    plotly: Dict[str, Any]


# ---------------------------------------------------------------------------
# MEVerse - compare policies
# ---------------------------------------------------------------------------


class ComparePoliciesRequest(BaseModel):
    task: str
    seed: Optional[int] = Field(default=42, ge=0, le=999_999)


class PolicyResult(BaseModel):
    policy: str
    score: float
    detection: float
    fp: float
    fn: float
    health: float
    overblock: float
    total_reward: float


class ComparePoliciesResponse(BaseModel):
    task: str
    seed: int
    results: List[PolicyResult]


# ---------------------------------------------------------------------------
# MEVerse - telemetry viewer
# ---------------------------------------------------------------------------


class TelemetryStep(BaseModel):
    step: int
    action: str
    reward: float


class TelemetryResponse(BaseModel):
    task: str
    model: str
    steps: List[TelemetryStep]
    total_reward: float
    final_score: Optional[float]


# ---------------------------------------------------------------------------
# TradeX - live market replay
# ---------------------------------------------------------------------------


class TradexEpisodeRequest(BaseModel):
    seed: int = 4021
    stage: int = Field(default=5, ge=1, le=5)
    use_overseer: bool = True


class TradexAgentTrade(BaseModel):
    agent_id: int
    agent_type: str
    action: str  # BUY, SELL, HOLD, BLOCKED
    size: float


class TradexStep(BaseModel):
    step: int
    price_before: float
    price_after: float
    threat_score: float
    threat_reasons: List[str]
    decision: str
    confidence: Optional[float]
    reward: float
    trades: List[TradexAgentTrade]


class TradexEpisodeResponse(BaseModel):
    threat_level: str
    max_threat: float
    intervention_rate: float
    correct_blocks: int
    missed_attacks: int
    false_positives: int
    precision: float
    recall: float
    stability_gain: float
    final_price: float
    steps: List[TradexStep]


# ---------------------------------------------------------------------------
# TradeX - benchmark compare
# ---------------------------------------------------------------------------


class TradexCompareRequest(BaseModel):
    num_episodes: int = Field(default=50, ge=10, le=200)


class TradexBenchmarkRow(BaseModel):
    policy: str
    avg_reward: float
    price_error: float
    precision: float
    recall: float
    f1: float
    intervention_rate: float


class TradexCompareResponse(BaseModel):
    rows: List[TradexBenchmarkRow]
    summary: str
