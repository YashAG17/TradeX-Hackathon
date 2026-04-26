// Mirrors backend/schemas.py — kept in sync manually.

export interface GradeBreakdown {
  score: number;
  detection_score: number;
  false_positive_score: number;
  false_negative_score: number;
  health_score: number;
  overblocking_score: number;
}

export interface EpisodeSummary {
  task: string;
  title: string;
  difficulty: string;
  policy: string;
  seed: number | null;
  steps: number;
  total_reward: number;
  avg_reward: number;
  grade: GradeBreakdown;
}

export interface StepRecord {
  step: number;
  action: string;
  label: string;
  reward: number;
  burst: number;
  pattern: number;
  suspicion: number;
  manipulation: number;
}

export interface AmmSeries {
  prices: number[];
  liquidity: number[];
  bot_confidence: number[];
  volatility: number[];
  health: number[];
}

export interface RewardTimeline {
  steps: number[];
  rewards: number[];
  cumulative: number[];
  actions: string[];
  labels: string[];
}

export interface ActionLabelCounts {
  actions: string[];
  normal: number[];
  suspicious: number[];
}

export interface ConfusionMatrix {
  actions: string[];
  labels: string[];
  matrix: number[][];
}

export interface PlotlyFigureJSON {
  data: unknown[];
  layout: Record<string, unknown>;
  frames?: unknown[];
}

export interface RunEpisodeResponse {
  summary: EpisodeSummary;
  step_history: StepRecord[];
  amm_series: AmmSeries;
  signal_matrix: number[][];
  signal_names: string[];
  reward_timeline: RewardTimeline;
  action_label_counts: ActionLabelCounts;
  confusion: ConfusionMatrix;
  plotly: {
    amm_chart: PlotlyFigureJSON;
    signal_heatmap: PlotlyFigureJSON;
    grade_radar: PlotlyFigureJSON;
    amm_gauges: PlotlyFigureJSON;
    confusion: PlotlyFigureJSON;
  };
}

export interface PolicyResult {
  policy: string;
  score: number;
  detection: number;
  fp: number;
  fn: number;
  health: number;
  overblock: number;
  total_reward: number;
}

export interface ComparePoliciesResponse {
  task: string;
  seed: number;
  results: PolicyResult[];
}

export interface TelemetryStep {
  step: number;
  action: string;
  reward: number;
}

export interface TelemetryResponse {
  task: string;
  model: string;
  steps: TelemetryStep[];
  total_reward: number;
  final_score: number | null;
}

export interface TradexAgentTrade {
  agent_id: number;
  agent_type: string;
  action: string;
  size: number;
}

export interface TradexStep {
  step: number;
  price_before: number;
  price_after: number;
  threat_score: number;
  threat_reasons: string[];
  decision: string;
  confidence: number | null;
  reward: number;
  trades: TradexAgentTrade[];
}

export interface TradexEpisodeResponse {
  threat_level: "SAFE" | "ELEVATED" | "CRITICAL";
  max_threat: number;
  intervention_rate: number;
  correct_blocks: number;
  missed_attacks: number;
  false_positives: number;
  precision: number;
  recall: number;
  stability_gain: number;
  final_price: number;
  steps: TradexStep[];
}

export interface TradexBenchmarkRow {
  policy: string;
  avg_reward: number;
  price_error: number;
  precision: number;
  recall: number;
  f1: number;
  intervention_rate: number;
}

export interface TradexCompareResponse {
  rows: TradexBenchmarkRow[];
  summary: string;
}
