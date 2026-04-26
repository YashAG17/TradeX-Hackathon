// Mirrors the COLORS dict in dashboard.py:38 so React charts and Plotly figures
// share one source of truth.
export const COLORS = {
  bg: "#0f1117",
  surface: "#1a1d27",
  border: "#2a2d3a",
  text: "#e4e6eb",
  muted: "#8b8fa3",
  accent: "#00c9a7",
  accent2: "#0088cc",
  danger: "#ff4757",
  warning: "#ffa502",
  success: "#2ed573",
  info: "#3498db",
} as const;

export const ACTION_COLORS: Record<string, string> = {
  ALLOW: COLORS.success,
  FLAG: COLORS.warning,
  BLOCK: COLORS.danger,
  MONITOR: COLORS.info,
};

export const LABEL_COLORS: Record<string, string> = {
  suspicious: COLORS.danger,
  normal: COLORS.success,
};
