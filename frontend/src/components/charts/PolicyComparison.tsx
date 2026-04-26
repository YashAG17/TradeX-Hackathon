import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PolicyResult } from "../../api/types";
import { COLORS } from "../../theme/colors";

interface PolicyComparisonProps {
  results: PolicyResult[];
  height?: number;
}

const METRICS: Array<{ key: keyof PolicyResult; label: string; color: string }> = [
  { key: "score", label: "Final Score", color: COLORS.accent },
  { key: "detection", label: "Detection", color: COLORS.accent2 },
  { key: "fp", label: "False Pos.", color: COLORS.success },
  { key: "fn", label: "False Neg.", color: COLORS.danger },
  { key: "health", label: "Health", color: COLORS.info },
  { key: "overblock", label: "Overblocking", color: COLORS.warning },
];

export function PolicyComparison({ results, height = 380 }: PolicyComparisonProps) {
  const rows = results.map((r) => {
    const row: Record<string, number | string> = { policy: r.policy };
    for (const m of METRICS) {
      row[m.label] = r[m.key] as number;
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="policy" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 12 }} />
        <YAxis stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} domain={[0, 1]} />
        <Tooltip
          contentStyle={{
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.text,
            fontSize: 12,
          }}
          formatter={(value: number) => value.toFixed(4)}
        />
        <Legend wrapperStyle={{ color: COLORS.text, fontSize: 11 }} />
        {METRICS.map((m) => (
          <Bar key={m.label} dataKey={m.label} fill={m.color} radius={[2, 2, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
