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

import type { TradexBenchmarkRow } from "../../api/types";
import { COLORS } from "../../theme/colors";

interface TradexBenchmarkProps {
  rows: TradexBenchmarkRow[];
  height?: number;
}

const METRICS: Array<{ key: keyof TradexBenchmarkRow; label: string; color: string }> = [
  { key: "avg_reward", label: "Avg Reward", color: COLORS.accent },
  { key: "precision", label: "Precision", color: COLORS.accent2 },
  { key: "recall", label: "Recall", color: COLORS.success },
  { key: "f1", label: "F1", color: COLORS.warning },
];

export function TradexBenchmark({ rows, height = 380 }: TradexBenchmarkProps) {
  const data = rows.map((r) => {
    const row: Record<string, number | string> = { policy: r.policy };
    for (const m of METRICS) {
      row[m.label] = r[m.key] as number;
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="policy" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
        <YAxis stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.text,
            fontSize: 12,
          }}
          formatter={(value: number) => value.toFixed(2)}
        />
        <Legend wrapperStyle={{ color: COLORS.text, fontSize: 11 }} />
        {METRICS.map((m) => (
          <Bar key={m.label} dataKey={m.label} fill={m.color} radius={[2, 2, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
