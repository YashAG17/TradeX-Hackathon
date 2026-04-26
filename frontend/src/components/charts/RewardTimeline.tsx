import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Cell,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { RewardTimeline as RewardTimelineData } from "../../api/types";
import { ACTION_COLORS, COLORS } from "../../theme/colors";

interface RewardTimelineProps {
  data: RewardTimelineData;
  height?: number;
}

interface Row {
  step: number;
  reward: number;
  cumulative: number;
  action: string;
  label: string;
}

export function RewardTimeline({ data, height = 320 }: RewardTimelineProps) {
  const rows: Row[] = data.steps.map((step, i) => ({
    step,
    reward: data.rewards[i],
    cumulative: data.cumulative[i],
    action: data.actions[i],
    label: data.labels[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
        <YAxis
          yAxisId="left"
          stroke={COLORS.muted}
          tick={{ fill: COLORS.muted, fontSize: 11 }}
          label={{ value: "Reward", angle: -90, position: "insideLeft", fill: COLORS.muted, fontSize: 11 }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke={COLORS.accent}
          tick={{ fill: COLORS.accent, fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.text,
            fontSize: 12,
          }}
          formatter={(value: number, name: string) => [value.toFixed(3), name]}
          labelFormatter={(label, payload) => {
            const row = payload?.[0]?.payload as Row | undefined;
            return row
              ? `Step ${label} · ${row.action} · ${row.label}`
              : `Step ${label}`;
          }}
        />
        <Bar yAxisId="left" dataKey="reward" name="Reward" radius={[3, 3, 0, 0]}>
          {rows.map((row) => (
            <Cell key={row.step} fill={ACTION_COLORS[row.action] || COLORS.muted} />
          ))}
        </Bar>
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="cumulative"
          name="Cumulative"
          stroke={COLORS.accent}
          strokeWidth={2}
          strokeDasharray="3 3"
          dot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
