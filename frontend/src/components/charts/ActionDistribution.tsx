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

import type { ActionLabelCounts } from "../../api/types";
import { COLORS } from "../../theme/colors";

interface ActionDistributionProps {
  data: ActionLabelCounts;
  height?: number;
}

export function ActionDistribution({ data, height = 320 }: ActionDistributionProps) {
  const rows = data.actions.map((action, i) => ({
    action,
    Normal: data.normal[i],
    Suspicious: data.suspicious[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="action" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 12 }} />
        <YAxis stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} allowDecimals={false} />
        <Tooltip
          contentStyle={{
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.text,
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ color: COLORS.text, fontSize: 12 }} />
        <Bar dataKey="Normal" stackId="a" fill={COLORS.success} radius={[0, 0, 0, 0]} />
        <Bar dataKey="Suspicious" stackId="a" fill={COLORS.danger} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
