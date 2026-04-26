import type { StepRecord } from "../api/types";
import { ACTION_COLORS, LABEL_COLORS } from "../theme/colors";

interface StepTableProps {
  rows: StepRecord[];
  maxHeight?: number;
}

const HEADERS = [
  "Step",
  "Action",
  "Label",
  "Reward",
  "Burst",
  "Pattern",
  "Suspicion",
  "Manipulation",
];

export function StepTable({ rows, maxHeight = 360 }: StepTableProps) {
  return (
    <div
      className="overflow-auto rounded-lg border border-border animate-table-slide"
      style={{ maxHeight }}
    >
      <table className="w-full text-sm">
        <thead className="bg-bg sticky top-0 z-[1]">
          <tr>
            {HEADERS.map((h) => (
              <th
                key={h}
                className="text-left text-xs font-semibold uppercase tracking-wide text-accent px-3 py-2 border-b border-border"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.step} className="hover:bg-accent/5">
              <td className="px-3 py-2 border-b border-border tabular-nums">{row.step}</td>
              <td className="px-3 py-2 border-b border-border">
                <span
                  className="inline-block px-2 py-0.5 rounded text-xs font-semibold"
                  style={{
                    color: ACTION_COLORS[row.action] || "#888",
                    backgroundColor: `${ACTION_COLORS[row.action] || "#888"}22`,
                  }}
                >
                  {row.action}
                </span>
              </td>
              <td className="px-3 py-2 border-b border-border">
                <span
                  className="text-xs font-medium"
                  style={{ color: LABEL_COLORS[row.label] || "#888" }}
                >
                  {row.label}
                </span>
              </td>
              <td className="px-3 py-2 border-b border-border tabular-nums">
                {row.reward.toFixed(3)}
              </td>
              <td className="px-3 py-2 border-b border-border tabular-nums">
                {row.burst.toFixed(3)}
              </td>
              <td className="px-3 py-2 border-b border-border tabular-nums">
                {row.pattern.toFixed(3)}
              </td>
              <td className="px-3 py-2 border-b border-border tabular-nums">
                {row.suspicion.toFixed(3)}
              </td>
              <td className="px-3 py-2 border-b border-border tabular-nums">
                {row.manipulation.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
