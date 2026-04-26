import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { meverseApi } from "../../api/meverse";
import { ApiError } from "../../api/client";
import { ChartCard } from "../../components/ChartCard";
import { ACTION_COLORS, COLORS } from "../../theme/colors";

export function TelemetryViewer() {
  const [fileName, setFileName] = useState<string | null>(null);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => meverseApi.uploadTelemetry(file),
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    uploadMutation.mutate(file);
  }

  const data = uploadMutation.data;
  const error = uploadMutation.error;

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="font-semibold mb-1">Upload telemetry</h3>
        <p className="text-muted text-sm mb-3">
          Drop a <code className="text-text">.jsonl</code> file produced by{" "}
          <code className="text-text">inference.py</code> to replay rewards and grade.
        </p>
        <label className="block border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:border-accent hover:bg-accent/5 transition-colors">
          <input
            type="file"
            accept=".jsonl,application/x-ndjson"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="text-muted text-sm">
            {fileName ? <span className="text-text">{fileName}</span> : "Click to choose .jsonl"}
          </div>
        </label>
        {error ? (
          <p className="text-danger text-xs mt-2">
            {(error as ApiError).detail || (error as Error).message}
          </p>
        ) : null}
      </div>

      {uploadMutation.isPending ? (
        <div className="card text-muted text-sm">Parsing...</div>
      ) : null}

      {data ? (
        <>
          <ChartCard title="Telemetry Replay">
            <div className="grid sm:grid-cols-4 gap-3 mb-4 text-sm">
              <Stat label="Task" value={data.task} />
              <Stat label="Model" value={data.model} />
              <Stat label="Steps" value={data.steps.length} />
              <Stat label="Total reward" value={data.total_reward.toFixed(3)} />
            </div>
            {data.final_score !== null ? (
              <div className="text-sm mb-4">
                <span className="label-text mr-2">Final score</span>
                <span className="text-accent font-semibold">{data.final_score.toFixed(4)}</span>
              </div>
            ) : null}

            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data.steps} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="step" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
                <YAxis stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: COLORS.surface,
                    border: `1px solid ${COLORS.border}`,
                    borderRadius: 8,
                    color: COLORS.text,
                    fontSize: 12,
                  }}
                  formatter={(value: number, _name: string, props) => [
                    value.toFixed(3),
                    `Reward (${props.payload.action})`,
                  ]}
                />
                <Bar dataKey="reward" radius={[2, 2, 0, 0]}>
                  {data.steps.map((step) => (
                    <Cell key={step.step} fill={ACTION_COLORS[step.action] || COLORS.muted} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="label-text">{label}</div>
      <div className="text-text font-medium break-words">{value}</div>
    </div>
  );
}
