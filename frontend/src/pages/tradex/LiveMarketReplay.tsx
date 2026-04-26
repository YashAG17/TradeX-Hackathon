import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { tradexApi, type TradexEpisodePayload } from "../../api/tradex";
import { ApiError } from "../../api/client";
import { ChartCard } from "../../components/ChartCard";
import { KpiTile } from "../../components/KpiTile";
import type { TradexStep } from "../../api/types";
import { COLORS } from "../../theme/colors";

const TONE_BY_LEVEL: Record<string, "success" | "warning" | "danger"> = {
  SAFE: "success",
  ELEVATED: "warning",
  CRITICAL: "danger",
};

export function LiveMarketReplay() {
  const [seed, setSeed] = useState<number>(4021);
  const [stage, setStage] = useState<number>(5);
  const [useOverseer, setUseOverseer] = useState<boolean>(true);

  const runMutation = useMutation({
    mutationFn: (payload: TradexEpisodePayload) => tradexApi.runEpisode(payload),
  });

  function handleRun(e: React.FormEvent) {
    e.preventDefault();
    runMutation.mutate({ seed, stage, use_overseer: useOverseer });
  }

  const data = runMutation.data;
  const error = runMutation.error;

  return (
    <div className="grid lg:grid-cols-[280px_1fr] gap-6">
      <aside>
        <form onSubmit={handleRun} className="card space-y-4">
          <h3 className="font-semibold">Dashboard Controls</h3>

          <div>
            <label htmlFor="seed" className="label-text block mb-1">Simulation Seed</label>
            <input
              id="seed"
              type="number"
              className="input-field"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>

          <div>
            <label htmlFor="stage" className="label-text block mb-1">
              Market Chaos Stage: <span className="text-text">{stage}</span>
            </label>
            <input
              id="stage"
              type="range"
              min={1}
              max={5}
              step={1}
              value={stage}
              onChange={(e) => setStage(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useOverseer}
              onChange={(e) => setUseOverseer(e.target.checked)}
              className="accent-accent"
            />
            Enable Deep RL Overseer
          </label>

          <button type="submit" className="btn-primary w-full" disabled={runMutation.isPending}>
            {runMutation.isPending ? "Simulating..." : "Execute Market Simulation"}
          </button>

          {error ? (
            <p className="text-danger text-xs">
              {(error as ApiError).detail || (error as Error).message}
            </p>
          ) : null}
        </form>
      </aside>

      <section className="space-y-6">
        {!data && !runMutation.isPending ? (
          <div className="card text-muted text-sm">
            Configure parameters and execute a simulation to see governance decisions.
          </div>
        ) : null}

        {runMutation.isPending ? (
          <div className="card text-muted text-sm">Running simulation...</div>
        ) : null}

        {data ? (
          <>
            <div className="grid sm:grid-cols-3 gap-4">
              <KpiTile
                label="Threat Level"
                value={`${data.threat_level} (${data.max_threat.toFixed(2)})`}
                tone={TONE_BY_LEVEL[data.threat_level]}
                delay={0}
              />
              <KpiTile
                label="Intervention Rate"
                value={`${data.intervention_rate.toFixed(1)}%`}
                tone="info"
                delay={50}
              />
              <KpiTile
                label="Final Price"
                value={data.final_price.toFixed(2)}
                hint={`Stability gain ${data.stability_gain.toFixed(1)}%`}
                delay={100}
              />
              <KpiTile label="Correct Blocks" value={data.correct_blocks} tone="success" delay={150} />
              <KpiTile label="Missed Attacks" value={data.missed_attacks} tone="danger" delay={200} />
              <KpiTile label="False Positives" value={data.false_positives} tone="warning" delay={250} />
              <KpiTile label="Precision" value={`${data.precision.toFixed(1)}%`} delay={300} />
              <KpiTile label="Recall" value={`${data.recall.toFixed(1)}%`} delay={350} />
              <KpiTile label="Stability Gain" value={`${data.stability_gain.toFixed(1)}%`} delay={400} />
            </div>

            <ChartCard title="Price Trajectory">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart
                  data={data.steps.map((s) => ({ step: s.step, price: s.price_after }))}
                  margin={{ top: 16, right: 24, bottom: 8, left: 8 }}
                >
                  <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="step" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
                  <YAxis stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{
                      background: COLORS.surface,
                      border: `1px solid ${COLORS.border}`,
                      borderRadius: 8,
                      color: COLORS.text,
                      fontSize: 12,
                    }}
                  />
                  <Line type="monotone" dataKey="price" stroke={COLORS.accent} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Per-step Intervention Log">
              <div className="space-y-3 max-h-[640px] overflow-auto pr-2">
                {data.steps.map((step) => (
                  <StepCard key={step.step} step={step} />
                ))}
              </div>
            </ChartCard>
          </>
        ) : null}
      </section>
    </div>
  );
}

function StepCard({ step }: { step: TradexStep }) {
  const blocked = step.decision !== "ALLOW";
  const decisionColor = blocked ? COLORS.danger : COLORS.success;

  return (
    <div className="border border-border rounded-lg p-4 bg-bg/50 animate-fade-up">
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="label-text mr-2">Step</span>
          <span className="font-semibold">{String(step.step).padStart(2, "0")}</span>
          <span className="text-muted mx-2">·</span>
          <span className="text-muted text-xs">
            Price {step.price_before.toFixed(1)} → {step.price_after.toFixed(1)}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-muted">Threat</span>
          <span style={{ color: step.threat_score > 0.5 ? COLORS.danger : COLORS.text }}>
            {step.threat_score.toFixed(2)}
          </span>
          <span
            className="px-2 py-0.5 rounded text-xs font-semibold"
            style={{ color: decisionColor, backgroundColor: `${decisionColor}22` }}
          >
            {step.decision}
          </span>
          {step.confidence !== null ? (
            <span className="text-muted">{step.confidence.toFixed(0)}%</span>
          ) : null}
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-2 text-xs">
        {step.trades.map((t) => (
          <div key={t.agent_id} className="flex items-center justify-between border-b border-border/50 py-1">
            <span className="text-muted">{t.agent_type}</span>
            <span
              className={
                t.action.startsWith("BLOCKED")
                  ? "text-danger"
                  : t.action === "HOLD"
                    ? "text-muted"
                    : "text-text"
              }
            >
              {t.action} {t.size > 0 ? t.size.toFixed(1) : ""}
            </span>
          </div>
        ))}
      </div>

      {step.threat_reasons.length > 0 ? (
        <div className="mt-3 text-xs">
          <span className="label-text">Detected:</span>
          <ul className="list-disc pl-5 text-warning">
            {step.threat_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-2 text-xs text-muted">
        Reward: <span className="text-text tabular-nums">{step.reward.toFixed(2)}</span>
      </div>
    </div>
  );
}
