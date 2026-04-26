import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { meverseApi, type RunEpisodePayload } from "../../api/meverse";
import { ApiError } from "../../api/client";
import { ChartCard } from "../../components/ChartCard";
import { KpiTile } from "../../components/KpiTile";
import { PlotlyFigure } from "../../components/PlotlyFigure";
import { ScoreBar } from "../../components/ScoreBar";
import { StepTable } from "../../components/StepTable";
import { ActionDistribution } from "../../components/charts/ActionDistribution";
import { RewardTimeline } from "../../components/charts/RewardTimeline";

const POLICIES = ["Heuristic", "Always Allow", "Random"];

export function EpisodeRunner() {
  const tasksQuery = useQuery({
    queryKey: ["meverse-tasks"],
    queryFn: meverseApi.listTasks,
  });

  const [task, setTask] = useState<string>("full_market_surveillance");
  const [policy, setPolicy] = useState<string>("Heuristic");
  const [seed, setSeed] = useState<number>(42);

  useEffect(() => {
    if (tasksQuery.data?.tasks?.length && !tasksQuery.data.tasks.includes(task)) {
      setTask(tasksQuery.data.tasks[0]);
    }
  }, [tasksQuery.data, task]);

  const runMutation = useMutation({
    mutationFn: (payload: RunEpisodePayload) => meverseApi.runEpisode(payload),
  });

  const result = runMutation.data;
  const error = runMutation.error;

  function handleRun(e: React.FormEvent) {
    e.preventDefault();
    runMutation.mutate({ task, policy, seed: seed === 0 ? null : seed });
  }

  return (
    <div className="grid lg:grid-cols-[280px_1fr] gap-6">
      <aside className="space-y-4">
        <form onSubmit={handleRun} className="card space-y-4">
          <div>
            <h3 className="font-semibold mb-1">Configuration</h3>
            <p className="text-muted text-xs">
              Surveillance scenario. Burst = easy, Pattern = medium, Full = hard.
            </p>
          </div>

          <div>
            <label htmlFor="task" className="label-text block mb-1">Task</label>
            <select
              id="task"
              className="input-field"
              value={task}
              onChange={(e) => setTask(e.target.value)}
            >
              {(tasksQuery.data?.tasks ?? []).map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="policy" className="label-text block mb-1">Policy</label>
            <select
              id="policy"
              className="input-field"
              value={policy}
              onChange={(e) => setPolicy(e.target.value)}
            >
              {POLICIES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <p className="text-muted text-xs mt-1">
              Heuristic uses threshold rules. Always Allow never blocks. Random
              picks randomly.
            </p>
          </div>

          <div>
            <label htmlFor="seed" className="label-text block mb-1">Seed (0 = random)</label>
            <input
              id="seed"
              type="number"
              min={0}
              max={999999}
              step={1}
              className="input-field"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>

          <button type="submit" className="btn-primary w-full" disabled={runMutation.isPending}>
            {runMutation.isPending ? "Running..." : "Run Episode"}
          </button>

          {error ? (
            <p className="text-danger text-xs">
              {(error as ApiError).detail || (error as Error).message}
            </p>
          ) : null}
        </form>
      </aside>

      <section className="space-y-6">
        {!result && !runMutation.isPending ? (
          <div className="card text-muted text-sm">Run an episode to see results.</div>
        ) : null}

        {runMutation.isPending ? (
          <div className="card text-muted text-sm">Running...</div>
        ) : null}

        {result ? (
          <>
            <ChartCard title="Episode Summary" delay={0}>
              <div className="grid sm:grid-cols-3 gap-3 mb-4 text-sm">
                <SummaryStat label="Task" value={`${result.summary.title} (${result.summary.difficulty})`} />
                <SummaryStat label="Policy" value={result.summary.policy} />
                <SummaryStat label="Seed" value={result.summary.seed?.toString() ?? "random"} />
                <SummaryStat label="Steps" value={result.summary.steps} />
                <SummaryStat label="Total Reward" value={result.summary.total_reward.toFixed(3)} />
                <SummaryStat label="Avg Reward" value={result.summary.avg_reward.toFixed(3)} />
              </div>

              <div className="border-t border-border pt-4 space-y-3">
                <div className="flex items-baseline justify-between">
                  <span className="font-semibold">Final Score</span>
                  <span className="text-accent text-2xl tabular-nums font-semibold">
                    {result.summary.grade.score.toFixed(4)}
                  </span>
                </div>
                <ScoreBar score={result.summary.grade.score} />

                <div className="grid sm:grid-cols-5 gap-3 pt-2">
                  <ScoreBar label="Detection (50%)" score={result.summary.grade.detection_score} />
                  <ScoreBar label="False Positive (20%)" score={result.summary.grade.false_positive_score} />
                  <ScoreBar label="False Negative (15%)" score={result.summary.grade.false_negative_score} />
                  <ScoreBar label="Health (10%)" score={result.summary.grade.health_score} />
                  <ScoreBar label="Overblocking (5%)" score={result.summary.grade.overblocking_score} />
                </div>
              </div>
            </ChartCard>

            <ChartCard title="AMM Final State" delay={50}>
              <div className="animate-gauge-pop">
                <PlotlyFigure figure={result.plotly.amm_gauges} height={180} />
              </div>
            </ChartCard>

            <div className="grid lg:grid-cols-2 gap-6">
              <ChartCard title="Reward Timeline" subtitle="Bars colored by action; line = cumulative reward" delay={100}>
                <RewardTimeline data={result.reward_timeline} />
              </ChartCard>
              <ChartCard title="Action Distribution" subtitle="Stacked by ground-truth label" delay={150}>
                <ActionDistribution data={result.action_label_counts} />
              </ChartCard>
            </div>

            <ChartCard title="Signal Heatmap Over Time" delay={200}>
              <PlotlyFigure figure={result.plotly.signal_heatmap} height={300} />
            </ChartCard>

            <ChartCard title="AMM State Evolution" delay={250}>
              <PlotlyFigure figure={result.plotly.amm_chart} height={540} />
            </ChartCard>

            <div className="grid lg:grid-cols-2 gap-6">
              <ChartCard title="Grade Radar" delay={300}>
                <PlotlyFigure figure={result.plotly.grade_radar} height={360} />
              </ChartCard>
              <ChartCard title="Action vs Truth" delay={350}>
                <PlotlyFigure figure={result.plotly.confusion} height={260} />
              </ChartCard>
            </div>

            <ChartCard title="Step-by-Step Log" delay={400}>
              <StepTable rows={result.step_history} />
            </ChartCard>

            <div className="grid sm:grid-cols-3 gap-4">
              <KpiTile label="Steps" value={result.summary.steps} delay={0} />
              <KpiTile label="Total Reward" value={result.summary.total_reward.toFixed(3)} tone="success" delay={50} />
              <KpiTile label="Final Score" value={result.summary.grade.score.toFixed(4)} tone="info" delay={100} />
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="label-text">{label}</div>
      <div className="text-text font-medium">{value}</div>
    </div>
  );
}
