import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { tradexApi, type TradexComparePayload } from "../../api/tradex";
import { ApiError } from "../../api/client";
import { ChartCard } from "../../components/ChartCard";
import { TradexBenchmark } from "../../components/charts/TradexBenchmark";

export function BaselineVsLearned() {
  const [numEpisodes, setNumEpisodes] = useState<number>(50);

  const compareMutation = useMutation({
    mutationFn: (payload: TradexComparePayload) => tradexApi.compare(payload),
  });

  function handleRun(e: React.FormEvent) {
    e.preventDefault();
    compareMutation.mutate({ num_episodes: numEpisodes });
  }

  const data = compareMutation.data;
  const error = compareMutation.error;

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="font-semibold mb-1">PPO vs Baseline Benchmark</h3>
        <p className="text-muted text-sm mb-4">
          Run a direct benchmark comparing the heuristic baseline, deterministic
          PPO, stochastic PPO, and a rule-hybrid policy. This actually executes
          {" "}<strong className="text-text">{numEpisodes} × 4</strong> episodes server-side, so
          it can take a while.
        </p>
        <form onSubmit={handleRun} className="grid sm:grid-cols-[1fr_auto] gap-3 items-end">
          <div>
            <label htmlFor="episodes" className="label-text block mb-1">
              Evaluation Episodes: <span className="text-text">{numEpisodes}</span>
            </label>
            <input
              id="episodes"
              type="range"
              min={10}
              max={100}
              step={10}
              value={numEpisodes}
              onChange={(e) => setNumEpisodes(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <button type="submit" className="btn-primary" disabled={compareMutation.isPending}>
            {compareMutation.isPending ? "Running..." : "Run Benchmark"}
          </button>
        </form>
        {error ? (
          <p className="text-danger text-xs mt-2">
            {(error as ApiError).detail || (error as Error).message}
          </p>
        ) : null}
      </div>

      {data ? (
        <>
          <ChartCard title="Baseline vs Learned Policy">
            <TradexBenchmark rows={data.rows} />
          </ChartCard>

          <ChartCard title="Benchmark Metrics">
            <div className="overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-bg">
                  <tr>
                    {["Policy", "Avg Reward", "Price Error", "Precision", "Recall", "F1", "Intervention Rate"].map((h) => (
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
                  {data.rows.map((row) => (
                    <tr key={row.policy} className="hover:bg-accent/5">
                      <td className="px-3 py-2 border-b border-border font-medium">{row.policy}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.avg_reward.toFixed(2)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.price_error.toFixed(2)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.precision.toFixed(2)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.recall.toFixed(2)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.f1.toFixed(2)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{row.intervention_rate.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>

          <ChartCard title="Summary">
            <pre className="text-sm text-text whitespace-pre-wrap font-mono">{data.summary}</pre>
          </ChartCard>
        </>
      ) : null}
    </div>
  );
}
