import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { meverseApi, type ComparePoliciesPayload } from "../../api/meverse";
import { ApiError } from "../../api/client";
import { ChartCard } from "../../components/ChartCard";
import { PolicyComparison as PolicyComparisonChart } from "../../components/charts/PolicyComparison";

export function PolicyComparison() {
  const tasksQuery = useQuery({
    queryKey: ["meverse-tasks"],
    queryFn: meverseApi.listTasks,
  });

  const [task, setTask] = useState<string>("full_market_surveillance");
  const [seed, setSeed] = useState<number>(42);

  useEffect(() => {
    if (tasksQuery.data?.tasks?.length && !tasksQuery.data.tasks.includes(task)) {
      setTask(tasksQuery.data.tasks[0]);
    }
  }, [tasksQuery.data, task]);

  const compareMutation = useMutation({
    mutationFn: (payload: ComparePoliciesPayload) => meverseApi.comparePolicies(payload),
  });

  function handleCompare(e: React.FormEvent) {
    e.preventDefault();
    compareMutation.mutate({ task, seed });
  }

  const data = compareMutation.data;
  const error = compareMutation.error;

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="font-semibold mb-1">Compare baselines</h3>
        <p className="text-muted text-sm mb-4">
          Run Heuristic, Always-Allow, and Random on the same seed for a fair comparison.
        </p>
        <form onSubmit={handleCompare} className="grid sm:grid-cols-[1fr_180px_auto] gap-3 items-end">
          <div>
            <label htmlFor="cmp-task" className="label-text block mb-1">Task</label>
            <select
              id="cmp-task"
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
            <label htmlFor="cmp-seed" className="label-text block mb-1">Seed</label>
            <input
              id="cmp-seed"
              type="number"
              min={1}
              max={999999}
              step={1}
              className="input-field"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
          <button type="submit" className="btn-primary" disabled={compareMutation.isPending}>
            {compareMutation.isPending ? "Comparing..." : "Compare"}
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
          <ChartCard title={`Policy Comparison \u2014 ${data.task}`} subtitle={`Seed = ${data.seed}`}>
            <PolicyComparisonChart results={data.results} />
          </ChartCard>

          <ChartCard title="Detail">
            <div className="overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-bg">
                  <tr>
                    {["Policy", "Score", "Detection", "FP", "FN", "Health", "Overblocking", "Total Reward"].map((h) => (
                      <th key={h} className="text-left text-xs font-semibold uppercase tracking-wide text-accent px-3 py-2 border-b border-border">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r) => (
                    <tr key={r.policy} className="hover:bg-accent/5">
                      <td className="px-3 py-2 border-b border-border font-medium">{r.policy}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.score.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.detection.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.fp.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.fn.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.health.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.overblock.toFixed(4)}</td>
                      <td className="px-3 py-2 border-b border-border tabular-nums">{r.total_reward.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </>
      ) : null}
    </div>
  );
}
