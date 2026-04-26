import { useState } from "react";

import { ChartCard } from "../../components/ChartCard";

interface PlotEntry {
  file: string;
  label: string;
}

const PLOTS: PlotEntry[] = [
  { file: "reward_vs_episode.png", label: "Reward Moving Average" },
  { file: "precision_recall.png", label: "Precision / Recall Boundary" },
  { file: "detection_capability.png", label: "Correct Blocks vs False Positives" },
  { file: "final_price_error_vs_episode.png", label: "Action Distribution Target" },
];

export function RewardCurves() {
  return (
    <div className="space-y-6">
      <p className="text-muted text-sm">
        Static training curves rendered by the TradeX training pipeline. Generate them by running{" "}
        <code className="text-text">python -m tradex.train</code> or{" "}
        <code className="text-text">python -m tradex.plot_trl</code>; outputs land in{" "}
        <code className="text-text">plots/</code>.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        {PLOTS.map((p, i) => (
          <ChartCard key={p.file} title={p.label} delay={i * 80}>
            <PlotImage file={p.file} alt={p.label} />
          </ChartCard>
        ))}
      </div>
    </div>
  );
}

function PlotImage({ file, alt }: { file: string; alt: string }) {
  const [errored, setErrored] = useState(false);
  if (errored) {
    return (
      <div className="text-muted text-sm border border-dashed border-border rounded-lg p-6 text-center">
        Plot <code className="text-text">{file}</code> not found. Run training to generate it.
      </div>
    );
  }
  return (
    <img
      src={`/static/plots/${file}`}
      alt={alt}
      onError={() => setErrored(true)}
      className="w-full rounded-lg border border-border"
    />
  );
}
