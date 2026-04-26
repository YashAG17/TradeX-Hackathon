interface ScoreBarProps {
  score: number; // 0..1
  label?: string;
}

export function ScoreBar({ score, label }: ScoreBarProps) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div>
      {label ? (
        <div className="flex items-baseline justify-between mb-1">
          <span className="label-text">{label}</span>
          <span className="text-text text-sm tabular-nums">{pct.toFixed(1)}%</span>
        </div>
      ) : null}
      <div className="h-2 rounded-full bg-border overflow-hidden">
        <div
          className="h-full bg-accent transition-[width] duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
