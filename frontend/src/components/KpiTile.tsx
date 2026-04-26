interface KpiTileProps {
  label: string;
  value: string | number;
  hint?: string;
  tone?: "default" | "success" | "warning" | "danger" | "info";
  delay?: number;
}

const toneClasses: Record<NonNullable<KpiTileProps["tone"]>, string> = {
  default: "text-text",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
  info: "text-info",
};

export function KpiTile({ label, value, hint, tone = "default", delay = 0 }: KpiTileProps) {
  return (
    <div
      className="card animate-fade-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="label-text mb-2">{label}</div>
      <div className={`text-2xl font-semibold ${toneClasses[tone]}`}>{value}</div>
      {hint ? <div className="text-muted text-xs mt-1">{hint}</div> : null}
    </div>
  );
}
