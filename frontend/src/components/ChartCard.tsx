import type { ReactNode } from "react";

interface ChartCardProps {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  delay?: number;
}

export function ChartCard({
  title,
  subtitle,
  children,
  className = "",
  delay = 0,
}: ChartCardProps) {
  return (
    <div
      className={`card animate-fade-up ${className}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {title ? (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          {subtitle ? <p className="text-xs text-muted mt-1">{subtitle}</p> : null}
        </div>
      ) : null}
      <div className="min-h-[1px]">{children}</div>
    </div>
  );
}
