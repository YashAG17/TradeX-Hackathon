import { Link } from "react-router-dom";

export function Home() {
  return (
    <div className="space-y-8 animate-fade-up">
      <header className="card bg-gradient-to-br from-[#0a1628] via-[#0d2137] to-[#0a1628] border-[#1a3a5c]">
        <h1 className="text-3xl font-bold text-accent tracking-tight mb-2">
          TradeX + MEVerse Surveillance
        </h1>
        <p className="text-muted text-sm max-w-2xl">
          Bot-aware market surveillance in simulated AMM trading. Run episodes,
          compare policies, replay telemetry, and benchmark a PPO-trained
          overseer against heuristic baselines.
        </p>
      </header>

      <div className="grid md:grid-cols-2 gap-6">
        <Link to="/meverse" className="card block group">
          <div className="text-xs uppercase tracking-wide text-muted mb-2">
            OpenEnv benchmark
          </div>
          <div className="text-2xl font-semibold mb-2 group-hover:text-accent transition-colors">
            MEVerse
          </div>
          <p className="text-muted text-sm leading-relaxed">
            Run surveillance episodes (burst, pattern, full), inspect signal
            heatmaps and AMM dynamics, compare baseline policies, and replay
            recorded telemetry files.
          </p>
        </Link>

        <Link to="/tradex" className="card block group">
          <div className="text-xs uppercase tracking-wide text-muted mb-2">
            PPO overseer playground
          </div>
          <div className="text-2xl font-semibold mb-2 group-hover:text-accent transition-colors">
            TradeX
          </div>
          <p className="text-muted text-sm leading-relaxed">
            Live multi-agent AMM replays with deep-RL governance, training
            curves, and a baseline vs learned-policy benchmark.
          </p>
        </Link>
      </div>
    </div>
  );
}
