import { Outlet } from "react-router-dom";
import { Tabs } from "../../components/Tabs";

const TABS = [
  { to: "/tradex/live-replay", label: "Live Market Replay" },
  { to: "/tradex/reward-curves", label: "Reward Optimization Curves" },
  { to: "/tradex/baseline-vs-learned", label: "Baseline vs Learned" },
];

export function TradexLayout() {
  return (
    <div className="space-y-6 animate-fade-up">
      <div>
        <h1 className="text-2xl font-bold text-accent tracking-tight">TradeX</h1>
        <p className="text-muted text-sm">
          Multi-agent AMM governance with a PPO-trained overseer. Replay
          episodes, inspect training curves, benchmark policies.
        </p>
      </div>
      <Tabs items={TABS} />
      <Outlet />
    </div>
  );
}
