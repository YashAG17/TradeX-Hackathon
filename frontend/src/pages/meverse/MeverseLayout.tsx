import { Outlet } from "react-router-dom";
import { Tabs } from "../../components/Tabs";

const TABS = [
  { to: "/meverse/episode-runner", label: "Episode Runner" },
  { to: "/meverse/policy-comparison", label: "Policy Comparison" },
  { to: "/meverse/telemetry", label: "Telemetry Viewer" },
  { to: "/meverse/about", label: "About" },
];

export function MeverseLayout() {
  return (
    <div className="space-y-6 animate-fade-up">
      <div>
        <h1 className="text-2xl font-bold text-accent tracking-tight">MEVerse</h1>
        <p className="text-muted text-sm">
          OpenEnv-compliant market surveillance benchmark. Run an episode,
          compare baseline policies, replay telemetry.
        </p>
      </div>
      <Tabs items={TABS} />
      <Outlet />
    </div>
  );
}
