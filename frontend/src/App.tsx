import { Navigate, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { MeverseLayout } from "./pages/meverse/MeverseLayout";
import { EpisodeRunner } from "./pages/meverse/EpisodeRunner";
import { PolicyComparison } from "./pages/meverse/PolicyComparison";
import { TelemetryViewer } from "./pages/meverse/TelemetryViewer";
import { About } from "./pages/meverse/About";
import { TradexLayout } from "./pages/tradex/TradexLayout";
import { LiveMarketReplay } from "./pages/tradex/LiveMarketReplay";
import { RewardCurves } from "./pages/tradex/RewardCurves";
import { BaselineVsLearned } from "./pages/tradex/BaselineVsLearned";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />

        <Route path="meverse" element={<MeverseLayout />}>
          <Route index element={<Navigate to="episode-runner" replace />} />
          <Route path="episode-runner" element={<EpisodeRunner />} />
          <Route path="policy-comparison" element={<PolicyComparison />} />
          <Route path="telemetry" element={<TelemetryViewer />} />
          <Route path="about" element={<About />} />
        </Route>

        <Route path="tradex" element={<TradexLayout />}>
          <Route index element={<Navigate to="live-replay" replace />} />
          <Route path="live-replay" element={<LiveMarketReplay />} />
          <Route path="reward-curves" element={<RewardCurves />} />
          <Route path="baseline-vs-learned" element={<BaselineVsLearned />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
