# TradeX + MEVerse — React Frontend

A React + Vite SPA that ports the two existing Gradio dashboards
(`dashboard.py` for MEVerse surveillance, `app.py` for TradeX governance) into
a single browser app. The Gradio apps still run unchanged; this is an
alternative UI on top of the same simulation logic.

## Stack

- React 18 + TypeScript + Vite
- Tailwind CSS (dark theme mirrors the palette in `dashboard.py`)
- TanStack Query for API calls
- Recharts for simple bar / line / composed charts
- react-plotly.js for the figures the backend ships pre-built (heatmap, AMM
  subplots, grade radar, gauges, confusion matrix)
- Routing via `react-router-dom`

## Architecture

```
React SPA  ──/api──▶  FastAPI backend (../backend)  ──▶  meverse/, tradex/ packages
   :5173                       :8000                       (existing Python code)
```

Vite proxies both `/api` and `/static` to the FastAPI port. The backend
re-uses the original Python simulation modules — no logic is duplicated.

## Running locally

In two terminals from the repo root (`TradeX-Hackathon/`):

```bash
# Terminal 1 — backend
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

```bash
# Terminal 2 — frontend
cd frontend
cp .env.example .env        # optional, override VITE_API_PROXY_TARGET
npm install
npm run dev                 # http://localhost:5173
```

Open http://localhost:5173 — Home links to the MEVerse and TradeX dashboards.

## Build

```bash
npm run build       # type-checks then emits dist/
npm run preview     # serve dist/ on :4173 (still proxies /api at runtime)
```

`dist/` is a static bundle. Production deployment can either:

1. Serve `dist/` from any static host and run the FastAPI backend separately
   (configure CORS for the static origin), or
2. Mount `dist/` from FastAPI itself by adding
   `app.mount("/", StaticFiles(directory="frontend/dist", html=True))` after
   the API routers in `backend/app.py`.

## Layout

```
src/
  main.tsx, App.tsx               # entry + router
  api/
    client.ts                     # fetch wrapper, error type
    meverse.ts, tradex.ts         # per-domain API calls
    types.ts                      # response types matching backend/schemas.py
  components/
    Layout.tsx, Tabs.tsx          # shell + tab strip
    ChartCard.tsx, KpiTile.tsx    # presentational primitives
    PlotlyFigure.tsx              # react-plotly.js wrapper
    ScoreBar.tsx, StepTable.tsx
    charts/
      RewardTimeline.tsx          # Recharts ComposedChart (bar + cumulative line)
      ActionDistribution.tsx      # Recharts stacked BarChart
      PolicyComparison.tsx        # Recharts grouped BarChart
      TradexBenchmark.tsx         # Recharts grouped BarChart
  pages/
    Home.tsx
    meverse/                      # 4 pages, mirrors dashboard.py tabs
    tradex/                       # 3 pages, mirrors app.py tabs
  theme/
    colors.ts                     # mirrors dashboard.py:38 palette
    index.css                     # Tailwind layers + scrollbar polish
```

## Notes

- `RewardCurves` references PNGs at `/static/plots/*.png` produced by the
  TradeX training pipeline (`tradex/plot_trl.py` etc.). If the `plots/`
  directory is empty the page renders a graceful placeholder per file.
- The Plotly figures from the backend already include their `frames` arrays
  (start→end transitions). They are loaded but not auto-played; flip on by
  passing `config={{ displayModeBar: false, animate: true }}` in
  `PlotlyFigure.tsx` if desired.
- Type definitions in `src/api/types.ts` are kept in sync with
  `backend/schemas.py` manually. If a schema changes, mirror it there.
