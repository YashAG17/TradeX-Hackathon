"""FastAPI dashboard backend.

Run:
    cd TradeX-Hackathon
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import meverse, tradex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"

app = FastAPI(
    title="TradeX + MEVerse Dashboard API",
    description="REST wrapper around the Gradio dashboard logic for the React frontend.",
    version="0.1.0",
)

# Vite dev server lives on :5173; allow any localhost port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meverse.router, prefix="/api/meverse", tags=["meverse"])
app.include_router(tradex.router, prefix="/api/tradex", tags=["tradex"])

# Serve any pre-rendered TradeX training plots (created by tradex/plot_trl.py etc.).
# The directory is optional — if it does not exist the route is simply skipped.
if PLOTS_DIR.is_dir():
    app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
else:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "plots_dir_exists": PLOTS_DIR.is_dir()}
