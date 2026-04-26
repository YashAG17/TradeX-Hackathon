"""Unified TradeX runner: train policy + compare variants + combined final output."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tradex.compare_all import compare_all
from tradex.eval_trl import DEFAULT_TRL_PATH, DEFAULT_UNSLOTH_PATH
from tradex.train import train as train_policy


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def _run_train(episodes: int, onsite: bool, verbose: bool) -> dict[str, Any]:
    started = time.time()
    args = SimpleNamespace(episodes=episodes, onsite=onsite, verbose=verbose, allow_streak=0)

    print(f"[TRAIN] start episodes={episodes} onsite={onsite} verbose={verbose}", flush=True)
    train_policy(args)

    model_path = Path("models") / "best_model.pth"
    result = {
        "started_at_utc": _now_utc_iso(),
        "episodes": episodes,
        "onsite": onsite,
        "verbose": verbose,
        "best_model_path": str(model_path),
        "best_model_exists": model_path.exists(),
        "duration_seconds": round(time.time() - started, 2),
    }
    print(
        f"[TRAIN] end model_exists={result['best_model_exists']} duration_s={result['duration_seconds']}",
        flush=True,
    )
    return result


def _run_compare(episodes: int, trl_model_path: str, unsloth_model_path: str, output_csv: str) -> dict[str, Any]:
    started = time.time()
    print(
        (
            "[COMPARE] start "
            f"episodes={episodes} trl_model_path={trl_model_path} unsloth_model_path={unsloth_model_path}"
        ),
        flush=True,
    )

    df, rows = compare_all(episodes, trl_model_path, unsloth_model_path)
    _ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)

    best_by_reward = max(rows, key=lambda row: row.get("avg_reward", float("-inf"))) if rows else {}
    best_by_f1 = max(rows, key=lambda row: row.get("f1", float("-inf"))) if rows else {}

    result = {
        "started_at_utc": _now_utc_iso(),
        "episodes": episodes,
        "output_csv": output_csv,
        "rows": rows,
        "best_by_avg_reward": best_by_reward.get("policy"),
        "best_by_f1": best_by_f1.get("policy"),
        "duration_seconds": round(time.time() - started, 2),
    }
    print(
        (
            "[COMPARE] end "
            f"rows={len(rows)} best_reward={result['best_by_avg_reward']} best_f1={result['best_by_f1']} "
            f"duration_s={result['duration_seconds']}"
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TradeX training + comparison and emit one combined final output."
    )
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--compare-episodes", type=int, default=100)
    parser.add_argument("--onsite", action="store_true", help="Use CUDA if available during training.")
    parser.add_argument("--verbose", action="store_true", help="Verbose training logs.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run compare only.")
    parser.add_argument("--trl-model-path", type=str, default=DEFAULT_TRL_PATH)
    parser.add_argument("--unsloth-model-path", type=str, default=DEFAULT_UNSLOTH_PATH)
    parser.add_argument("--output-csv", type=str, default="outputs/final_benchmark.csv")
    parser.add_argument("--output-json", type=str, default="outputs/final_combined_output.json")
    args = parser.parse_args()

    combined: dict[str, Any] = {
        "run_started_at_utc": _now_utc_iso(),
        "config": vars(args),
        "train": None,
        "compare": None,
    }

    if args.skip_train:
        print("[TRAIN] skipped via --skip-train", flush=True)
    else:
        combined["train"] = _run_train(
            episodes=args.train_episodes,
            onsite=args.onsite,
            verbose=args.verbose,
        )

    combined["compare"] = _run_compare(
        episodes=args.compare_episodes,
        trl_model_path=args.trl_model_path,
        unsloth_model_path=args.unsloth_model_path,
        output_csv=args.output_csv,
    )
    combined["run_finished_at_utc"] = _now_utc_iso()

    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\nFINAL COMBINED OUTPUT", flush=True)
    print(json.dumps(combined, indent=2), flush=True)
    print(f"\nSaved combined JSON: {args.output_json}", flush=True)
    print(f"Saved compare CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
