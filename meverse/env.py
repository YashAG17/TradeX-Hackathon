"""Shared environment-variable loading helpers."""

from __future__ import annotations

from pathlib import Path


def load_repo_env() -> None:
    """Load repo-root .env if python-dotenv is available.

    This keeps local CLI behavior ergonomic without making .env mandatory.
    If python-dotenv is unavailable or .env is absent, the function is a no-op.
    """

    try:
        from dotenv import load_dotenv
    except Exception:
        return

    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
