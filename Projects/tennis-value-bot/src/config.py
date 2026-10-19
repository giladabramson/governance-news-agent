"""Config loading: config.yaml + .env. Live mode is refused at load time."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ledger.db"
ALIAS_PATH = PROJECT_ROOT / "src" / "player_aliases.csv"


def load_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("mode") != "paper":
        raise SystemExit(
            "Only paper mode is implemented. Live trading requires (1) paper "
            "validation per spec §9, (2) i_confirm_legal_access, and (3) a live "
            "executor that this build intentionally does not contain.")
    if os.environ.get("POLYMARKET_PRIVATE_KEY"):
        raise SystemExit("POLYMARKET_PRIVATE_KEY is set but this is a paper-only "
                         "build — remove it from the environment.")
    cfg["odds_api_key"] = os.environ.get("ODDS_API_KEY", "")
    DATA_DIR.mkdir(exist_ok=True)
    return cfg


def arm_configs(cfg: dict) -> dict[str, dict]:
    """Expand the `arms` section into full per-arm configs.

    Each arm inherits the top-level config with its overrides applied. A
    config without an `arms` section behaves as a single 'base' arm, so
    pre-arms ledgers and configs keep working unchanged.
    """
    arms = cfg.get("arms") or {"base": {}}
    return {name: {**cfg, **(overrides or {})} for name, overrides in arms.items()}


def halted() -> bool:
    return os.environ.get("BOT_HALT") == "1"
