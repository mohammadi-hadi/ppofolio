from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg: dict, path: str | Path) -> None:
    """Fail fast on the config mistakes that used to fail silently."""
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config did not parse to a mapping")
    if "model_prefix" not in cfg:
        raise ValueError(f"{path}: missing 'model_prefix' (e.g. policy_s / policy_h) — "
                         "without it the daily ensemble is silently loaded under any config")
    if cfg.get("timeframe", "1d") != "1d":
        for k in ("bars_per_day", "bars_per_year"):
            if k not in cfg:
                raise ValueError(f"{path}: timeframe={cfg['timeframe']} requires '{k}'")
    sp = cfg.get("splits") or {}

    def _end(d):
        return str(d.get("end", ""))

    def _start(d):
        return str(d.get("start", ""))

    if "train" in sp and "val" in sp and _end(sp["train"]) >= _start(sp["val"]):
        raise ValueError(f"{path}: splits.train.end must precede splits.val.start")
    if "val" in sp and "test_regimes" in sp:
        for name, (s, _e) in sp["test_regimes"].items():
            if str(s) <= _end(sp["val"]):
                raise ValueError(f"{path}: test regime '{name}' starts inside/before the val split — "
                                 "test windows must be held out from model selection")


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
