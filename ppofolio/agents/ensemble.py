"""Ensemble inference wrapper — averages target weights across K policies.

Handles both PPO (MlpPolicy) and RecurrentPPO (MlpLstmPolicy). For LSTM models
the hidden state is reset on each predict() call (single-shot daily inference),
which is correct for daily rebalancing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class EnsembleMember:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.is_lstm = self.path.name.startswith("policy_lstm")
        if self.is_lstm:
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(str(self.path))
        else:
            self.model = PPO.load(str(self.path))

    def predict(self, obs, deterministic: bool = True):
        if self.is_lstm:
            # RecurrentPPO.predict takes extra (state, episode_start) — use fresh state each call.
            action, _ = self.model.predict(
                obs, state=None, episode_start=np.array([True]), deterministic=deterministic,
            )
            return action
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


class EnsemblePolicy:
    def __init__(self, model_paths: list[str | Path], disagreement: dict | None = None):
        self.members = [EnsembleMember(p) for p in model_paths]
        # Optional uncertainty scaling: when members disagree on target weights,
        # shrink gross exposure toward cash. {enabled, k, floor} from config
        # (ensemble.disagreement_scaling). Off by default.
        self.disagreement = dict(disagreement or {})

    @property
    def models(self):  # kept for backward-compat with old backtest code
        return self.members

    def predict(self, obs, deterministic: bool = True):
        stacked = np.stack([m.predict(obs, deterministic=deterministic) for m in self.members], axis=0)
        mean = stacked.mean(axis=0)
        if self.disagreement.get("enabled") and len(self.members) > 1:
            k = float(self.disagreement.get("k", 2.0))
            floor = float(self.disagreement.get("floor", 0.25))
            spread = float(stacked.std(axis=0).mean())
            mean = mean * max(floor, 1.0 - k * spread)
        return mean, None


_CACHE: dict[tuple, EnsemblePolicy] = {}


def load_ensemble(models_dir: str | Path = "models", prefix: str = "policy_s",
                  disagreement: dict | None = None) -> EnsemblePolicy | None:
    """Load ensemble members whose filenames start with `prefix`.

    Config contract: each deployed service sets `model_prefix` in its config so
    projects that ship both daily (policy_s*) and hourly (policy_h*) models in
    the same Docker image load only their own ensemble.

    Cached on (dir, prefix, file mtimes) — the live loop calls this every run and
    reloading ~100 MB of policies from disk each time was pure waste. A changed
    or replaced model file invalidates the cache automatically.
    """
    import os
    d = Path(models_dir)
    paths = sorted(d.glob(f"{prefix}*.zip"))
    if os.environ.get("INCLUDE_LSTM", "false").lower() in ("1", "true", "yes"):
        paths += sorted(d.glob("policy_lstm*.zip"))
    if not paths:
        return None
    key = (str(d), prefix, tuple((p.name, p.stat().st_mtime_ns) for p in paths),
           tuple(sorted((disagreement or {}).items())))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = EnsemblePolicy(paths, disagreement=disagreement)
    return _CACHE[key]
