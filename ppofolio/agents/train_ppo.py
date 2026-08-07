"""Train PPO on CryptoTradingEnv using Stable-Baselines3."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ppofolio.env.crypto_trading_env import CryptoTradingEnv
from ppofolio.utils import ROOT, ensure_dir, load_config


def make_env_fn(panel: pd.DataFrame, cfg: dict, seed: int):
    def _fn():
        env = CryptoTradingEnv(panel, cfg)
        env.reset(seed=seed)
        return env
    return _fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="models/policy.zip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)

    tr = cfg["splits"]["train"]
    train_panel = panel.loc[pd.Timestamp(tr["start"], tz="UTC"): pd.Timestamp(tr["end"], tz="UTC")]
    print(f"[train] panel {train_panel.shape} {train_panel.index.min().date()}..{train_panel.index.max().date()}")

    seed = cfg["train"]["seed"]
    n_envs = cfg["train"]["n_envs"]
    vec = DummyVecEnv([make_env_fn(train_panel, cfg, seed + i) for i in range(n_envs)])

    tb_dir = ensure_dir(Path(ROOT) / "runs")
    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=cfg["train"]["learning_rate"],
        gamma=cfg["train"]["gamma"],
        gae_lambda=cfg["train"]["gae_lambda"],
        ent_coef=cfg["train"]["ent_coef"],
        policy_kwargs=cfg["train"]["policy_kwargs"],
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=cfg["train"]["total_timesteps"])

    out_path = Path(ROOT) / args.out
    ensure_dir(out_path.parent)
    model.save(out_path)
    print(f"[train] saved {out_path}")


if __name__ == "__main__":
    main()
