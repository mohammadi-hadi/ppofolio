"""Train one ensemble member. Can be called repeatedly with different seeds /
algorithms to build up models/policy_sN.zip (or policy_lstm.zip)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ppofolio.env.crypto_trading_env import CryptoTradingEnv
from ppofolio.utils import ROOT, ensure_dir, load_config


def make_env_fn(panel: pd.DataFrame, cfg: dict, seed: int):
    def _fn():
        env = Monitor(CryptoTradingEnv(panel, cfg))
        env.reset(seed=seed)
        return env
    return _fn


def make_val_env(panel: pd.DataFrame, cfg: dict):
    """Deterministic full-length rollout over the val slice for EvalCallback."""
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    env = CryptoTradingEnv(panel, cfg, episode_start_idx=lookback, enforce_horizon=False)
    return Monitor(env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--algo", choices=["ppo", "recurrent_ppo"], default="ppo")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)
    tr = cfg["splits"]["train"]
    train_panel = panel.loc[pd.Timestamp(str(tr["start"]), tz="UTC"): pd.Timestamp(str(tr["end"]), tz="UTC")]
    print(f"[train_one] algo={args.algo} seed={args.seed} panel={train_panel.shape}")

    n_envs = cfg["train"]["n_envs"]
    vec = DummyVecEnv([make_env_fn(train_panel, cfg, args.seed + i) for i in range(n_envs)])
    tb_dir = ensure_dir(Path(ROOT) / "runs")
    timesteps = args.timesteps or cfg["train"]["total_timesteps"]

    # Periodic deterministic evaluation on the val split; saves the best snapshot.
    callbacks = []
    va = cfg["splits"].get("val")
    out_path = Path(ROOT) / args.out
    if va:
        lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
        v_start = pd.Timestamp(str(va["start"]), tz="UTC")
        v_end = pd.Timestamp(str(va["end"]), tz="UTC")
        buffer_start = panel.index[max(panel.index.searchsorted(v_start) - lookback, 0)]
        val_panel = panel.loc[buffer_start:v_end]
        best_dir = ensure_dir(Path(ROOT) / "models" / "best" / out_path.stem)
        callbacks.append(EvalCallback(
            DummyVecEnv([lambda: make_val_env(val_panel, cfg)]),
            best_model_save_path=str(best_dir),
            log_path=str(best_dir),
            eval_freq=max(50_000 // int(n_envs), 1),
            n_eval_episodes=1,
            deterministic=True,
            verbose=1,
        ))

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=cfg["train"]["learning_rate"],
            gamma=cfg["train"]["gamma"],
            gae_lambda=cfg["train"]["gae_lambda"],
            ent_coef=cfg["train"]["ent_coef"],
            policy_kwargs=cfg["train"]["policy_kwargs"],
            tensorboard_log=str(tb_dir),
            seed=args.seed, verbose=1,
        )
    else:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy", vec,
            learning_rate=cfg["train"]["learning_rate"],
            gamma=cfg["train"]["gamma"],
            gae_lambda=cfg["train"]["gae_lambda"],
            ent_coef=cfg["train"]["ent_coef"],
            policy_kwargs={"net_arch": [128], "lstm_hidden_size": 128},
            tensorboard_log=str(tb_dir),
            seed=args.seed, verbose=1,
        )
    model.learn(total_timesteps=timesteps, callback=callbacks or None)

    ensure_dir(out_path.parent)
    model.save(out_path)
    print(f"[train_one] saved {out_path}")


if __name__ == "__main__":
    main()
