"""Smoke test: env passes SB3 check + a random rollout with synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppofolio.env.crypto_trading_env import CryptoTradingEnv


def _fake_panel(n_days: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    cols = {}
    for a in ["BTC", "ETH", "SOL"]:
        price = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n_days)))
        cols[f"{a}__close"] = price
        cols[f"{a}__volume"] = rng.lognormal(15, 0.3, n_days)
        for c in ["ret_1d", "log_vol_7d", "rsi_14", "macd", "bb_pct", "atr_14",
                  "active_addresses", "tx_count", "mean_fee_usd",
                  "sent_mean", "sent_count"]:
            cols[f"{a}__{c}"] = rng.normal(0, 1, n_days)
    return pd.DataFrame(cols, index=idx)


def _cfg():
    return {
        "assets": [
            {"symbol": "BTC/USDT", "coinmetrics": "btc", "slippage_std": 0.0005},
            {"symbol": "ETH/USDT", "coinmetrics": "eth", "slippage_std": 0.0005},
            {"symbol": "SOL/USDT", "coinmetrics": "sol", "slippage_std": 0.0012},
        ],
        "lookback_days": 30,
        "initial_cash": 1_000_000,
        "fee_bps": 10,
        "features": {
            "technical": ["ret_1d", "log_vol_7d", "rsi_14", "macd", "bb_pct", "atr_14"],
            "onchain": ["active_addresses", "tx_count", "mean_fee_usd"],
            "sentiment": ["sent_mean", "sent_count"],
        },
        "action": {"allow_short": False, "max_leverage": 1.0, "rebalance_threshold": 0.02},
        "reward": {
            "weights": {"return": 1.0, "sharpe": 0.3, "drawdown": 0.5, "liquidity": 0.1, "sentiment": 0.1},
            "sharpe_window": 30,
            "liquidity_threshold_pct": 0.01,
        },
    }


def test_random_rollout():
    env = CryptoTradingEnv(_fake_panel(), _cfg())
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    assert "equity" in info


def test_sb3_check():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    env = CryptoTradingEnv(_fake_panel(), _cfg())
    check_env(env, warn=True, skip_render_check=True)
