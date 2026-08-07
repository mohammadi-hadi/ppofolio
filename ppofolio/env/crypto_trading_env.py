"""Gymnasium environment for multi-asset crypto portfolio trading.

State:  flattened lookback window of per-asset features + current weights + cash fraction.
Action: target weights, bounded by config (long-only or long/short), leverage ≤ max_leverage.
Step:   rebalance at today's close with fees + gaussian slippage, advance 1 day, compute
        multi-objective reward based on portfolio equity change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ppofolio.rewards.multi_reward import RewardState, compute


def _feature_cols_for(asset: str, tech: list[str], onchain: list[str], sent: list[str]) -> list[str]:
    return [f"{asset}__{c}" for c in tech + onchain + sent]


class CryptoTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel: pd.DataFrame, cfg: dict, episode_start_idx: int | None = None,
                 enforce_horizon: bool = True):
        super().__init__()
        self.cfg = cfg
        # Copy: we may add MACRO fallback columns below and must not mutate a
        # caller-owned slice shared across vectorized envs.
        panel = panel.copy()
        self.panel = panel
        self.enforce_horizon = enforce_horizon
        self.assets = [a["symbol"].split("/")[0] for a in cfg["assets"]]
        self.n = len(self.assets)
        # Lookback is in bars (not days). Support legacy "lookback_days" key.
        self.lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
        self.bars_per_day = int(cfg.get("bars_per_day", 1))
        self.bars_per_year = int(cfg.get("bars_per_year", 365))
        self.borrow_rate_per_bar = float(cfg.get("borrow_rate_daily", 0.0)) / max(self.bars_per_day, 1)
        self.fee = cfg["fee_bps"] / 10_000.0
        self.slip_std = np.array([a["slippage_std"] for a in cfg["assets"]], dtype=np.float32)
        self.allow_short = cfg["action"]["allow_short"]
        self.max_leverage = cfg["action"]["max_leverage"]
        self.rebalance_thr = cfg["action"]["rebalance_threshold"]
        # Optional per-asset weight cap so training sees the same concentration
        # limit the live risk layer enforces (invest_limits.max_single_position).
        _cap = cfg["action"].get("max_single_position")
        self.max_single = float(_cap) if _cap is not None else None
        self.initial_cash = cfg["initial_cash"]

        self.feat_cols: list[list[str]] = [
            _feature_cols_for(a, cfg["features"]["technical"], cfg["features"]["onchain"], cfg["features"]["sentiment"])
            for a in self.assets
        ]
        self.close_cols = [f"{a}__close" for a in self.assets]
        self.vol_cols = [f"{a}__volume" for a in self.assets]
        self.sent_cols = [f"{a}__sent_mean" for a in self.assets]
        self.n_feat = len(self.feat_cols[0])
        self.macro_feats = cfg["features"].get("macro", [])
        self.n_macro = len(self.macro_feats)
        self.macro_cols = [f"MACRO__{c}" for c in self.macro_feats]

        # Precompute numpy arrays — avoids pandas iloc in the hot step() loop.
        flat_feat_cols = [c for cols in self.feat_cols for c in cols]
        self._feat_arr = panel[flat_feat_cols].to_numpy(dtype=np.float32)
        self._price_arr = panel[self.close_cols].to_numpy(dtype=np.float64)
        self._vol_arr = panel[self.vol_cols].to_numpy(dtype=np.float64)
        # Fall back to zero sentiment if the panel was built without those columns
        # (features.sentiment: []) — the reward's sentiment term still reads them.
        if all(c in panel.columns for c in self.sent_cols):
            self._sent_arr = panel[self.sent_cols].to_numpy(dtype=np.float64)
        else:
            self._sent_arr = np.zeros((len(panel), self.n), dtype=np.float64)
        if self.n_macro > 0:
            # Fall back to zeros if macro columns are missing (old datasets).
            for c in self.macro_cols:
                if c not in panel.columns:
                    panel[c] = 0.0
            self._macro_arr = panel[self.macro_cols].to_numpy(dtype=np.float32)
        else:
            self._macro_arr = np.zeros((len(panel), 0), dtype=np.float32)
        self._T = len(panel)

        obs_dim = self.lookback * (self.n * self.n_feat + self.n_macro) + self.n + 1
        # Features are z-scored or pct-like and sentiment is in [-1,1]; cap at ±50 to be safe.
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(obs_dim,), dtype=np.float32)
        low = -1.0 if self.allow_short else 0.0
        self.action_space = spaces.Box(low=low, high=1.0, shape=(self.n,), dtype=np.float32)

        self._episode_start_idx = episode_start_idx
        self._t = 0
        self._rng = np.random.default_rng()

        # Bear-augmented sampling: build a list of episode starts that sit inside
        # historical drawdown regimes. Detection window = 20 * bars_per_day so it
        # still means "~20 calendar days of drawdown > 10%" regardless of timeframe.
        self._bear_sample_prob = float(cfg.get("train", {}).get("bear_sample_prob", 0.0))
        self._episode_horizon = int(cfg.get("train", {}).get("episode_horizon_bars",
                                                             60 * max(self.bars_per_day, 1)))
        btc_close = panel[self.close_cols[0]].to_numpy(dtype=float)
        self._bear_starts: np.ndarray = np.array([], dtype=np.int64)
        bear_window = 20 * max(self.bars_per_day, 1)
        if self._bear_sample_prob > 0 and len(btc_close) > bear_window + self._episode_horizon:
            ret_win = np.full_like(btc_close, np.nan)
            ret_win[bear_window:] = btc_close[bear_window:] / btc_close[:-bear_window] - 1.0
            bear_mask = np.where(ret_win < -0.10)[0]
            lo = self.lookback
            hi = len(panel) - self._episode_horizon
            self._bear_starts = bear_mask[(bear_mask >= lo) & (bear_mask < hi)]

        self._reset_state()

    # ---- state helpers ----
    def _reset_state(self):
        self.weights = np.zeros(self.n, dtype=np.float64)
        self.cash = float(self.initial_cash)
        self.holdings = np.zeros(self.n, dtype=np.float64)
        self.equity = float(self.initial_cash)
        sharpe_window = int(self.cfg.get("reward", {}).get("sharpe_window", 60))
        self.reward_state = RewardState.create(peak_equity=self.equity, sharpe_window=sharpe_window)

    def _prices(self, t: int) -> np.ndarray:
        return self._price_arr[t]

    def _volumes(self, t: int) -> np.ndarray:
        return self._vol_arr[t]

    def _sentiment(self, t: int) -> np.ndarray:
        return self._sent_arr[t]

    def _observation(self) -> np.ndarray:
        start = self._t - self.lookback + 1
        asset_feats = self._feat_arr[start : self._t + 1]  # (lookback, n*n_feat)
        if self.n_macro > 0:
            macro_feats = self._macro_arr[start : self._t + 1]  # (lookback, n_macro)
            feats = np.concatenate([asset_feats, macro_feats], axis=1).ravel()
        else:
            feats = asset_feats.ravel()
        cash_frac = np.float32(self.cash / max(self.equity, 1e-9))
        obs = np.empty(feats.size + self.n + 1, dtype=np.float32)
        obs[:feats.size] = feats
        obs[feats.size:feats.size + self.n] = self.weights.astype(np.float32)
        obs[-1] = cash_frac
        return np.clip(obs, -50.0, 50.0)

    # ---- gym API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        T = self._T
        ep = self._episode_horizon
        if self._episode_start_idx is not None:
            self._t = self._episode_start_idx
        elif self._bear_sample_prob > 0 and len(self._bear_starts) > 0 and self._rng.random() < self._bear_sample_prob:
            self._t = int(self._rng.choice(self._bear_starts))
        else:
            self._t = int(self._rng.integers(self.lookback, max(T - ep, self.lookback + 1)))
        self._ep_start = self._t
        self._reset_state()
        # initialize holdings to zero, all in cash
        return self._observation(), {}

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        if not self.allow_short:
            a = np.clip(a, 0.0, 1.0)
        else:
            a = np.clip(a, -1.0, 1.0)
        if self.max_single is not None:
            a = np.where(a >= 0, np.minimum(a, self.max_single), np.maximum(a, -self.max_single))
        gross = np.abs(a).sum()
        if gross > self.max_leverage and gross > 0:
            a = a * (self.max_leverage / gross)
        return a

    def _rebalance(self, target_w: np.ndarray, price: np.ndarray) -> np.ndarray:
        """Trade from current (holdings, cash) to target weights at given prices.

        Returns traded notional per asset (|delta| * execution_price).
        """
        # skip trivially small changes
        delta_w = target_w - self.weights
        trade_mask = np.abs(delta_w) >= self.rebalance_thr
        if not trade_mask.any():
            # still mark positions to market below
            self.weights = (self.holdings * price) / max(self.equity, 1e-9)
            return np.zeros(self.n)

        exec_slip = self._rng.normal(0.0, self.slip_std).astype(np.float64)
        exec_price = price * (1.0 + exec_slip)  # adverse slippage is randomly signed

        # target notional per asset
        target_notional = target_w * self.equity
        cur_notional = self.holdings * price
        delta_notional = np.where(trade_mask, target_notional - cur_notional, 0.0)

        # fee on absolute traded notional
        fees = np.abs(delta_notional) * self.fee

        # update cash: buying uses cash, selling returns cash; both incur fees
        self.cash -= float(delta_notional.sum()) + float(fees.sum())
        # update holdings in units at exec_price; full exits snap to exactly zero
        # units (units at slipped price vs notional at mid leaves ±dust otherwise)
        new_units = self.holdings + np.where(exec_price > 0, delta_notional / exec_price, 0.0)
        full_exit = trade_mask & (target_w == 0.0)
        self.holdings = np.where(full_exit, 0.0, new_units)
        # mark weights to market at mid-price (close)
        self.weights = (self.holdings * price) / max(self.equity, 1e-9)
        return np.abs(delta_notional)

    def step(self, action: np.ndarray):
        target_w = self._clip_action(action)
        price_today = self._prices(self._t)
        vol_today = self._volumes(self._t)

        prev_equity = self.equity
        prev_weights = self.weights.copy()
        traded_notional = self._rebalance(target_w, price_today)
        # Sentiment known at decision time (bar t), not the next bar's value.
        sentiment = self._sentiment(self._t)

        # advance one bar, compute new equity with next-bar close
        self._t += 1
        done = self._t >= self._T - 1
        if self.enforce_horizon and (self._t - self._ep_start) >= self._episode_horizon:
            done = True
        price_next = self._prices(self._t)
        self.equity = float(self.cash + (self.holdings * price_next).sum())
        self.weights = (self.holdings * price_next) / max(self.equity, 1e-9)

        # Liquidity threshold is defined against DAILY volume regardless of timeframe.
        daily_volume_usd = vol_today * price_today * max(self.bars_per_day, 1)

        reward_cfg = dict(self.cfg["reward"])
        reward_cfg["bars_per_year"] = self.bars_per_year
        reward_cfg["bars_per_day"] = self.bars_per_day
        reward_cfg["borrow_rate_per_bar"] = self.borrow_rate_per_bar
        reward, components = compute(
            prev_equity=prev_equity,
            new_equity=self.equity,
            weights=self.weights,
            prev_weights=prev_weights,
            traded_notional=traded_notional,
            daily_volume_usd=daily_volume_usd,
            sentiment=sentiment,
            state=self.reward_state,
            cfg=reward_cfg,
        )

        bankrupt = self.equity < 0.1 * self.initial_cash
        terminated = bankrupt
        truncated = done

        info = {"equity": self.equity, **components, "weights": self.weights.copy(),
                "traded_notional": float(traded_notional.sum()),
                "fees_paid": float(traded_notional.sum() * self.fee)}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info
