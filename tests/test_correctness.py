"""P1 correctness guards: turnover semantics, episode horizon, obs parity,
fee-adjusted baselines, kill-switch window."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppofolio.backtest.baselines import bh_rebalanced, buy_and_hold, equity_from_weights
from ppofolio.env.crypto_trading_env import CryptoTradingEnv
from ppofolio.live.kill_switch import (check_and_update, is_decommissioned, is_halted,
                                  maybe_send_halted_reminder, set_halt)
from ppofolio.live.loop import _observation_from_panel
from ppofolio.live.paper_broker import PaperBroker
from ppofolio.rewards.multi_reward import RewardState, compute

TECH = ["ret_1d", "log_vol_7d", "rsi_14", "macd", "bb_pct", "atr_14",
        "ret_60d", "ret_180d", "vol_20d", "ma200_dist"]
ONCHAIN = ["active_addresses", "tx_count", "mean_fee_usd"]
SENT = ["sent_mean", "sent_count"]
MACRO = ["fng", "btc_dominance", "funding_rate"]


def _cfg(**over):
    cfg = {
        "assets": [
            {"symbol": "BTC/USDT", "coinmetrics": "btc", "slippage_std": 0.0},
            {"symbol": "ETH/USDT", "coinmetrics": "eth", "slippage_std": 0.0},
            {"symbol": "SOL/USDT", "coinmetrics": "sol", "slippage_std": 0.0},
        ],
        "lookback_days": 30,
        "initial_cash": 1_000_000,
        "fee_bps": 10,
        "features": {"technical": TECH, "onchain": ONCHAIN, "sentiment": SENT, "macro": MACRO},
        "reward": {
            "weights": {"return": 1.0, "sharpe": 0.3, "drawdown": 0.5, "liquidity": 0.1,
                        "sentiment": 0.1, "loss_asymm": 0.5},
            "sharpe_window": 30,
            "liquidity_threshold_pct": 0.01,
        },
        "action": {"allow_short": False, "max_leverage": 1.0, "rebalance_threshold": 0.02},
        "train": {"episode_horizon_bars": 10},
    }
    cfg.update(over)
    return cfg


def _panel(n: int = 300, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    cols = {}
    for a in ["BTC", "ETH", "SOL"]:
        price = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
        cols[f"{a}__close"] = price
        cols[f"{a}__volume"] = rng.lognormal(15, 0.3, n)
        for c in TECH + ONCHAIN + SENT:
            if c not in ("close", "volume"):
                cols[f"{a}__{c}"] = rng.normal(0, 1, n)
    for c in MACRO:
        cols[f"MACRO__{c}"] = rng.normal(0, 1, n)
    return pd.DataFrame(cols, index=idx)


def _reward_kwargs(**over):
    kw = dict(
        prev_equity=1_000_000.0,
        new_equity=1_000_000.0,
        weights=np.array([0.3, 0.3, 0.0]),
        prev_weights=np.array([0.1, 0.5, 0.0]),
        traded_notional=np.zeros(3),
        daily_volume_usd=np.full(3, 1e9),
        sentiment=np.zeros(3),
        state=RewardState.create(peak_equity=1_000_000.0),
        cfg={"weights": {"return": 1.0, "sharpe": 0.0, "drawdown": 0.0, "liquidity": 0.0,
                         "sentiment": 0.0, "turnover": 1.0},
             "liquidity_threshold_pct": 0.01},
    )
    kw.update(over)
    return kw


def test_turnover_charges_trades_not_drift():
    # No trades, weights drifted passively -> zero turnover penalty.
    _, comps = compute(**_reward_kwargs())
    assert comps["r_turnover"] == 0.0
    # Trades happened -> penalty = notional / prev_equity.
    _, comps = compute(**_reward_kwargs(traded_notional=np.array([100_000.0, 0.0, 0.0])))
    assert comps["r_turnover"] == pytest.approx(-0.1)


def test_return_is_log_return():
    _, comps = compute(**_reward_kwargs(new_equity=1_100_000.0))
    assert comps["r_return"] == pytest.approx(np.log(1.1))


def test_episode_horizon_enforced():
    env = CryptoTradingEnv(_panel(), _cfg(), enforce_horizon=True)
    env.reset(seed=1)
    steps = 0
    done = False
    while not done and steps < 500:
        _, _, term, trunc, _ = env.step(np.zeros(3))
        done = term or trunc
        steps += 1
    assert steps == 10  # episode_horizon_bars

    env = CryptoTradingEnv(_panel(60), _cfg(), episode_start_idx=30, enforce_horizon=False)
    env.reset(seed=1)
    steps = 0
    done = False
    while not done and steps < 500:
        _, _, term, trunc, _ = env.step(np.zeros(3))
        done = term or trunc
        steps += 1
    assert steps == 29  # runs to the end of the panel, ignoring the horizon


def test_obs_parity_env_vs_live():
    """The live loop must build byte-identical observations to the training env."""
    panel = _panel()
    cfg = _cfg()
    env = CryptoTradingEnv(panel, cfg, episode_start_idx=len(panel) - 1)
    env_obs, _ = env.reset(seed=0)
    live_obs = _observation_from_panel(panel, cfg, weights=np.zeros(3), cash_frac=1.0)
    assert env_obs.shape == live_obs.shape
    np.testing.assert_allclose(env_obs, live_obs, rtol=0, atol=1e-6)


def test_baselines_charge_fees():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100, 110, 50), "B": np.linspace(50, 55, 50)}, index=idx)
    free = buy_and_hold(prices, 1_000_000, fee_bps=0.0)
    paid = buy_and_hold(prices, 1_000_000, fee_bps=10.0)
    assert paid.iloc[-1] < free.iloc[-1]
    # Rebalanced variant trades every bar, so fees must compound to a bigger gap.
    reb_free = bh_rebalanced(prices, 1_000_000, fee_bps=0.0)
    reb_paid = bh_rebalanced(prices, 1_000_000, fee_bps=10.0)
    assert reb_paid.iloc[-1] < reb_free.iloc[-1]
    # Weights applied with one-bar delay: a strategy that only targets weights on
    # the final bar can never earn any return.
    w = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    w.iloc[-1] = 0.5
    eq = equity_from_weights(prices, w, 1_000_000, fee_bps=0.0)
    assert eq.iloc[-2] == pytest.approx(1_000_000)


def test_kill_switch_window(tmp_path):
    import time as _time
    broker = PaperBroker(tmp_path / "t.db", assets=["BTC"], initial_cash=1_000_000,
                         fee_bps=0, slippage_std=[0.0])
    cfg = {"kill_switch": {"enabled": True, "max_daily_loss": 0.05,
                           "max_drawdown_peak": 0.50, "window_hours": 3}}
    # Slow bleed of ~2%/hour: no single-step 5% loss, but vs 3 HOURS ago it's -6%.
    # Extra rows inside the window (e.g. from /run-now) must not shrink it.
    now = int(_time.time())
    rows = [(now - 4 * 3600, 1_000_000), (now - 2 * 3600, 980_000), (now - 3600, 960_000)]
    for ts, v in rows:
        broker.db.execute("INSERT OR REPLACE INTO equity(ts, value, weights) VALUES (?,?,?)",
                          (ts, v, "[0.0]"))
    broker.db.commit()
    assert broker.equity_at_or_before(now - 3 * 3600) == 1_000_000
    assert broker.equity_peak() == 1_000_000
    halt, reason = check_and_update(broker, 940_000, cfg)
    assert halt and "daily loss" in reason
    assert is_halted(broker)


def test_decommission_state(tmp_path):
    broker = PaperBroker(tmp_path / "d.db", assets=["BTC"], initial_cash=1_000_000,
                         fee_bps=0, slippage_std=[0.0])
    set_halt(broker, "no-qualified-model", decommission=True)
    assert is_halted(broker) and is_decommissioned(broker)
    # No nagging reminders and no auto-resume for a decommissioned service.
    assert maybe_send_halted_reminder(broker) is False
    cfg = {"kill_switch": {"enabled": True, "auto_resume_hours": 0.0001}}
    halt, reason = check_and_update(broker, 1_000_000, cfg)
    assert halt and reason == "decommissioned"
    # reset() brings it back to life.
    broker.reset(1_000_000)
    assert not is_halted(broker) and not is_decommissioned(broker)


def test_gold_signal_states():
    from ppofolio.signals.watch import compute_signals
    idx = pd.date_range("2024-01-01", periods=320, freq="D", tz="UTC")
    # Accelerating trends so MACD momentum is decisively signed at the last bar
    # (a linear ramp converges macd_diff to ~0 and makes the test flaky).
    # MACD works in absolute price units, so the trends must accelerate in
    # ABSOLUTE terms (a %-wise steady decline flattens as the price shrinks and
    # the crossover legitimately reads "momentum improving").
    x = np.linspace(0, 1.0, 320)
    up = pd.Series(2000 + 2000 * x ** 2, index=idx)
    sig = compute_signals(up)
    assert sig["state"] == "bullish" and sig["above_ma200"] and sig["momentum_up"]
    down = pd.Series(3000 - 2000 * x ** 2, index=idx)
    sig = compute_signals(down)
    assert sig["state"] == "bearish" and not sig["above_ma200"] and not sig["momentum_up"]
    for k in ("price", "ma200", "rsi", "as_of"):
        assert k in sig


def test_broker_rejects_non_finite():
    import tempfile
    broker = PaperBroker(tempfile.mktemp(), assets=["BTC"], initial_cash=1_000_000,
                         fee_bps=0, slippage_std=[0.0])
    with pytest.raises(ValueError):
        broker.rebalance(np.array([np.nan]), np.array([100.0]))
    with pytest.raises(ValueError):
        broker.rebalance(np.array([0.5]), np.array([np.inf]))


def _eval_system_module():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_system
    return eval_system


def test_env_fee_reconciliation():
    cfg = _cfg()
    env = CryptoTradingEnv(_panel(), cfg, episode_start_idx=30, enforce_horizon=False)
    env.reset(seed=0)
    cash0 = env.cash
    _, _, _, _, info = env.step(np.array([0.3, 0.2, 0.1]))
    fee = cfg["fee_bps"] / 10_000
    assert info["traded_notional"] > 0
    assert info["fees_paid"] == pytest.approx(info["traded_notional"] * fee)
    # from all-cash, every delta is a buy: cash falls by traded notional + fees
    assert cash0 - env.cash == pytest.approx(info["traded_notional"] + info["fees_paid"])


def test_band_widening_requires_reorder():
    # Old live order (limits, then band): the $50k per-trade cap keeps every
    # post-limits leg below 50k/1M = 5% weight, so a 0.08 band would skip every
    # trade forever. Band-first on the desired move trades. Documents why the
    # loop applies the band before vol scale + limits.
    from ppofolio.live.risk_limits import apply_limits
    cfg = _cfg()
    cfg["invest_limits"] = {"max_trade_notional": 50_000, "min_trade_notional": 10}
    equity = 1_000_000.0
    weights = np.zeros(3)
    desired = np.array([0.3, 0.0, 0.0])
    limited, _ = apply_limits(desired, weights * equity, equity, cfg)
    thr = 0.08
    assert np.all(np.abs(limited - weights) < thr)          # old order: frozen
    assert np.any(np.abs(desired - weights) >= thr)         # fixed order: trades


def test_roll_system_limits_mode_trades_and_tracks_fees():
    eval_system = _eval_system_module()
    cfg = _cfg()
    cfg["invest_limits"] = {"max_trade_notional": 50_000, "min_trade_notional": 10}
    cfg["action"]["rebalance_threshold"] = 0.08
    cfg["kill_switch"] = {"enabled": False}

    class _Stub:
        def predict(self, obs, deterministic=True):
            return np.array([0.5, 0.3, 0.2]), None

    eq, meta = eval_system.roll_system(_Stub(), _panel(), cfg, resume_policy="off",
                                       limits_mode=True)
    assert meta["gross_traded"] > 0                          # not frozen under 0.08 band
    assert meta["fees_paid"] == pytest.approx(meta["gross_traded"] * 0.001)
    assert len(eq) > 100


def test_set_by_path_overrides():
    eval_system = _eval_system_module()
    cfg = {"action": {"rebalance_threshold": 0.02}}
    eval_system.set_by_path(cfg, "action.rebalance_threshold", "0.05")
    assert cfg["action"]["rebalance_threshold"] == 0.05
    eval_system.set_by_path(cfg, "new.nested.flag", "true")
    assert cfg["new"]["nested"]["flag"] is True
