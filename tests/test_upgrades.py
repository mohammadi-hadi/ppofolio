"""Tests for the 2026-07-18 upgrade batch: full-exit dust clamp, recovery-gated
kill-switch resume, ensemble disagreement scaling, new baselines, signal history."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ppofolio.agents.ensemble import EnsemblePolicy
from ppofolio.backtest.baselines import risk_parity, tsmom_ma200
from ppofolio.env.crypto_trading_env import CryptoTradingEnv
from ppofolio.live.kill_switch import basket_recovery_ok, check_and_update, is_halted, set_halt
from ppofolio.live.paper_broker import PaperBroker
from ppofolio.signals.watch import read_history, record_history
from tests.test_correctness import _cfg, _panel


def _broker(tmp_path, slippage=0.005):
    return PaperBroker(tmp_path / "t.db", assets=["BTC", "ETH", "SOL"], initial_cash=1_000_000,
                       fee_bps=10, slippage_std=[slippage] * 3)


# ---- full-exit dust clamp ----

def test_broker_full_exit_leaves_zero_units(tmp_path):
    b = _broker(tmp_path)  # large slippage: units at exec price != notional at mid
    price = np.array([100.0, 50.0, 10.0])
    b.rebalance(np.array([0.3, 0.2, 0.1]), price)
    b.rebalance(np.array([0.0, 0.0, 0.0]), price)
    assert np.array_equal(b.holdings(), np.zeros(3)), b.holdings()


def test_broker_partial_trade_still_slips(tmp_path):
    b = _broker(tmp_path)
    price = np.array([100.0, 50.0, 10.0])
    b.rebalance(np.array([0.3, 0.0, 0.0]), price)
    b.rebalance(np.array([0.1, 0.0, 0.0]), price)  # partial sell — no snap
    assert b.holdings()[0] > 0


def test_env_full_exit_leaves_zero_units():
    cfg = _cfg()
    cfg["assets"] = [dict(a, slippage_std=0.005) for a in cfg["assets"]]
    env = CryptoTradingEnv(_panel(), cfg, episode_start_idx=30, enforce_horizon=False)
    env.reset(seed=0)
    env.step(np.array([0.3, 0.2, 0.1]))
    env.step(np.zeros(3))
    assert np.array_equal(env.holdings, np.zeros(3)), env.holdings


# ---- basket recovery + gated resume ----

def test_basket_recovery_states():
    up = np.linspace(80, 120, 40).reshape(-1, 1) * np.array([[1.0, 2.0, 3.0]])
    down = np.linspace(120, 80, 40).reshape(-1, 1) * np.array([[1.0, 2.0, 3.0]])
    assert basket_recovery_ok(up) is True
    assert basket_recovery_ok(down) is False
    assert basket_recovery_ok(up[:5]) is None  # not enough history


def test_resume_waits_for_recovery(tmp_path):
    b = _broker(tmp_path, slippage=0.0)
    ks = {"kill_switch": {"enabled": True, "auto_resume_hours": 0.0001,
                          "resume_requires_recovery": True,
                          "max_daily_loss": 0.99, "max_drawdown_peak": 0.99}}
    set_halt(b, "test halt")
    time.sleep(0.5)  # cooldown (0.36s) elapses
    halted, reason = check_and_update(b, 1_000_000.0, ks, recovery_ok=False)
    assert halted and is_halted(b) and "awaiting" in reason
    halted, _ = check_and_update(b, 1_000_000.0, ks, recovery_ok=True)
    assert not halted and not is_halted(b)


def test_resume_fails_open_when_recovery_unknown(tmp_path):
    b = _broker(tmp_path, slippage=0.0)
    ks = {"kill_switch": {"enabled": True, "auto_resume_hours": 0.0001,
                          "resume_requires_recovery": True,
                          "max_daily_loss": 0.99, "max_drawdown_peak": 0.99}}
    set_halt(b, "test halt")
    time.sleep(0.5)
    halted, _ = check_and_update(b, 1_000_000.0, ks, recovery_ok=None)
    assert not halted  # unknown data must not strand the account halted


# ---- ensemble disagreement scaling ----

class _Fixed:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def predict(self, obs, deterministic=True):
        return self.a


def test_disagreement_scales_exposure_down():
    ens = EnsemblePolicy.__new__(EnsemblePolicy)
    ens.members = [_Fixed([0.8, 0.0, 0.0]), _Fixed([0.0, 0.8, 0.0])]
    ens.disagreement = {"enabled": True, "k": 2.0, "floor": 0.25}
    scaled, _ = ens.predict(np.zeros(4))
    ens.disagreement = {}
    plain, _ = ens.predict(np.zeros(4))
    assert np.all(scaled <= plain + 1e-12) and scaled.sum() < plain.sum()


def test_agreement_leaves_action_alone():
    ens = EnsemblePolicy.__new__(EnsemblePolicy)
    ens.members = [_Fixed([0.5, 0.2, 0.1]), _Fixed([0.5, 0.2, 0.1])]
    ens.disagreement = {"enabled": True, "k": 2.0, "floor": 0.25}
    out, _ = ens.predict(np.zeros(4))
    assert np.allclose(out, [0.5, 0.2, 0.1])


# ---- new baselines ----

def test_new_baselines_sane():
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {a: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 400))) for a in ["BTC", "ETH", "SOL"]},
        index=idx)
    window = prices.iloc[250:]
    for eq in (risk_parity(window, history=prices), tsmom_ma200(window, history=prices)):
        assert len(eq) == len(window)
        assert (eq > 0).all()
        assert np.isfinite(eq).all()


def test_tsmom_goes_to_cash_in_downtrend():
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    prices = pd.DataFrame({a: np.linspace(200, 50, 400) for a in ["BTC", "ETH"]}, index=idx)
    eq = tsmom_ma200(prices.iloc[300:], history=prices, initial_cash=1000.0)
    assert eq.iloc[-1] == 1000.0  # every asset below its MA -> fully cash, no losses


# ---- signal history ----

def test_signal_history_roundtrip(tmp_path):
    b = _broker(tmp_path, slippage=0.0)
    sig = {"state": "bullish", "rsi_zone": "normal", "price": 4000.0, "rsi": 55.0}
    record_history(b, 1_784_000_000, "Gold", None, sig)
    record_history(b, 1_784_100_000, "Gold", "bullish", dict(sig, state="neutral"))
    rows = read_history(b, limit=10)
    assert [r["state"] for r in rows] == ["neutral", "bullish"]  # newest first
    assert rows[0]["prev_state"] == "bullish" and rows[0]["asset"] == "Gold"
