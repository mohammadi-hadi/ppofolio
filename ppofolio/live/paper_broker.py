"""Paper broker: SQLite-backed simulator that tracks holdings, cash, trades.

Fills at current mid-price with gaussian slippage + configurable fee, matching the
model used during training (ppofolio/env/crypto_trading_env.py).
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from ppofolio.live.alerts import format_alert, send as send_alert


SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
  asset TEXT PRIMARY KEY,
  units REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  asset TEXT NOT NULL,
  delta_units REAL NOT NULL,
  exec_price REAL NOT NULL,
  fee REAL NOT NULL,
  notional REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
  ts INTEGER PRIMARY KEY,
  value REAL NOT NULL,
  weights TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades_archive (
  ts INTEGER, asset TEXT, delta_units REAL, exec_price REAL, fee REAL, notional REAL,
  archived_at INTEGER
);
CREATE TABLE IF NOT EXISTS equity_archive (
  ts INTEGER, value REAL, weights TEXT,
  archived_at INTEGER
);
"""


class PaperBroker:
    def __init__(self, db_path: str | Path, assets: list[str], initial_cash: float, fee_bps: float,
                 slippage_std: list[float], borrow_rate_daily: float = 0.0):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.assets = assets
        self.fee = fee_bps / 10_000.0
        self.slip_std = np.asarray(slippage_std, dtype=float)
        self.borrow_rate_daily = float(borrow_rate_daily)
        self._init_state(initial_cash)
        self._rng = np.random.default_rng()

    def _deduct_borrow_cost(self, price: np.ndarray, ts: int) -> float:
        """Charge borrow cost on short positions based on elapsed time since last update."""
        if self.borrow_rate_daily <= 0:
            return 0.0
        holdings = self.holdings()
        shorts_notional = float(np.sum(np.maximum(0.0, -holdings) * price))
        if shorts_notional <= 0:
            return 0.0
        with self._lock:
            row = self.db.execute("SELECT v FROM meta WHERE k='last_ts'").fetchone()
            last_ts = int(row[0]) if row else ts
        elapsed_days = max(0.0, (ts - last_ts) / 86_400.0)
        cost = shorts_notional * self.borrow_rate_daily * elapsed_days
        if cost > 0:
            self.cash = self.cash - cost
        return cost

    def _init_state(self, initial_cash: float) -> None:
        with self._lock:
            cur = self.db.cursor()
            cur.execute("SELECT COUNT(*) FROM meta WHERE k='cash'")
            if cur.fetchone()[0] == 0:
                cur.execute("INSERT INTO meta(k, v) VALUES ('cash', ?)", (str(initial_cash),))
                for a in self.assets:
                    cur.execute("INSERT OR IGNORE INTO state(asset, units) VALUES (?, 0)", (a,))
                self.db.commit()

    @property
    def cash(self) -> float:
        with self._lock:
            row = self.db.execute("SELECT v FROM meta WHERE k='cash'").fetchone()
        return float(row[0])

    @cash.setter
    def cash(self, v: float) -> None:
        with self._lock:
            self.db.execute("UPDATE meta SET v=? WHERE k='cash'", (str(v),))

    def holdings(self) -> np.ndarray:
        with self._lock:
            rows = dict(self.db.execute("SELECT asset, units FROM state").fetchall())
        return np.array([rows.get(a, 0.0) for a in self.assets], dtype=float)

    def equity(self, price: np.ndarray) -> float:
        return float(self.cash + (self.holdings() * price).sum())

    def reset(self, initial_cash: float, ts: int | None = None) -> None:
        """Archive history and restart the account flat at initial_cash, halt cleared."""
        ts = ts or int(time.time())
        with self._lock:
            cur = self.db.cursor()
            cur.execute(
                "INSERT INTO trades_archive SELECT ts, asset, delta_units, exec_price, fee, notional, ? FROM trades",
                (ts,))
            cur.execute("INSERT INTO equity_archive SELECT ts, value, weights, ? FROM equity", (ts,))
            cur.execute("DELETE FROM trades")
            cur.execute("DELETE FROM equity")
            cur.execute("UPDATE state SET units=0")
            cur.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('cash', ?)", (str(initial_cash),))
            cur.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('halted', '0')")
            cur.execute("DELETE FROM meta WHERE k IN ('last_ts', 'last_halt_reminder', 'halt_reason', "
                        "'halted_at', 'peak_since', 'decommissioned')")
            cur.execute("INSERT INTO equity(ts, value, weights) VALUES (?,?,?)",
                        (ts, float(initial_cash), json.dumps([0.0] * len(self.assets))))
            self.db.commit()

    def rebalance(self, target_w: np.ndarray, price: np.ndarray, ts: int | None = None) -> dict:
        ts = ts or int(time.time())
        target_w = np.asarray(target_w, dtype=float)
        price = np.asarray(price, dtype=float)
        if not (np.isfinite(target_w).all() and np.isfinite(price).all()):
            raise ValueError("rebalance rejected: non-finite target weights or prices")
        # deduct borrow cost before computing equity (so trades happen on post-cost cash)
        borrow_cost = self._deduct_borrow_cost(price, ts)
        eq = self.equity(price)
        holdings = self.holdings()
        target_notional = target_w * eq
        delta_notional = target_notional - holdings * price

        slip = self._rng.normal(0.0, self.slip_std)
        exec_price = price * (1.0 + slip)
        fees = np.abs(delta_notional) * self.fee
        delta_units = np.where(exec_price > 0, delta_notional / exec_price, 0.0)
        # Full exits snap to exactly zero units: units are computed at the slipped
        # exec price while notional is at mid, which leaves ±dust that can flip a
        # long-only book short.
        full_exit = (target_w == 0.0) & (np.abs(holdings) > 0)
        delta_units = np.where(full_exit, -holdings, delta_units)

        new_cash = self.cash - float(delta_notional.sum()) - float(fees.sum())
        new_holdings = holdings + delta_units
        equity_now = new_cash + float((new_holdings * price).sum())
        weights = (new_holdings * price) / max(equity_now, 1e-9)
        ts_iso = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        executed: list[dict] = []
        with self._lock:
            cur = self.db.cursor()
            cur.execute("UPDATE meta SET v=? WHERE k='cash'", (str(new_cash),))
            for a, u in zip(self.assets, new_holdings):
                cur.execute("UPDATE state SET units=? WHERE asset=?", (float(u), a))
            for i, a in enumerate(self.assets):
                if abs(delta_units[i]) > 1e-12:
                    cur.execute(
                        "INSERT INTO trades(ts, asset, delta_units, exec_price, fee, notional) VALUES (?,?,?,?,?,?)",
                        (ts, a, float(delta_units[i]), float(exec_price[i]), float(fees[i]), float(delta_notional[i])),
                    )
                    executed.append({
                        "asset": a,
                        "price": float(exec_price[i]),
                        "delta_units": float(delta_units[i]),
                        "prev_units": float(holdings[i]),
                        "new_units": float(new_holdings[i]),
                        "weight": float(weights[i]),
                    })
            cur.execute("INSERT OR REPLACE INTO equity(ts, value, weights) VALUES (?,?,?)", (ts, equity_now, json.dumps(weights.tolist())))
            cur.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('last_ts', ?)", (str(ts),))
            self.db.commit()

        # Fire one alert per executed trade (outside the DB lock).
        for t in executed:
            try:
                msg = format_alert(ts_iso=ts_iso, **t)
                sign = 1 if t["delta_units"] > 0 else -1
                send_alert(msg, asset=t["asset"], sign=sign)
            except Exception:  # never let alerting break the broker
                pass

        return {"equity": equity_now, "cash": new_cash, "weights": weights.tolist(),
                "fees": float(fees.sum()), "borrow_cost": borrow_cost}

    def recent_trades(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT ts, asset, delta_units, exec_price, fee, notional FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"ts": r[0], "asset": r[1], "delta_units": r[2], "exec_price": r[3], "fee": r[4], "notional": r[5]}
            for r in rows
        ]

    def equity_curve(self, since: int | None = None, limit: int | None = None) -> list[dict]:
        q = "SELECT ts, value, weights FROM equity"
        params: list = []
        if since is not None:
            q += " WHERE ts >= ?"
            params.append(int(since))
        q += " ORDER BY ts"
        if limit is not None:
            q = f"SELECT * FROM ({q.replace(' ORDER BY ts', ' ORDER BY ts DESC')} LIMIT ?) ORDER BY ts"
            params.append(int(limit))
        with self._lock:
            rows = self.db.execute(q, params).fetchall()
        return [{"ts": r[0], "value": r[1], "weights": json.loads(r[2])} for r in rows]

    def equity_at_or_before(self, ts: int) -> float | None:
        with self._lock:
            row = self.db.execute(
                "SELECT value FROM equity WHERE ts <= ? ORDER BY ts DESC LIMIT 1", (int(ts),)).fetchone()
        return float(row[0]) if row else None

    def equity_peak(self, since: int | None = None) -> float | None:
        q = "SELECT MAX(value) FROM equity"
        params: list = []
        if since is not None:
            q += " WHERE ts >= ?"
            params.append(int(since))
        with self._lock:
            row = self.db.execute(q, params).fetchone()
        return float(row[0]) if row and row[0] is not None else None
