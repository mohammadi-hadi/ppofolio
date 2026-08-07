"""Postgres-backed broker. Same public interface as PaperBroker.

Activated when DATABASE_URL is set (most container platforms inject this when you
attach a Postgres plugin to the service).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading

import numpy as np
import psycopg
from psycopg.rows import dict_row

from ppofolio.live.alerts import format_alert, send as send_alert

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS broker_state (
    asset TEXT PRIMARY KEY,
    units DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS broker_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS broker_trades (
    id BIGSERIAL PRIMARY KEY,
    ts BIGINT NOT NULL,
    asset TEXT NOT NULL,
    delta_units DOUBLE PRECISION NOT NULL,
    exec_price DOUBLE PRECISION NOT NULL,
    fee DOUBLE PRECISION NOT NULL,
    notional DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS broker_trades_ts_idx ON broker_trades(ts DESC);

CREATE TABLE IF NOT EXISTS broker_equity (
    ts BIGINT PRIMARY KEY,
    value DOUBLE PRECISION NOT NULL,
    weights TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS broker_trades_archive (
    ts BIGINT, asset TEXT, delta_units DOUBLE PRECISION, exec_price DOUBLE PRECISION,
    fee DOUBLE PRECISION, notional DOUBLE PRECISION, archived_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS broker_equity_archive (
    ts BIGINT, value DOUBLE PRECISION, weights TEXT, archived_at TIMESTAMPTZ DEFAULT NOW()
);
"""


class PostgresBroker:
    def __init__(self, dsn: str, assets: list[str], initial_cash: float,
                 fee_bps: float, slippage_std: list[float], borrow_rate_daily: float = 0.0):
        self.dsn = dsn
        self.assets = assets
        self.fee = fee_bps / 10_000.0
        self.slip_std = np.asarray(slippage_std, dtype=float)
        self.borrow_rate_daily = float(borrow_rate_daily)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()
        self._ensure_schema()
        self._init_state(initial_cash)

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def _ensure_schema(self):
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(SCHEMA)

    def _init_state(self, initial_cash: float):
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM broker_meta WHERE k='cash'")
                if cur.fetchone()[0] == 0:
                    cur.execute("INSERT INTO broker_meta(k, v) VALUES ('cash', %s)", (str(initial_cash),))
                    for a in self.assets:
                        cur.execute(
                            "INSERT INTO broker_state(asset, units) VALUES (%s, 0) ON CONFLICT DO NOTHING",
                            (a,),
                        )

    @property
    def cash(self) -> float:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT v FROM broker_meta WHERE k='cash'")
            row = cur.fetchone()
            return float(row[0]) if row else 0.0

    def holdings(self) -> np.ndarray:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT asset, units FROM broker_state")
            rows = dict(cur.fetchall())
        return np.array([rows.get(a, 0.0) for a in self.assets], dtype=float)

    def equity(self, price: np.ndarray) -> float:
        return float(self.cash + (self.holdings() * price).sum())

    def reset(self, initial_cash: float, ts: int | None = None) -> None:
        """Archive history and restart the account flat at initial_cash, halt cleared.

        Runs as ONE transaction (autocommit off) so a crash mid-reset cannot leave
        half-cleared state.
        """
        ts = ts or int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp())
        with self._lock, psycopg.connect(self.dsn) as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO broker_trades_archive(ts, asset, delta_units, exec_price, fee, notional) "
                "SELECT ts, asset, delta_units, exec_price, fee, notional FROM broker_trades")
            cur.execute(
                "INSERT INTO broker_equity_archive(ts, value, weights) "
                "SELECT ts, value, weights FROM broker_equity")
            cur.execute("DELETE FROM broker_trades")
            cur.execute("DELETE FROM broker_equity")
            cur.execute("UPDATE broker_state SET units=0")
            cur.execute(
                "INSERT INTO broker_meta(k, v) VALUES ('cash', %s) "
                "ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v", (str(initial_cash),))
            cur.execute(
                "INSERT INTO broker_meta(k, v) VALUES ('halted', '0') "
                "ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v")
            cur.execute("DELETE FROM broker_meta WHERE k IN ('last_ts', 'last_halt_reminder', 'halt_reason', "
                        "'halted_at', 'peak_since', 'decommissioned')")
            cur.execute(
                "INSERT INTO broker_equity(ts, value, weights) VALUES (%s, %s, %s)",
                (ts, float(initial_cash), json.dumps([0.0] * len(self.assets))))

    def rebalance(self, target_w: np.ndarray, price: np.ndarray, ts: int | None = None) -> dict:
        """Execute a rebalance as ONE read-modify-write transaction.

        All reads (cash, holdings, last_ts) and writes happen on the same
        non-autocommit connection so a crash mid-rebalance rolls back cleanly and
        concurrent readers never observe half-applied state.
        """
        ts = ts or int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp())
        target_w = np.asarray(target_w, dtype=float)
        price = np.asarray(price, dtype=float)
        if not (np.isfinite(target_w).all() and np.isfinite(price).all()):
            raise ValueError("rebalance rejected: non-finite target weights or prices")

        slip = self._rng.normal(0.0, self.slip_std)
        ts_iso = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        executed: list[dict] = []
        with self._lock, psycopg.connect(self.dsn) as c, c.cursor() as cur:
            cur.execute("SELECT v FROM broker_meta WHERE k='cash'")
            row = cur.fetchone()
            cash = float(row[0]) if row else 0.0
            cur.execute("SELECT asset, units FROM broker_state")
            units_by_asset = dict(cur.fetchall())
            holdings = np.array([units_by_asset.get(a, 0.0) for a in self.assets], dtype=float)

            # Borrow cost on short exposure since the last update.
            borrow_cost = 0.0
            if self.borrow_rate_daily > 0:
                shorts_notional = float(np.sum(np.maximum(0.0, -holdings) * price))
                if shorts_notional > 0:
                    cur.execute("SELECT v FROM broker_meta WHERE k='last_ts'")
                    row = cur.fetchone()
                    last_ts = int(row[0]) if row else ts
                    elapsed_days = max(0.0, (ts - last_ts) / 86_400.0)
                    borrow_cost = shorts_notional * self.borrow_rate_daily * elapsed_days
                    cash -= borrow_cost

            eq = cash + float((holdings * price).sum())
            target_notional = target_w * eq
            delta_notional = target_notional - holdings * price
            exec_price = price * (1.0 + slip)
            fees = np.abs(delta_notional) * self.fee
            delta_units = np.where(exec_price > 0, delta_notional / exec_price, 0.0)
            # Full exits snap to exactly zero units: units are computed at the
            # slipped exec price while notional is at mid, which leaves ±dust
            # that can flip a long-only book short.
            full_exit = (target_w == 0.0) & (np.abs(holdings) > 0)
            delta_units = np.where(full_exit, -holdings, delta_units)

            new_cash = cash - float(delta_notional.sum()) - float(fees.sum())
            new_holdings = holdings + delta_units
            equity_now = new_cash + float((new_holdings * price).sum())
            weights = (new_holdings * price) / max(equity_now, 1e-9)

            cur.execute("UPDATE broker_meta SET v=%s WHERE k='cash'", (str(new_cash),))
            for a, u in zip(self.assets, new_holdings):
                cur.execute("UPDATE broker_state SET units=%s WHERE asset=%s", (float(u), a))
            for i, a in enumerate(self.assets):
                if abs(delta_units[i]) > 1e-12:
                    cur.execute(
                        "INSERT INTO broker_trades(ts, asset, delta_units, exec_price, fee, notional) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (ts, a, float(delta_units[i]), float(exec_price[i]), float(fees[i]), float(delta_notional[i])),
                    )
                    executed.append({
                        "asset": a, "price": float(exec_price[i]),
                        "delta_units": float(delta_units[i]),
                        "prev_units": float(holdings[i]),
                        "new_units": float(new_holdings[i]),
                        "weight": float(weights[i]),
                    })
            cur.execute(
                "INSERT INTO broker_equity(ts, value, weights) VALUES (%s, %s, %s) "
                "ON CONFLICT (ts) DO UPDATE SET value=EXCLUDED.value, weights=EXCLUDED.weights",
                (ts, equity_now, json.dumps(weights.tolist())),
            )
            cur.execute(
                "INSERT INTO broker_meta(k, v) VALUES ('last_ts', %s) "
                "ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v",
                (str(ts),),
            )

        for t in executed:
            try:
                sign = 1 if t["delta_units"] > 0 else -1
                send_alert(format_alert(ts_iso=ts_iso, **t), asset=t["asset"], sign=sign)
            except Exception:
                pass

        return {"equity": equity_now, "cash": new_cash, "weights": weights.tolist(),
                "fees": float(fees.sum()), "borrow_cost": borrow_cost}

    def recent_trades(self, limit: int = 50) -> list[dict]:
        with self._conn() as c, c.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT ts, asset, delta_units, exec_price, fee, notional "
                "FROM broker_trades ORDER BY id DESC LIMIT %s", (limit,))
            return list(cur.fetchall())

    def equity_curve(self, since: int | None = None, limit: int | None = None) -> list[dict]:
        q = "SELECT ts, value, weights FROM broker_equity"
        params: list = []
        if since is not None:
            q += " WHERE ts >= %s"
            params.append(int(since))
        q += " ORDER BY ts"
        if limit is not None:
            # newest N rows, still returned in ascending order
            q = f"SELECT * FROM ({q.replace(' ORDER BY ts', ' ORDER BY ts DESC')} LIMIT %s) sub ORDER BY ts"
            params.append(int(limit))
        with self._conn() as c, c.cursor(row_factory=dict_row) as cur:
            cur.execute(q, params)
            rows = cur.fetchall()
        for r in rows:
            r["weights"] = json.loads(r["weights"])
        return rows

    def equity_at_or_before(self, ts: int) -> float | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT value FROM broker_equity WHERE ts <= %s ORDER BY ts DESC LIMIT 1", (int(ts),))
            row = cur.fetchone()
            return float(row[0]) if row else None

    def equity_peak(self, since: int | None = None) -> float | None:
        q = "SELECT MAX(value) FROM broker_equity"
        params: list = []
        if since is not None:
            q += " WHERE ts >= %s"
            params.append(int(since))
        with self._conn() as c, c.cursor() as cur:
            cur.execute(q, params)
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None


def make_broker(assets, initial_cash, fee_bps, slippage_std, sqlite_fallback_path,
                borrow_rate_daily: float = 0.0):
    """Return PostgresBroker if DATABASE_URL is set, else PaperBroker (SQLite)."""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        log.info("using PostgresBroker")
        return PostgresBroker(dsn, assets, initial_cash, fee_bps, slippage_std,
                              borrow_rate_daily=borrow_rate_daily)
    from ppofolio.live.paper_broker import PaperBroker
    log.info("DATABASE_URL not set — falling back to SQLite at %s", sqlite_fallback_path)
    return PaperBroker(sqlite_fallback_path, assets, initial_cash, fee_bps, slippage_std,
                       borrow_rate_daily=borrow_rate_daily)
