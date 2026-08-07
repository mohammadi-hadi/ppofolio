# Evaluation protocol — pre-registered

Written 2026-07-18, BEFORE the results it governs exist. Rules here are fixed in
advance so evaluation cannot drift into vibes; change them only prospectively
(never for a window already running) and log the change in this file.

## Forward paper evaluation (v5.1 daily, production)

- **Window**: 2026-07-18 → 2026-10-16 (90 days). Baseline = the `/admin/reset`
  performed on 2026-07-18 after deploying the execution-layer upgrade batch.
- **Metrics** (computed from `/equity` + `/benchmark` at window end):
  1. total return vs the equal-notional buy-and-hold benchmark (`/benchmark`)
  2. Sharpe of daily log returns (rf = 0, annualized ×√365)
  3. max drawdown
  4. fees as % of gross P&L (churn monitor)
- **Judgment** (one evaluation, at window end — no peeking-based decisions):
  - *Continue forward eval*: return ≥ benchmark − 2pts, or Sharpe > 0.5.
  - *Retire v5.1*: return < benchmark − 10pts, or a kill-switch halt caused by
    model behavior rather than market-wide crash.
  - Anything between: judgment call, written down with reasons.
- Interim looks are fine for ops (halts, bugs) but trigger no model decisions.

## Held-out test gates (for any new candidate, e.g. v5.2)

- Selection happens ONLY on the val split (2025-10-01..2026-01-31 for v5.2).
- The held-out test window is touched ONCE per candidate generation, then burned:
  - burned: 2026-02-15..2026-07-10 (v5.1 gate, verdict in models/MANIFEST.json)
  - next gate: a window that ends ≤ 14 days before the gate date and starts
    after the last burned window ends, min 8 weeks long. For a gate run in
    Oct 2026 that means ≈ 2026-07-18..2026-10-01.
- **Gate bar** (deploy for forward paper eval): beat `bh_hold` AND `tsmom_ma200`
  on total return on the held-out window, with max drawdown no worse than
  1.5× bh_hold's. (tsmom added 2026-07-18 — beating buy-and-hold alone is a
  regime-dependent bar; the 200d-MA trend filter is the honest simple rival.)
- Baselines reported on every window: bh_hold, bh_rebalanced, macd,
  risk_parity, tsmom_ma200.
- The SYSTEM (policy + circuit breaker + kill switch, via scripts/eval_system.py)
  is what gates, not the raw policy.

## Execution-layer change log

| date | change | evidence |
|---|---|---|
| 2026-07-18 | full-exit dust clamp (brokers + env) | SOL −0.045-unit dust on long-only book, live |
| 2026-07-18 | kill switch: max_daily_loss 0.05→0.08 | 2 spurious halts on val cost 13.5pts (reports/system_abval_*.csv: off +16.5% / timer +2.9%) |
| 2026-07-18 | kill switch: resume gated on basket>MA10 after cooldown | timer resume re-bought falling markets (test window: 4 halts, −4.3pts vs raw) |

## Rebalance-band A/B — rule pre-registered 2026-08-07, BEFORE results

Live evidence (18 days of forward window): $954k gross notional churned to hold
$20–70k net exposure; fees $954 vs equity change −$601 — fees flipped the sign
of the window. Root cause: `apply_limits` truncates every leg to
`max_trade_notional` $50k ≈ 5% weight BEFORE the 2% band check, so the daily
±5pt flip-flop always clears the band; widening the band without reordering
would freeze the book (post-limits |Δw| ≤ ~5.2% < 0.08). Fix under test:
band moves BEFORE vol scale + limits (training-env semantics, band vs current
weights) + wider threshold. Simulated via `eval_system.py --limits`.

- **Sweep**: `action.rebalance_threshold` ∈ {0.02 control, 0.05, 0.08, 0.10},
  val window 2025-10-01..2026-01-31, resume-policy recovery, slippage seeds
  {0,1,2} for control and any qualifying winner.
- **Decision rule (fixed in advance)**: choose the SMALLEST band in
  {0.05, 0.08, 0.10} whose seed-mean vs control satisfies ALL of:
  1. val total_return ≥ control − 1.5 pts
  2. fees_paid ≤ 0.5 × control
  3. |max_drawdown| ≤ 1.1 × control
  If none qualifies → no deploy; design a stateful hysteresis band instead.
- **Burned-test veto** (descriptive only, selection stays on val): winner +
  control run ONCE on 2026-02-15..2026-07-10; if winner underperforms control
  by > 3 pts there, veto the deploy and take the hysteresis route.
- Deploy (if passed): loop reorder + band value + `min_trade_notional` 10→2000
  in one commit, then `/admin/reset` and a FRESH forward window (the running
  2026-07-18 window is closed early — superseded by an execution-layer fix; its
  partial result is recorded but triggers no model decision).

### Band A/B result (2026-08-07) + prospective amendment: notional-cap sweep

Result of the band sweep (seed 0, `--limits`, val): control 0.02 → −8.06%,
0.05 → −7.27%, 0.08 → −9.06%, 0.10 → −9.27%. **No band qualifies** (fee
criterion: best cut is 28%, rule demands ≥50%; returns statistically flat).
Bigger finding: live-parity execution of the CURRENT config scores −8.1% on
val while the raw policy scores +16.6% (net of its own fees) — the dominant
execution distortion is `max_trade_notional: 50000` itself (mean
desired-vs-executed gap 2.4/9.0/15.1 weight-pts per bar BTC/ETH/SOL;
policy swings 0→100%). The model was trained AND selected without any
notional cap; live imposes one → 24.7-pt execution gap on val.

**Amendment (pre-registered 2026-08-07 before results): notional-cap sweep.**
- Sweep `invest_limits.max_trade_notional` ∈ {50000 control, 250000, 500000}
  at band 0.02 (trained semantics), `--limits`, val window, recovery resume.
- Rule: pick the SMALLEST cap whose seed-{0,1,2} mean satisfies BOTH:
  1. val total_return ≥ raw-policy reference (+16.6%, same seeds) − 2 pts
  2. |max_drawdown| ≤ 1.1 × raw reference
  If none qualifies → keep 50000, no deploy, hysteresis/design route.
- Note 500000 never binds by construction (max_single_position 0.40 caps any
  leg at $400k on a $1M book) — it is "cap off" with a seatbelt.
- Band reorder (band BEFORE vol scale + limits, vs current weights) ships with
  any deploy regardless: it is freeze-proofing + training parity, not tuning.
- min_trade_notional 10→2000 ships too (dust hygiene, dominated by band).
- Burned-test veto unchanged: chosen config vs current-live config, once, on
  2026-02-15..2026-07-10; veto if chosen underperforms by > 3 pts.

### Notional-cap result + DECISION (2026-08-07) — judgment call, written down

Val (2025-10-01..2026-01-31, `--limits`, recovery, seeds {0,1,2} mean):

| max_trade_notional | return | sharpe | max_dd | fees |
|---|---|---|---|---|
| 50000 (control = live) | −8.07% | −1.00 | −12.7% | $9.1k |
| 250000 | **+11.31%** | 1.17 | −11.2% | $19.3k |
| 500000 (never binds) | +11.29% (seed 0) | 1.16 | −11.2% | $21.0k |

- **The pre-registered cap rule was infeasible by construction, not merely
  failed**: with max_single_position 0.40 kept, cap=500k never binds, so its
  +11.3% is the ceiling of the whole intervention class; the rule demanded
  ≥ +14.6% (raw − 2 pts). No cap value, including infinity, could ever pass —
  the raw reference includes 100% single-asset concentration the system
  forbids by design. Recording the misspecification and deciding on evidence.
- The remaining ~5.3-pt gap to raw (+16.6%) is the price of
  max_single_position 0.40 — a deliberate concentration limit, KEPT (loosening
  it now would be results-driven scope creep).
- The band rule was feasible and genuinely failed — bands stay at 0.02, closed.
- **Burned-test veto (one look, seed 0, 2026-02-15..2026-07-10)**: 250k −7.34%
  vs control −5.82% → 1.5-pt underperformance < 3-pt veto bar → no veto.
  Crash-window DD −18.3% vs −17.8%, one halt each (same date). Reported for
  the record; the crash-era pacing benefit of a small cap is real but small.
- **DECISION: deploy max_trade_notional 250000** (smallest cap within 0.05 pts
  of the unbinding ceiling) + band reorder + min_trade_notional 2000.
- Post-deploy expectation: the book will now take up to 40% single-asset
  positions — kill-switch halts become genuinely possible for the first time
  (the flat-book era could not trip them). Intended behavior, not regression.

## Forward paper evaluation #2 (v5.1 daily + fixed execution)

- **Window**: 2026-08-07 → 2026-11-05 (90 days). Baseline = the `/admin/reset`
  performed on 2026-08-07 after deploying the execution fix (band reorder,
  max_trade_notional 250k, min_trade_notional 2000).
- The 2026-07-18 window is CLOSED EARLY (20 days in): superseded by an
  execution-layer fix that changes the system under evaluation. Partial
  result for the record: agent −0.06% vs benchmark −0.76% (+0.7 pts), ~97%
  cash throughout, fees $954 vs price P&L +$353. Triggers no model decision.
- Metrics and judgment rules: identical to the 2026-07-18 window (see above).
- Expected live behavior change: real exposure (up to 40%/asset), higher
  fees in absolute terms, halts now reachable.

## Tested and REJECTED (do not re-enable without new evidence)

| feature | val result (2026-07-18, v5.1 ensemble) | verdict |
|---|---|---|
| action.target_smoothing 0.3 / 0.5 | +10.6% / +6.5% vs +16.5% plain | off (0.0) |
| ensemble.disagreement_scaling k=2 floor=.25 | +12.8% vs +16.5% plain | off; revisit with ≥3 members |

Both remain in code behind config flags for future A/Bs.

## v5.2 candidate plan — RESULT: rejected at val screen (2026-07-18)

- 3 PPO seeds (42/7/2026), configs/full.yaml with reward.weights.turnover
  (internalize churn — the live 2-day SOL round trip), rebuilt dataset.
- Outcome: turnover=0.002 ensemble −17.1% and turnover=0.001 ensemble −9.1% on
  val vs incumbent +16.6% on identical data → no gate touch, no deploy,
  incumbent v5.1 stays. Full numbers in models/MANIFEST.json (daily_v5.2).
- Val window has now been reused twice for selection; further candidate
  sweeps against this val window face growing selection-overfit risk — prefer
  a design change (features, action space, horizon) over more penalty tuning.
- Note: CoinMetrics dropped FeeMeanUSD from the community tier (2026-07). The
  fetcher degrades gracefully; `mean_fee_usd` is NaN→filled after 2026-04-19.
  Consider dropping it from `features.onchain` at the next retrain.
