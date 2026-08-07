# ppofolio

[![tests](https://github.com/mohammadi-hadi/ppofolio/actions/workflows/test.yml/badge.svg)](https://github.com/mohammadi-hadi/ppofolio/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/ppofolio)](https://pypi.org/project/ppofolio/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A PPO seed-ensemble that allocates a portfolio across BTC, ETH and SOL from technical,
on-chain and macro features. Classical deep RL (Stable-Baselines3, not LLM-based),
inspired by Meta-RL-Crypto ([arXiv:2509.09751](https://arxiv.org/abs/2509.09751)).
This is the public release of a private research project; the history starts at the
first public version.

What makes it different from the average "RL trading bot" repo is the evaluation
discipline, not the model:

- a **pre-registered evaluation protocol** ([EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)) written
  before results exist, with decision rules fixed in advance,
- **burned held-out test windows** — each is touched once per candidate generation,
  then never used for selection again,
- **honest baselines** (buy-and-hold, rebalanced B&H, MACD, risk parity, 200-day
  trend filter) evaluated on every window,
- a **model manifest** ([models/MANIFEST.json](models/MANIFEST.json)) recording every
  candidate's verdict — including the failures.

## Honest results

The current ensemble **failed its pre-registered held-out gate**: on the burned test
window it returned −12.4% against buy-and-hold's −8.25%. On the validation window it
returned +16.6% — but that is the selection window, so the number is selection-biased
by construction. Simulated under the deployed execution stack (concentration caps,
per-trade caps, volatility circuit breaker, kill switch) the validation figure is
+11.3%. The system is deployed for **forward paper evaluation only** and is not
qualified for real money by its own rules. The evaluation protocol exists precisely
so this paragraph cannot be written any other way.

## Architecture

```
data        ccxt OHLCV · CoinMetrics on-chain · GNews + LLM sentiment (optional) · fear/greed, dominance, funding
  └─ features    per-asset technical indicators + regime features -> one wide panel (ASSET__feature)
       └─ env         Gymnasium env: obs = lookback window + weights + cash, action = target weights
            └─ agents      PPO seed ensemble (Stable-Baselines3), averaged target weights
                 └─ safety chain   volatility circuit breaker -> kill switch (halt + recovery-gated resume) -> hard risk limits
                      └─ broker        paper broker (SQLite or Postgres), FastAPI service + dashboard
```

The safety-chain ordering (model → breaker → kill switch → limits → broker) is a
deliberate design constraint. The multi-objective reward mixes return, rolling
Sharpe, drawdown, liquidity and loss asymmetry.

## Install

Library (env, rewards, brokers, backtest metrics):

```bash
pip install ppofolio --extra-index-url https://download.pytorch.org/whl/cpu
```

Full research workflow (dataset building, training, backtesting, paper trading):

```bash
git clone https://github.com/mohammadi-hadi/ppofolio && cd ppofolio
make install          # pip install -e ".[dev]"
make dataset          # build the feature panel (public APIs, no keys required)
make train            # PPO training
make backtest         # walk-forward eval vs baselines -> reports/
make paper            # paper-trading loop
make serve            # FastAPI service + dashboard
```

`scripts/train_ensemble.sh` trains the 3-seed ensemble; `scripts/eval_system.py`
evaluates the full system (policy + circuit breaker + kill switch + risk limits)
over an arbitrary window, which is what the protocol gates on.

## Layout

| path | what |
|---|---|
| `ppofolio/data` | fetchers with cache-on-disk (`load_or_fetch`) + dataset assembly |
| `ppofolio/features` | technical indicators, regime features, panel assembly |
| `ppofolio/env` | Gymnasium trading environment |
| `ppofolio/rewards` | weighted multi-objective reward |
| `ppofolio/agents` | training entry points + seed ensemble |
| `ppofolio/backtest` | walk-forward evaluation + rule-based baselines |
| `ppofolio/live` | decision loop, circuit breaker, kill switch, risk limits, brokers |
| `ppofolio/serve` | FastAPI service, scheduler, dashboard |
| `configs/` | daily / hourly / smoke configs (`--config` on every entry point) |
| `EVAL_PROTOCOL.md` | pre-registered evaluation rules + change log |

## Caveats

- News/LLM sentiment is off by default: the shipped configuration was trained with
  sentiment zeroed, so enabling it at inference is distribution shift. Re-enable only
  together with a retrain.
- SOL on-chain data is patchy before 2021 (CoinMetrics community tier); the pipeline
  forward-fills then zero-fills.
- Trained model weights are not distributed; train your own with
  `scripts/train_ensemble.sh`.
- Training splits start 2021-08 because 180-day returns and the 200-day MA need
  warmup.

## Disclaimer

Research software for studying RL evaluation methodology on market data. Nothing here
is financial advice, and the honest-results section above should make clear that
using it to trade real money would be a bad idea.

## Cite

```bibtex
@software{mohammadi_ppofolio,
  author  = {Mohammadi, Hadi},
  title   = {ppofolio: PPO ensemble for multi-asset crypto portfolio allocation},
  url     = {https://github.com/mohammadi-hadi/ppofolio},
  version = {0.1.0},
  year    = {2026}
}
```

See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Hadi Mohammadi
