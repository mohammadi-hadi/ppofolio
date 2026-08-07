PY ?= python
START ?= 2020-01-01
END ?= 2025-12-31
CONFIG ?= configs/base.yaml

.PHONY: install dataset train backtest paper serve test clean

install:
	$(PY) -m pip install -e ".[dev]"

dataset:
	$(PY) -m ppofolio.data.build_dataset --start $(START) --end $(END) --config $(CONFIG)

train:
	$(PY) -m ppofolio.agents.train_ppo --config $(CONFIG)

backtest:
	$(PY) -m ppofolio.backtest.run_backtest --config $(CONFIG)

paper:
	$(PY) -m ppofolio.live.loop --mode paper --config $(CONFIG)

serve:
	uvicorn ppofolio.serve.app:app --host 0.0.0.0 --port $${PORT:-8000}

test:
	pytest -q

clean:
	rm -rf data/processed/*.parquet reports/*.html
