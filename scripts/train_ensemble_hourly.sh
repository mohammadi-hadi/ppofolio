#!/usr/bin/env bash
# v6 hourly+shorts ensemble: 3 PPO seeds. Writes to models/candidate/ — the
# incumbent models/policy_h*.zip stay live until the new ensemble passes the gate.
# CFG must point at the config with the chosen turnover weight.
set -e
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CFG=${CFG:-configs/hourly.yaml}
STEPS=${STEPS:-4000000}
OUT=models/candidate
mkdir -p $OUT

for SEED in 42 7 2026; do
  echo "===== PPO hourly seed=$SEED ($STEPS steps) ====="
  caffeinate -i $PY -m ppofolio.agents.train_one --config $CFG --algo ppo --seed $SEED \
    --out $OUT/policy_h$SEED.zip --timesteps $STEPS
done

echo "===== hourly ensemble done ====="
ls -lh $OUT/policy_h*.zip
