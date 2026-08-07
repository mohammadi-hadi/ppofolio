#!/usr/bin/env bash
# v5 daily ensemble: 3 PPO seeds. Writes to models/candidate/ — the incumbent
# models/policy_s*.zip stay live until the new ensemble passes the deploy gate.
set -e
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CFG=${CFG:-configs/full.yaml}
STEPS=${STEPS:-800000}
OUT=models/candidate
mkdir -p $OUT

for SEED in 42 7 2026; do
  echo "===== PPO daily seed=$SEED ($STEPS steps) ====="
  caffeinate -i $PY -m ppofolio.agents.train_one --config $CFG --algo ppo --seed $SEED \
    --out $OUT/policy_s$SEED.zip --timesteps $STEPS
done

echo "===== daily ensemble done ====="
ls -lh $OUT/policy_s*.zip
