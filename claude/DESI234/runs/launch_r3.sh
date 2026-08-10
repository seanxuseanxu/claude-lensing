#!/bin/bash
# DESI234 round 3: the two fits, launched as a pair (RESULTS_v3.md section 4b).
#
# Both are 2-source / 2-lens-light, unmasked, supersample=4, MCLMC, 50 chains, burnin 5000.
# They differ in exactly one thing: --foreground (36 params vs 46).
#
# Each fit gets TWO GPUs. That symmetry is deliberate: KNOWN_ISSUES.md item 2 records device
# count as the unattributed second variable in DESI257's frozen-chain failure, so the two
# round-3 fits must not differ in it. r3_fg therefore WAITS for GPU 3 rather than starting
# early on one device.
set -u
cd /global/u1/s/seanjx/claude-lensing
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
RUNS=claude/DESI234/runs

# Batch sizes are set by supersample=4's memory cost, MEASURED not guessed. The MAP/SVI
# batch buffer is allocated PER DEVICE (it does not shard across the two), so the ceiling is
# ~61 GiB -- one 80 GB A100 after JAX's 75% preallocation. At ss=4 the cost is 43.6 MiB per
# sample (178 GiB at 4096 and 67 GiB at 1536 both scale to that figure, and both OOM'd).
# 1024 x 43.6 MiB = 43.6 GiB, which is the SAME footprint round 2 ran at (4096 at ss=2 =
# 44.6 GiB) -- i.e. ss=4 buys its accuracy by quartering the batch at fixed memory, and
# --n-vi comes down from round 2's 2048 for the same reason. Smoke-tested map,svi on the
# 46-parameter model before launch. --map-steps 5000 (up from round 2's 1500) is what pays
# for the smaller batch; section 2's 86.5-nat deficit is the target it has to clear.
COMMON="--lens DESI234 --n-source 2 --n-lens-light 2 \
  --map-steps 5000 --n-samples 1024 \
  --n-vi 1024 --svi-steps 12000 --svi-peak-lr 1e-3 \
  --sampler mclmc --n-hmc 50 --burnin 5000"

echo "[launch] $(date -Is) r3_unmasked on GPU 0,1"
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0,1 \
  $PY claude/pipeline/run_fit.py $COMMON --label r3_unmasked \
  > $RUNS/r3_unmasked.log 2>&1 &
PID_UNMASKED=$!

sleep 90  # let r3_unmasked claim GPUs 0,1 before the second process probes the bus
echo "[launch] $(date -Is) r3_fg on GPU 2,3"

PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=2,3 \
  $PY claude/pipeline/run_fit.py $COMMON --label r3_fg --foreground \
  > $RUNS/r3_fg.log 2>&1 &
PID_FG=$!

wait $PID_UNMASKED; RC_UNMASKED=$?
wait $PID_FG;       RC_FG=$?
echo "[done] $(date -Is) r3_unmasked exit=$RC_UNMASKED  r3_fg exit=$RC_FG"
exit $(( RC_UNMASKED != 0 || RC_FG != 0 ))
