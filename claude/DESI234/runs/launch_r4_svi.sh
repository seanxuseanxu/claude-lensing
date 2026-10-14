#!/bin/bash
# DESI234 round 4, attempt 2: re-run SVI + sampling, resuming the MAP already on disk.
#
# WHY. Attempt 1 (launch_r4.sh) got through MAP -- chi2_nu 1.287, best log-posterior
# 1.897e+04, map.npz saved -- and then died in the SVI covariance guard at condition
# 9.958e+12 against the 1e10 limit (svi_health.py COND_MAX). That threshold is calibrated
# by measurement across stored runs: healthy runs sit at 9.4e5-5.9e8 and the one diverged
# run at 8.8e11. 9.96e12 is in the diverged population, so the guard is right and is NOT
# raised here.
#
# THE CAUSE, and why the fix is not the one the error message suggests. The guard advises
# lowering --svi-peak-lr. But r3_fg passed this same guard comfortably (condition 2.397e+07,
# min eigenvalue 3.312e-09) at the SAME --svi-peak-lr 1e-3 and the SAME --svi-steps 12000.
# The variable that actually changed between r3_fg and attempt 1 is --n-vi, 1024 -> 512,
# which attempt 1 lowered only because these 40 GB A100s could not hold MAP's 1024-sample
# batch. A smaller SVI batch is a noisier ELBO gradient, which is exactly what degrades the
# fitted covariance's conditioning. Attempt 1 also shows the symptom directly: its ELBO sd
# over the last 300 steps was 50.7.
#
# So: restore --n-vi 1024 and change nothing else. Measured before relaunch, on a scratch
# copy rather than this run directory: SVI at --n-vi 1024 runs on this allocation without
# OOM. SVI's buffer is far smaller than MAP's, so the 512 that MAP needs is not a constraint
# the SVI stage shares -- the two batch sizes were never required to match.
#
# MAP IS NOT RE-RUN. --stages svi,hmc resumes from the existing map.npz (823.1 s already
# paid). run_fit.merge_prior_metadata carries the MAP stage's recorded fields forward, so
# metadata.json still describes the MAP that was actually fitted -- KNOWN_ISSUES.md item 1.
# Note that attempt 1 crashed before writing metadata.json at all, so this invocation
# writes it fresh; the MAP fields it records are this command's own --map-steps/--n-samples
# values, which are deliberately kept identical to attempt 1's so that record stays true.
#
# Everything else is held at attempt 1's value, which is to say at r3_fg's, except the two
# round-4 changes themselves: --mass-prior-version 2 (the q >= 0.2 ellipticity bound) and
# the longer sampling (--burnin 25000 --num-results 4000) that RESULTS_v3.md section 5.4's
# ridge diagnosis calls for. The guard runs at its default 1e10.
set -u
cd /global/u1/s/seanjx/claude-lensing
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
RUNS=claude/DESI234/runs

echo "[launch] $(date -Is) r4_fg_q svi,hmc on GPU 0,1,2,3"
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $PY claude/pipeline/run_fit.py \
    --lens DESI234 --n-source 2 --n-lens-light 2 \
    --foreground --mass-prior-version 2 \
    --stages svi,hmc \
    --map-steps 8000 --n-samples 512 \
    --n-vi 1024 --svi-steps 12000 --svi-peak-lr 1e-3 \
    --sampler mclmc --n-hmc 50 --burnin 25000 --num-results 4000 \
    --seed 0 --label r4_fg_q \
  >> $RUNS/r4_fg_q.log 2>&1
RC=$?
echo "[done] $(date -Is) r4_fg_q exit=$RC"
exit $RC
