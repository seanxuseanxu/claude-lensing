#!/bin/bash
# DESI234 round 4: refit the foreground model with the deflector ellipticity bounded.
#
# The question (README.md, "Round 4"): does a foreground mass component survive once the
# deflector is required to be physically round? r3_fg answered "yes, theta_E = 0.370"" but
# did so on a 7:1 needle (q = 0.143, gamma = 2.951) and did not converge, so its answer
# measures nothing. This run changes exactly two things against r3_fg.
#
# 1. THE MODEL: --mass-prior-version 2. The host EPL's e1/e2 become TruncatedNormal on
#    +/-0.4714, the inscribed square of gigalens' own soft q >= 0.2 plausibility floor
#    (priors.py MASS_V2_ELLIP_ABS_MAX; gigalens.physicality.axis_ratio_constraint).
#    Verified before launch at ellip_sigma = 0.15: the box removes 3.359e-3 of the prior
#    and the true q >= 0.2 disc only 5.150e-5, reproducing priors.py's stated figures; and
#    20,000 prior draws give min q = 0.2445, none below the floor, against v1's 79.8% of
#    draws below it. r3_fg's c = 0.7498 is outside the bound; round 2's q = 0.900
#    (c = 0.053) and r3_unmasked's q = 0.943 (c = 0.029) are nowhere near it. The bound
#    excludes the pathology and almost nothing else. It is NOT an assertion that the
#    deflector is round.
#
# 2. THE SAMPLING: burnin 5000 -> 25000, num_results 750 -> 4000. RESULTS_v3.md section 5.4
#    diagnosed r3_fg (unlike r3_unmasked) as NOT frozen but under-mixed along a ridge --
#    R = 1.59 on theta_E, chains moving about as much as they are apart, clustering buying
#    almost nothing (max R-hat 1.92 -> 1.63 at k=3) -- and read the fix as chain length, not
#    another component. This buys that length. It is nearly free: round 3's MCLMC stage took
#    227 s against MAP's 1676 s and SVI's 3998 s, so 5x the burnin and 5.3x the draws costs
#    about 20 minutes on a run whose first two stages take 1.5 hours.
#
# WHAT ELSE CHANGED, AND WHY IT IS NOT A FREE VARIABLE: --n-samples 1024 -> 512.
# Round 3 ran on 80 GB A100s; this allocation has 40 GB cards. Measured here before launch,
# not guessed: 1024 asks for 46.37 GiB and dies RESOURCE_EXHAUSTED, 768 asks 34.19 GiB and
# also dies, 640 and 512 both run. That is ~46 MiB per sample, consistent with round 3's
# measured 43.6 MiB at ss=4. 512 is taken over 640 for headroom, since SVI allocates its own
# buffer. --map-steps rises 5000 -> 8000 to pay for the smaller batch, which is the same
# trade round 3 made when ss=2 -> 4 quartered its own batch.
#
# Everything else is held at r3_fg's value so the comparison is to that run: 2 source
# components, 2 lens-light components, unmasked (MASK_CIRCLES = ()), supersample = 4, MCLMC,
# 50 chains, seed 0, --svi-steps 12000 --svi-peak-lr 1e-3.
#
# Smoke-tested before launch at --map-steps 20 --svi-steps 30 --burnin 50: MAP -> SVI ->
# MCLMC ran to completion at 46 parameters under v2. (That smoke test needed
# --svi-max-cond raised; a 30-step SVI has had no chance to converge. This run does NOT
# raise it -- the guard runs at its default 1e12 here.)
set -u
cd /global/u1/s/seanjx/claude-lensing
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
RUNS=claude/DESI234/runs

echo "[launch] $(date -Is) r4_fg_q on GPU 0,1,2,3"
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $PY claude/pipeline/run_fit.py \
    --lens DESI234 --n-source 2 --n-lens-light 2 \
    --foreground --mass-prior-version 2 \
    --map-steps 8000 --n-samples 512 \
    --n-vi 512 --svi-steps 12000 --svi-peak-lr 1e-3 \
    --sampler mclmc --n-hmc 50 --burnin 25000 --num-results 4000 \
    --seed 0 --label r4_fg_q \
  > $RUNS/r4_fg_q.log 2>&1
RC=$?
echo "[done] $(date -Is) r4_fg_q exit=$RC"
exit $RC
