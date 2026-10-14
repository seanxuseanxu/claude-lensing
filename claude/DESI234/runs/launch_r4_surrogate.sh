#!/bin/bash
# DESI234 round 4, attempt 3: the same model, with a better surrogate.
#
# WHAT ATTEMPT 2 GAVE. r4_fg_q ran to completion and did NOT converge: 46/46 parameters
# above the 1.01 gate, worst R-hat 2.125, and the Vehtari ESS floor FAILED (min bulk-ESS 63
# against the 400 total / 50-per-split-half floor). Below that floor R-hat is uninterpretable
# in both directions, so nothing in that run is quotable.
#
# WHY THIS RUN CHANGES THE SURROGATE AND NOT THE SAMPLING. guide 04 sec. 4.3 step 3 ranks
# "get a better surrogate" second and notes it is where the leverage is at high dimension
# and that instinct ranks it too low. Both of its symptoms are present and measured:
#
#   - MAP finished 193.750 nats BELOW the best scored HMC draw (18973.945 vs 19167.695).
#     diagnose.py's own verdict: MAP did not find the mode, so SVI (seeded from MAP) and
#     MCLMC (seeded from SVI) both inherited a bad start.
#   - MAP's chisq tail slope over its last 800 of 8000 steps was -0.00002/step: STILL
#     DESCENDING when it stopped. It ran out of steps, it did not converge.
#   - ELBO tail sd 52.3. DESI238 converged at 0.1; the guide's worked example took 36
#     parameters from R-hat 1.338 to 1.003 on surrogate work alone.
#
# The SVI covariance itself is fine (condition 2.639e+07, inside the healthy 1e6-6e8 band),
# so this is not the step-0 failure; it is an under-converged optimum, not a broken one.
#
# WHAT IS NOT THE FIX, and why a longer sampling run is not being bought. The ESS-vs-draws
# slope is 0.08 -- SUBLINEAR. diagnose.py's verdict is explicit: more draws will not fix
# this, go to the model or the surrogate. Attempt 2 already spent 25000 burnin and 4000
# draws (5x and 5.3x r3_fg) and that is what produced the ESS floor failure, so a sixth
# multiple of the same thing would be spending the allocation on a lever already measured
# not to move. --burnin and --num-results are therefore held at attempt 2's values.
#
# WHAT CHANGES, and the measured reason for each:
#   --map-steps  8000 -> 24000   the tail slope says MAP was still descending at 8000
#   --n-samples   512 ->   640   measured ceiling on these 40 GB cards is between 640 and
#                                768 (768 dies asking 34.19 GiB; 640 runs), so this takes
#                                the largest batch that fits rather than the safe one
#   --svi-steps 12000 -> 24000   ELBO tail sd 52.3 is far from settled
#   --n-vi       1024 (unchanged -- already measured to fit, and the value r3_fg passed at)
#
# WHAT THIS RUN IS EXPECTED TO SETTLE, stated before launch so the result cannot be read
# generously after it. It tests ONLY whether attempt 2's non-convergence was surrogate
# quality. It is NOT expected to resolve the physical problem recorded in RESULTS_v4.md:
# attempt 2's deflector sits at q = 0.295 with e1 pinned against the v2 bound, and that
# pinning holds just as hard among the top 10% of draws by log posterior (96.0%) as over
# all draws (96.3%) -- so the likelihood genuinely prefers the excluded region and a better
# starting point will not move it off the wall. If this run converges, it converges on a
# deflector that is still too flat to quote. That is a modelling question (step 3 item 1),
# and it is the next thing to work, not this.
set -u
cd /global/u1/s/seanjx/claude-lensing
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
RUNS=claude/DESI234/runs

echo "[launch] $(date -Is) r4_fg_q2 on GPU 0,1,2,3"
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $PY claude/pipeline/run_fit.py \
    --lens DESI234 --n-source 2 --n-lens-light 2 \
    --foreground --mass-prior-version 2 \
    --map-steps 24000 --n-samples 640 \
    --n-vi 1024 --svi-steps 24000 --svi-peak-lr 1e-3 \
    --sampler mclmc --n-hmc 50 --burnin 25000 --num-results 4000 \
    --seed 0 --label r4_fg_q2 \
  > $RUNS/r4_fg_q2.log 2>&1
RC=$?
echo "[done] $(date -Is) r4_fg_q2 exit=$RC"
exit $RC
