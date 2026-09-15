# Round 6: sensor noise, a held-out hot climate, and best-checkpoint training

**Date:** 2026-09-15
**Status:** agreed in conversation, being implemented

## Why

Round 5 found no leakage and no overfitting (train and test error match; a
nearest-neighbour copy of the closest training window is 77x worse than the
models; a same-battery day-type lookup is 47x worse). The accuracy is still
inflated, for three reasons that make the test easier than a real car:

1. The daily observables are exact ODE output. `cell/sensors.py` exists and
   was never applied here.
2. Test batteries come from the same four climates and the same parameter
   ranges as training: interpolation, not unseen conditions.
3. RSSM training went unstable from epoch 72 and the evaluated checkpoint
   was the last epoch, not the best.

A fourth issue -- SoC and bay temperature are fed to the models although
`cell/sensors.py` classes them as unmeasurable -- is out of scope for this
round and stated as a limitation.

## Changes

### Noise (`experiment/noise.py`)

`cell/sensors.py` sigmas, applied once per (scenario, day) to the daily
aggregates of MEASURABLE channels only:

| Daily column | Sensor channel | sigma |
|---|---|---|
| v_mean, v_min, v_max | v_bat_v | 0.010 V |
| i_mean, i_mean_abs | i_bat_a | max(0.1 A, 0.5% of reading); i_mean_abs clamped >= 0 |
| t_bat_mean, t_bat_max | t_bat_c | 0.5 C |
| t_bay_mean, t_bay_max, soc | latent | none |

Each daily column gets independent noise. This treats a daily aggregate's
error as one draw at the sensor's per-sample sigma: conservative for means
(real averaging over a day would shrink white noise), roughly right for
calibration offset, which does not average out. Stated as a simplification.

Noise is written into a new dataset directory's `daily_trajectory.csv` by
`experiment/make_noisy_variant.py`; every hidden and resume field stays
clean, so the abduction test's ODE restarts are unaffected.

What models see and train on is noisy. Ground truth for the abduction test's
counterfactuals comes from the ODE and is clean, and the training
counterfactual targets are also clean ODE output. Factual truth in Gate 2 and
calibration is the noisy logged value, which is what a real deployment would
score against.

### Held-out climate (`experiment/windows.py --holdout-ambient 42`)

- **test:** every 42 C scenario (20 batteries), never seen in training.
- **train / val:** the -10, 10 and 25 C scenarios, split by battery 85/15.
- 42 C is outside the training range (max 25 C) and is the condition
  CLAUDE.md names as the primary ageing driver.

### Best checkpoint (both training scripts)

When `windows_val.npz` exists, each epoch computes the validation 30-day
factual rollout MSE, and the best epoch's weights are the ones saved.
Identical rule for gru_vae and RSSM. Disclosed as a change made after seeing
round 5's test results.

## Evaluation

Same five steps for both models into `runs/experiment_hot_noisy/`:
Gate 2, Gate 2b, abduction (vs last-day control), SAE probe, calibration,
plus real-unit accuracy against both noisy and clean truth.

## Decided before running

- Report round 6 regardless of direction, alongside rounds 4 and 5.
- A drop in accuracy on hot, noisy data is the expected, honest outcome, not
  a failure to fix.
- No hyperparameter changes between here and the test-set run.
