# Round 7: make the world models generalize to an unseen climate

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## Where round 6 left it

On the held-out 42 C climate, with sensor noise, both models scored ~30%
tolerance accuracy (within +/-0.05 V, +/-0.5 A, +/-1 C, +/-2 SoC points) on the
30-day factual forecast, against ~79-85% on seen climates. Their errors are a
bias toward the training climates (gru_vae forecast 37.1 C where truth was
43.4 C, after seeing 43.4 C on the window's last day).

Two diagnostics, run after round 6 on the 42 C test set (disclosed; no model
was fit to them):

| Baseline, no learning | 42 C tolerance accuracy |
|---|---|
| Persistence: every future day = last window day | 68.6% |
| Same-day-type: every future day = the most recent window day of the same type (drive / layup) | **83.3%** |
| gru_vae / RSSM (round 6) | 29.9% / 31.9% |

The models are far worse than copying. And temperatures expressed as offset
above ambient at 42 C (battery mean +1.4 C median, bay mean +8.8 C) fall
inside the range of the colder climates, where absolute temperatures do not.

## Changes (both models, identical)

### A. Anchored residual prediction

Every predicted day `t` is `anchor_t + learned correction`. `anchor_t` is the
same-day-type baseline: the most recent observed window day whose `is_layup`
matches the action on day `t` (falling back to the window's last day if the
window has no day of that type). It is computable at prediction time from
the observed window and the action sequence being asked about, so it uses
no future observation, and under a counterfactual schedule the anchor
follows the counterfactual day types.

- gru_vae: `rollout` output = anchor + decoder output; that sum is what feeds
  back as `prev_obs`.
- RSSM: imagined decode = anchor + decoder output. Window reconstruction uses
  an in-window anchor: the most recent EARLIER window day of the same type
  (day 0 anchors on itself), so filtering and imagination decode the same
  quantity.

A model that learns nothing reproduces the 83% baseline; training can only
add a correction. A zero-initialised output layer makes that the starting
point exactly.

### B. Temperature as offset above ambient

`t_bat_mean, t_bat_max, t_bay_mean, t_bay_max` are transformed to
`value - ambient_c` (ambient is an action input, known for every day) before
standardisation, for inputs and targets alike, and transformed back before
any real-unit evaluation. Voltage, current and SoC are untouched.

## Protocol: no tuning on the 42 C test

- **Development:** train on -10 and 10 C batteries (train windows filtered by
  scenario; the precomputed counterfactual targets stay row-aligned), select
  the best epoch on -10/10 validation batteries, and use every 25 C window as
  the stand-in unseen climate. All design choices (A, B, A+B, neither) are
  compared here.
- **Final:** the chosen design is trained on -10/10/25 C and run on the 42 C
  test set exactly once. Its number is reported whether or not it improves.
- The same-day-type baseline is reported as a baseline in every table from
  now on; a model that does not beat it is not claimed as useful.

## Success criteria (decided before running)

- Dev (25 C): the chosen design beats the same-day-type baseline's tolerance
  accuracy on the factual forecast.
- Final (42 C): report factual and counterfactual tolerance accuracy,
  real-unit MAE, abduction head-to-head and calibration against rounds 6 and
  the baseline. No minimum; the number is the result.
