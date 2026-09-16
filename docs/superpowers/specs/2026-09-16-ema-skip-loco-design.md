# Round 11: stacked relative-EMA skips, validated leave-one-climate-out

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## The range test that decides the design

Rounds 8 and 10 both failed for one reason: what they injected into the
network was outside the training range on an unseen climate. Measured on the
dev split (train -10/10 C, score 25 C), share of test values outside the
training range:

| Injected feature | Outside training range |
|---|---|
| raw last day | 26.6% (battery temperature 100%) |
| EMA of values, half-life 2 / 7 / 30 d | 26.4% / 24.9% / 25.8% |
| **EMA - last day** | **0.1 - 0.2%** |
| **EMA - anchor** | **0.1 - 0.2%** |
| **fast EMA - slow EMA (trend)** | **0.4%** |

An EMA is a convex mix of that battery's OWN days, so it stays inside the
battery's history but not inside what training saw -- on a hotter climate
every one of that battery's days is hotter than any training day. Differences
transfer; levels do not.

## Model (`experiment/ema.py`, RSSM flag `--ema-skip`)

Per predicted day `t`, with `anchor_t` the round 7 same-day-type anchor:

    e_h      = EMA of the window's observables at half-life h, h in {2, 7, 30}
    skip_t   = [ e_2 - anchor_t, e_7 - anchor_t, e_30 - anchor_t, e_2 - e_30 ]
    x_l      = ELU(W_l x_{l-1} + P_l skip_t)        decoder hidden layers
    pred_t   = anchor_t + head(x_L)                 head zero-initialised

So the decoder sees *how this battery's recent history sits relative to the
day being copied*, and its trend, never an absolute level. The anchor stays
where round 7 puts it: added at the output, through no nonlinearity.

The EMA state is computed from the OBSERVED window and frozen for the
rollout; it is not updated with the model's own predictions, which would
reintroduce drift. During filtering reconstruction the per-day EMA uses days
strictly before `t` (day 0 falls back to itself), matching `window_anchors`.

Zero-initialised head keeps round 7 as the starting point.

## Protocol: leave-one-climate-out

Chosen in conversation over another pass at the 25 C dev split, which has now
selected among thirteen variants and is spent as an estimator.

Four folds on the round 6 noisy dataset: hold out -10, 10, 25, 42 C in turn,
train on the other three (`windows --holdout-ambient`, validation split by
battery from the training climates). Two seeds each, two models:

| Model | Flags |
|---|---|
| round 7 reference | `--anchored` |
| round 11 | `--anchored --ema-skip` |

16 training runs. Reported per fold and averaged, factual and what-if
tolerance accuracy, against the same-day-type baseline computed per fold.
This also fixes rounds 5-10's one-seed caveat for the reference design, and
tests colder extrapolation (-10 C held out), not only hotter.

Cost, measured honestly: counterfactual training targets are ~35 min per fold
(6,000 windows, kept at 6,000 so the numbers stay comparable with rounds
7-10), plus ~9 min per fold for the test set's what-if ground truth, plus 16
trainings. Total ~4-5 hours, run stage by stage because two memory-heavy jobs
at once killed round 7's first attempt.

## Decided before running

- Success = beating BOTH the round 7 reference and the per-fold baseline on
  the average across folds, not on a single fold.
- Per-fold numbers are reported even where they disagree with the average.
- No new variant is added on the strength of these folds; if round 11 loses,
  round 7 stands and the search stops.
