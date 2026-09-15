# Round 9: a 48 C climate, hotter than anything trained on

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## Why

Asked for in conversation: push the unseen climate from 42 C to 48 C.
42 C is now inside the training set's natural range for a deployed model;
48 C tests generalization past the hottest condition the models have seen.

## Feasibility, checked before building

- `load/spec.py` allows ambient -40 to 60 C.
- One battery simulated 60 days at 42 C and 48 C (same settings):

| | 42 C | 48 C |
|---|---|---|
| battery mean / peak | 43.3 / 47.9 C | 49.2 / 53.2 C |
| bay mean / peak | 51.3 / 94.2 C | 56.2 / 94.9 C |
| corrosion-hours at day 60 | 4,405 | 6,707 (+52%) |

**Limitation:** peak bay temperature barely rises, because the bay model is
bounded by the recorded drive's coolant and oil temperatures
(`load/thermal.py`, hot anchor), and a single recorded engine does not run
hotter on a hotter simulated day. Peak bay temperature at 48 C is likely
understated.

## Data

`experiment/add_ambient.py` appends 20 batteries at 48 C (indices 80-99) to
the 80 existing ones. Their parameters continue the SAME seeded draw sequence
(`SAMPLING_SEED`, same order via a shared `draw_scenario_params`), so the
existing 80 scenarios are bit-identical and the new 20 are what
`build_dataset.py` would have produced with a fifth ambient appended. Rows are
simulated with `run_calendar_scenario`, exactly as before.

Then, in `runs/experiment_48/`:
1. sensor noise (`make_noisy_variant`, seed 3; the first 80 scenarios get
   the same noise as round 6 because the generator walks scenarios in order),
2. `windows --holdout-ambient 48`: test = all 48 C batteries, train/val = the
   -10/10/25/42 C batteries,
3. counterfactual training targets, then cached what-if ground truth for the
   test set -- run one at a time (round 7 was killed for memory when these
   overlapped with training).

## Model and protocol

No new design choices. Round 7's selected model (RSSM `--anchored`,
absolute temperatures) and gru_vae `--anchored` for the head-to-head, same
hyperparameters, best-epoch on validation. The 48 C test set is scored once.
Reported against the same-day-type baseline and round 7's 42 C result.
