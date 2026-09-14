#!/usr/bin/env python3
"""Curriculum fix: precompute a RANDOM counterfactual continuation per train window.

The corrected Step 3 (`experiment/abduction.py`) found the world model was
not abducting the hidden `crystal` state -- it was conditioning on cheap,
currently-observable battery state instead. The most likely reason (see the
experiment report's "what I would do differently"): the model NEVER saw
anything but the FACTUAL action continuation during training
(`experiment/train_world_model.py`). Under factual-only training, "what
state resulted" and "what happens next" are always correlated with the
actual schedule, so nothing forces the latent to keep information that is
only useful under a schedule that didn't actually happen. This script is
the fix's data half: for every window in `windows_train.npz`, generate ONE
randomized counterfactual action sequence (`experiment.abduction`'s
`random_perturbation_day_types` -- each day's is_layup independently flipped
with probability 0.5, a different perturbation SHAPE from Step 3's
deterministic full flip, so training augmentation is never literally
"train on the eval transformation") and rerun the SAME physics ODE
(`calendar_sim.roll_days`) from the window's true end-of-day hidden state to
get the exact ground-truth continuation under that counterfactual schedule
-- the same generator-owned trick Step 3 uses for evaluation, now used to
manufacture a training signal.

`experiment/train_world_model.py` then trains the SAME encoded latent `z`
to correctly roll out under BOTH the factual continuation AND this
counterfactual one. If a single `z` has to explain two divergent futures
from the same true history, memorizing "what a factual continuation from
this observable state usually looks like" stops being sufficient --
generalizing across action sequences becomes the cheapest remaining
strategy, which is exactly the pressure that was missing.

Held-out test scenarios are untouched: this script only ever reads
`windows_train.npz` and only ever writes `windows_train_cf.npz`. The
counterfactual generalization test in Step 3 remains meaningful because the
model never saw a counterfactual action sequence -- random or full-flip --
for any TEST scenario, in training or otherwise.

Run: `python3 -m experiment.build_counterfactual_targets --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cell.aging import AgingRates
from experiment.abduction import (
    actions_from_day_types,
    ground_truth_counterfactual,
    random_perturbation_day_types,
)
from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.data import load_daily_trajectory
from sweep_dataset import load_driving

SEED = 11


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--trajectory", default="runs/telemetry.csv")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    windows_meta = json.loads((base / "windows_meta.json").read_text())
    k = windows_meta["rollout_k"]

    scenarios_list = json.loads((base / "scenarios.json").read_text())
    scenario_meta_by_name = {s["scenario"]: s for s in scenarios_list}
    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")
    driving_raw, _dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()

    train = dict(np.load(base / "windows_train.npz", allow_pickle=True))
    n = train["actions"].shape[0]
    n_action, n_obs = len(ACTION_FEATURES), len(OBSERVABLE_FEATURES)

    rng = np.random.default_rng(SEED)
    cf_actions = np.zeros((n, k, n_action), dtype=np.float32)
    cf_observables = np.zeros((n, k, n_obs), dtype=np.float32)

    for idx in range(n):
        name = str(train["scenario"][idx])
        end = int(train["end_day"][idx])
        arr = scenarios[name]
        meta = scenario_meta_by_name[name]

        day_types = random_perturbation_day_types(arr, end, k, rng)
        cf_actions[idx] = actions_from_day_types(arr, end, k, day_types)
        cf_observables[idx] = ground_truth_counterfactual(
            meta, driving_raw, arr, end, day_types, rates,
        )

        if (idx + 1) % 500 == 0 or idx == n - 1:
            print(f"[{idx + 1}/{n}] built counterfactual targets", flush=True)

    out = base / "windows_train_cf.npz"
    np.savez(
        out, cf_actions=cf_actions, cf_observables=cf_observables,
        scenario=train["scenario"], end_day=train["end_day"], seed=SEED,
    )
    print(f"\nwrote {out} ({n} examples, {k}-day randomized counterfactual continuations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
