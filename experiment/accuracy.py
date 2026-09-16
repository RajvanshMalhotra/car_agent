#!/usr/bin/env python3
"""Real-unit accuracy on the test set, for one model, against the same-day-type baseline.

Two numbers people can read: mean absolute error in volts, amps, degrees and
SoC points, and **tolerance accuracy** -- the share of (window, day) forecasts
within `TOL` of the truth. Scored against CLEAN truth: for a noisy dataset
pass `--clean-trajectory` (the pre-noise `daily_trajectory.csv`); the model's
inputs stay whatever the dataset holds.

- factual: the real 30-day continuation.
- `--cf`: every day's `is_layup` flipped (`experiment.abduction`'s eval
  perturbation), truth from rerunning the ODE from the true hidden state. That
  truth depends only on the dataset, so it is computed once and cached as
  `cf_truth_test.npz` beside the windows.

The representation (temperature offsets, anchored residuals) is read from the
checkpoint and applied through `experiment/pipeline.py`, the same path the
training scripts use. The same-day-type baseline -- each future day equals the
latest window day of the same type -- is scored alongside on identical inputs;
a model below it is not doing useful work.

Run: `python3 -m experiment.accuracy --dataset-dir runs/r7_dev/offsets_anchored --arch rssm
      --clean-trajectory runs/experiment/daily_trajectory.csv [--cf]`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiment.calendar_sim import OBSERVABLE_FEATURES
from experiment.checkpoints import ARCHS, artifact, load_model, rollout_kwargs
from experiment.data import load_daily_trajectory
from experiment.pipeline import prepare, real_units, standardize_prepared
from experiment.representation import from_offsets

#: "Accurate" = within these bands, in real units. SoC is a 0-1 fraction: 0.02 = 2 points.
TOL = {
    "v_mean": 0.05, "v_min": 0.05, "v_max": 0.05,
    "i_mean": 0.5, "i_mean_abs": 0.5,
    "t_bat_mean": 1.0, "t_bat_max": 1.0, "t_bay_mean": 1.0, "t_bay_max": 1.0,
    "soc": 0.02,
}
_TOL = np.array([TOL[f] for f in OBSERVABLE_FEATURES], dtype=np.float32)


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Per-feature MAE and tolerance accuracy over `[N, K, F]`, plus the overall mean."""
    err = np.abs(pred - truth)
    within = (err <= _TOL).mean(axis=(0, 1))
    return {
        "tolerance_accuracy": float(within.mean()),
        "per_feature": {
            f: {"mae": float(err[..., j].mean()), "within_tol": float(within[j])}
            for j, f in enumerate(OBSERVABLE_FEATURES)
        },
    }


def _cf_truth(base: Path, test: dict, scenarios: dict, horizon: int, trajectory: Path) -> dict:
    # Beside the REAL windows file, so variant directories symlinking one test set share one cache.
    cache = (base / "windows_test.npz").resolve().parent / "cf_truth_test.npz"
    if cache.exists():
        cached = dict(np.load(cache, allow_pickle=True))
        if np.array_equal(cached["scenario"], test["scenario"]) and np.array_equal(cached["end_day"], test["end_day"]):
            return cached
    from cell.aging import AgingRates
    from experiment.abduction import (
        SOAK_H,
        build_counterfactual_actions,
        ground_truth_counterfactual,
        planned_counterfactual,
        planned_ground_truth,
    )
    from sweep_dataset import load_driving

    meta = {s["scenario"]: s for s in json.loads((base / "scenarios.json").read_text())}
    driving, _ = load_driving(trajectory)
    n = test["scenario"].shape[0]
    cf_actions = np.zeros((n, horizon, test["actions"].shape[-1]), np.float32)
    cf_truth = np.zeros((n, horizon, len(OBSERVABLE_FEATURES)), np.float32)
    # Round 12 datasets carry per-day dials, so the counterfactual is "same
    # driving minutes, twice the cold starts" rather than a drive/park flip.
    per_day = "trips_today" in scenarios[str(test["scenario"][0])]
    for idx in range(n):
        name, end = str(test["scenario"][idx]), int(test["end_day"][idx])
        if per_day:
            cf_actions[idx], plans = planned_counterfactual(scenarios[name], end, horizon, SOAK_H)
            cf_truth[idx] = planned_ground_truth(meta[name], driving, scenarios[name], end, plans, AgingRates())
        else:
            cf_actions[idx], day_types = build_counterfactual_actions(scenarios[name], end, horizon)
            cf_truth[idx] = ground_truth_counterfactual(meta[name], driving, scenarios[name], end, day_types, AgingRates())
        if (idx + 1) % 500 == 0:
            print(f"[{idx + 1}/{n}] counterfactual ground truth", flush=True)
    out = {"scenario": test["scenario"], "end_day": test["end_day"], "cf_actions": cf_actions, "cf_truth": cf_truth}
    np.savez(cache, **out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--arch", choices=ARCHS, required=True)
    parser.add_argument("--clean-trajectory", default=None)
    parser.add_argument("--trajectory", default="runs/telemetry.csv")
    parser.add_argument("--cf", action="store_true")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    model, ck = load_model(base, args.arch)
    offsets = bool(ck.get("offsets", False))
    horizon = json.loads((base / "windows_meta.json").read_text())["rollout_k"]
    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))
    scenarios, _ = load_daily_trajectory(base / "daily_trajectory.csv")
    clean = load_daily_trajectory(Path(args.clean_trajectory))[0] if args.clean_trajectory else scenarios

    if args.cf:
        gt = _cf_truth(base, test, scenarios, horizon, Path(args.trajectory))
        cf = {"cf_actions": gt["cf_actions"], "cf_observables": np.zeros_like(gt["cf_truth"])}
        prepared = prepare(test, scenarios, horizon, offsets=offsets, cf=cf)
        act_key, anchor_key, truth = "cf_actions", "cf_anchors", gt["cf_truth"]
    else:
        prepared = prepare(test, scenarios, horizon, offsets=offsets)
        act_key, anchor_key = "future_actions", "future_anchors"
        truth = prepare(test, clean, horizon, offsets=False)["future_observables"]

    std = standardize_prepared(prepared, ck["action_mean"], ck["action_std"], ck["obs_mean"], ck["obs_std"])
    with torch.no_grad():
        actions = torch.tensor(std["actions"])
        observables = torch.tensor(std["observables"])
        state = model.abduct(actions, observables)
        anchors = torch.tensor(std[anchor_key]) if getattr(model, "anchored", False) else None
        pred_std = model.rollout(
            state, torch.tensor(std[act_key]), observables[:, -1, :], anchors=anchors,
            **rollout_kwargs(model, actions, observables),
        ).numpy()
    pred = real_units(pred_std, prepared[act_key], ck["obs_mean"], ck["obs_std"], offsets=offsets)
    baseline = prepared[anchor_key]
    if offsets:
        baseline = from_offsets(baseline, prepared[act_key])

    mode = "cf" if args.cf else "factual"
    result = {
        "arch": args.arch, "mode": mode, "offsets": offsets,
        "anchored": bool(getattr(model, "anchored", False)), "n_windows": int(truth.shape[0]),
        "model": score(pred, truth), "same_day_type_baseline": score(baseline, truth),
    }
    path = base / artifact(args.arch, f"accuracy_{mode}.json")
    path.write_text(json.dumps(result, indent=2))

    m, b = result["model"], result["same_day_type_baseline"]
    print(f"{args.arch} {mode} (offsets={offsets}, anchored={result['anchored']}), {result['n_windows']} windows")
    print(f"  tolerance accuracy: model {m['tolerance_accuracy']:.1%}   baseline {b['tolerance_accuracy']:.1%}")
    for f in OBSERVABLE_FEATURES:
        print(f"    {f:11s} model {m['per_feature'][f]['within_tol']:6.1%} (MAE {m['per_feature'][f]['mae']:.4f})"
              f"   baseline {b['per_feature'][f]['within_tol']:6.1%}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
