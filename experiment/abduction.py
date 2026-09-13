#!/usr/bin/env python3
"""Step 3: the abduction test -- the core contribution.

For every held-out (test-scenario) window in `windows_test.npz`:

  1. ABDUCT -- encode the factual observation window with the trained world
     model, take the posterior mean `mu` as the inferred latent (the model's
     summary of "what this battery's hidden state must be" given only what
     it observed).
  2. ACT -- roll the model's decoder forward `ROLLOUT_K` days under a
     DIFFERENT action sequence: every day's `is_layup` flag is FLIPPED
     relative to what actually happened (a day the battery actually drove is
     replaced with a park, and vice versa), which is exactly the kind of
     "what if the driver had done something else" question a world model is
     supposed to answer.
  3. COMPARE against the EXACT ground-truth counterfactual: `CellState` and
     `AgingState` are rebuilt from the window's true end-of-day resume
     fields (`experiment/calendar_sim.py`'s `RESUME_FIELDS` plus the true
     `crystal`/`corrosion_hours`/`shedding`), and `calendar_sim.roll_days` is
     called with the SAME flipped-action sequence -- literally reruns the
     physics ODE from the real hidden state under the counterfactual
     schedule. This is the ground truth: same generator, same hidden state,
     different action, which is the one thing a real dataset can never give
     you (see the module docstring in the experiment brief).

Three errors, all standardized MSE over the `K`-day rollout of
`OBSERVABLE_FEATURES` (same normalization Gate 2 used, for a like-for-like
scale):

  FACTUAL       -- model rollout vs ODE, under the FACTUAL action sequence.
                   How well the model predicts at all.
  COUNTERFACTUAL-- model rollout vs ODE, under the FLIPPED action sequence,
                   using the ABDUCTED latent.
  INTERVENTIONAL-- model rollout vs the SAME ODE counterfactual, but with the
                   latent replaced by the prior mean (zero vector) -- what
                   you get WITHOUT abduction, i.e. an action-conditioned
                   average over whatever hidden states the training
                   distribution contains.

Interpretation (stated plainly, in the printed summary and the JSON output):
counterfactual error close to factual error AND clearly below the
interventional baseline means the model is doing abduction; counterfactual
error close to the interventional baseline means it is not -- it is
producing an interventional average, not a genuine counterfactual, and that
is the stronger, more publishable finding if that is what comes out.

Run: `python3 -m experiment.abduction --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cell.aging import AgingRates, AgingState
from cell.integrate import CellState
from experiment.build_dataset import SOAK_H, TRIPS_PER_DAY
from experiment.calendar_sim import (
    ACTION_FEATURES,
    OBSERVABLE_FEATURES,
    rebuild_scenario,
    roll_days,
    truncate_driving,
)
from experiment.data import load_daily_trajectory
from experiment.model import WorldModel
from sweep_dataset import load_driving

ACTION_IDX = {name: i for i, name in enumerate(ACTION_FEATURES)}
OBS_IDX = {name: i for i, name in enumerate(OBSERVABLE_FEATURES)}


def load_model(base: Path) -> tuple[WorldModel, dict]:
    # weights_only=False: this checkpoint carries numpy normalization arrays
    # alongside the state_dict, and it is always our own file written by
    # experiment/train_world_model.py in this same run -- never untrusted.
    checkpoint = torch.load(base / "world_model.pt", map_location="cpu", weights_only=False)
    model = WorldModel(
        n_action=checkpoint["n_action"], n_obs=checkpoint["n_obs"],
        enc_hidden=checkpoint["enc_hidden"], latent_dim=checkpoint["latent_dim"],
        dec_hidden=checkpoint["dec_hidden"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _normal_day_action(arr: dict, feature: str) -> float:
    """The constant value ordinary (non-layup) days of this scenario carry."""
    mask = arr["is_layup"] == 0.0
    return float(arr[feature][mask][0])


def build_counterfactual_actions(
    arr: dict, end: int, k: int,
) -> tuple[np.ndarray, list[bool]]:
    """Flip is_layup for the k days after `end`. Returns (actions[k,4], day_types)."""
    factual_layup = arr["is_layup"][end + 1:end + 1 + k]
    cf_layup = 1.0 - factual_layup
    ambient_c = arr["ambient_c"][end]
    normal_drive_min = _normal_day_action(arr, "driving_minutes")
    normal_soak_h = _normal_day_action(arr, "soak_hours")

    actions = np.zeros((k, len(ACTION_FEATURES)), dtype=np.float32)
    day_types = []
    for t in range(k):
        is_layup = bool(cf_layup[t] >= 0.5)
        day_types.append(is_layup)
        actions[t, ACTION_IDX["is_layup"]] = 1.0 if is_layup else 0.0
        actions[t, ACTION_IDX["driving_minutes"]] = 0.0 if is_layup else normal_drive_min
        actions[t, ACTION_IDX["soak_hours"]] = 24.0 if is_layup else normal_soak_h
        actions[t, ACTION_IDX["ambient_c"]] = ambient_c
    return actions, day_types


def ground_truth_counterfactual(
    scenario_meta: dict, driving_raw, arr: dict, end: int, day_types: list[bool],
    rates: AgingRates,
) -> np.ndarray:
    """Rerun the ODE from the TRUE hidden state at `end`, under `day_types`."""
    aging = AgingState(
        corrosion_hours=float(arr["corrosion_hours"][end]),
        crystal=float(arr["crystal"][end]),
        shedding=float(arr["shedding"][end]),
    )
    state = CellState(
        soc=float(arr["soc"][end]), temp_c=float(arr["resume_temp_c"][end]),
        aging=aging, bay_temp_c=float(arr["resume_bay_temp_c"][end]),
    )
    scenario = rebuild_scenario(
        scenario_meta["ambient_c"], scenario_meta["parasitic_a"], scenario_meta["seed"],
    )
    trip_samples = truncate_driving(driving_raw, scenario_meta["trip_minutes"])
    last_coolant_c = float(trip_samples[-1]["coolant_c"])
    rows = roll_days(
        state, scenario, trip_samples, rates, day_types, TRIPS_PER_DAY, SOAK_H,
        last_coolant_c,
    )
    return np.array(
        [[row[f] for f in OBSERVABLE_FEATURES] for row in rows], dtype=np.float32,
    )


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--trajectory", default="runs/telemetry.csv")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 = use every window in windows_test.npz")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    model, checkpoint = load_model(base)
    action_mean, action_std = checkpoint["action_mean"], checkpoint["action_std"]
    obs_mean, obs_std = checkpoint["obs_mean"], checkpoint["obs_std"]

    windows_meta = json.loads((base / "windows_meta.json").read_text())
    k = windows_meta["rollout_k"]

    scenarios_list = json.loads((base / "scenarios.json").read_text())
    scenario_meta_by_name = {s["scenario"]: s for s in scenarios_list}

    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")
    driving_raw, _dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()

    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))
    n = test["actions"].shape[0]
    if args.max_examples:
        n = min(n, args.max_examples)

    factual_errors, cf_errors, interventional_errors = [], [], []
    per_example = []

    for idx in range(n):
        name = str(test["scenario"][idx])
        end = int(test["end_day"][idx])
        arr = scenarios[name]
        meta = scenario_meta_by_name[name]

        window_actions = test["actions"][idx]
        window_observables = test["observables"][idx]

        cf_actions_raw, cf_day_types = build_counterfactual_actions(arr, end, k)
        factual_actions_raw = np.stack(
            [arr[f][end + 1:end + 1 + k] for f in ACTION_FEATURES], axis=-1,
        ).astype(np.float32)
        true_factual_obs_raw = np.stack(
            [arr[f][end + 1:end + 1 + k] for f in OBSERVABLE_FEATURES], axis=-1,
        ).astype(np.float32)
        true_cf_obs_raw = ground_truth_counterfactual(
            meta, driving_raw, arr, end, cf_day_types, rates,
        )

        with torch.no_grad():
            win_actions_t = torch.tensor(
                standardize(window_actions, action_mean, action_std),
            ).unsqueeze(0)
            win_obs_t = torch.tensor(
                standardize(window_observables, obs_mean, obs_std),
            ).unsqueeze(0)
            mu, _logvar = model.encode(win_actions_t, win_obs_t)
            prev_obs0 = win_obs_t[:, -1, :]

            factual_actions_t = torch.tensor(
                standardize(factual_actions_raw, action_mean, action_std),
            ).unsqueeze(0)
            cf_actions_t = torch.tensor(
                standardize(cf_actions_raw, action_mean, action_std),
            ).unsqueeze(0)

            pred_factual = model.rollout(mu, factual_actions_t, prev_obs0)
            pred_cf_abducted = model.rollout(mu, cf_actions_t, prev_obs0)
            zero_latent = torch.zeros_like(mu)
            pred_cf_interventional = model.rollout(zero_latent, cf_actions_t, prev_obs0)

        true_factual_std = standardize(true_factual_obs_raw, obs_mean, obs_std)
        true_cf_std = standardize(true_cf_obs_raw, obs_mean, obs_std)

        f_err = float(((pred_factual.squeeze(0).numpy() - true_factual_std) ** 2).mean())
        c_err = float(((pred_cf_abducted.squeeze(0).numpy() - true_cf_std) ** 2).mean())
        i_err = float(((pred_cf_interventional.squeeze(0).numpy() - true_cf_std) ** 2).mean())

        factual_errors.append(f_err)
        cf_errors.append(c_err)
        interventional_errors.append(i_err)
        per_example.append({
            "scenario": name, "end_day": end,
            "true_crystal": float(arr["crystal"][end]),
            "factual_mse": f_err, "counterfactual_mse": c_err,
            "interventional_mse": i_err,
        })

        if (idx + 1) % 200 == 0 or idx == n - 1:
            print(f"[{idx + 1}/{n}] running means: "
                  f"factual={np.mean(factual_errors):.4f} "
                  f"counterfactual={np.mean(cf_errors):.4f} "
                  f"interventional={np.mean(interventional_errors):.4f}", flush=True)

    factual_mean = float(np.mean(factual_errors))
    cf_mean = float(np.mean(cf_errors))
    interventional_mean = float(np.mean(interventional_errors))

    # How much of the interventional-vs-factual gap the abducted latent closes.
    gap = interventional_mean - factual_mean
    closed = (interventional_mean - cf_mean) / gap if abs(gap) > 1e-12 else float("nan")

    is_abducting = (
        cf_mean < interventional_mean
        and abs(cf_mean - factual_mean) < abs(interventional_mean - factual_mean)
    )

    result = {
        "n_examples": n,
        "rollout_k_days": k,
        "factual_mse": factual_mean,
        "counterfactual_mse": cf_mean,
        "interventional_mse": interventional_mean,
        "fraction_of_gap_closed_by_abduction": closed,
        "verdict": (
            "ABDUCTING: counterfactual error is close to factual error and "
            "clearly below the interventional baseline."
            if is_abducting else
            "NOT ABDUCTING: counterfactual error is no better than the "
            "interventional (prior-latent) baseline -- the model is producing "
            "an interventional average conditioned on the action sequence, "
            "not a genuine per-battery counterfactual."
        ),
        "per_example": per_example,
    }
    (base / "abduction_results.json").write_text(json.dumps(result, indent=2))

    print("\nSTEP 3 -- abduction test:")
    print(f"  n_examples:                {n}")
    print(f"  FACTUAL error:             {factual_mean:.5f}")
    print(f"  COUNTERFACTUAL error:      {cf_mean:.5f}")
    print(f"  INTERVENTIONAL baseline:   {interventional_mean:.5f}")
    print(f"  fraction of gap closed:    {closed:.1%}" if not np.isnan(closed) else "  fraction of gap closed: n/a (no gap)")
    print(f"\n  {result['verdict']}")
    print(f"\nwrote {base / 'abduction_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
