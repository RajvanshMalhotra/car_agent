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

Four errors, all standardized MSE over the `K`-day rollout of
`OBSERVABLE_FEATURES` (same normalization Gate 2 used, for a like-for-like
scale):

  FACTUAL                -- model rollout vs ODE, under the FACTUAL action
                             sequence. How well the model predicts at all.
  COUNTERFACTUAL abducted-- model rollout vs ODE, under the FLIPPED action
                             sequence, using the ABDUCTED (full-window) latent.
  COUNTERFACTUAL lastday -- SAME flipped action sequence, but the latent comes
                             from re-encoding a window where every one of the
                             60 days is a COPY OF THE FINAL DAY -- the same
                             architecture, the same encoder, the same amount
                             of per-day information, with all path dependence
                             erased. This is the control that actually
                             isolates abduction: a model that just reads off
                             "today's SoC and temperature look like X" would
                             score identically here and on the abducted
                             condition, because that information survives the
                             flattening. Only genuinely PATH-dependent
                             information (what happened over the 60 days, not
                             just where it ended up) can make the abducted
                             latent do better than this one.
  INTERVENTIONAL zero    -- SAME flipped action sequence, latent replaced by
                             the prior mean (zero vector) -- the weakest
                             possible control (no information at all). Kept
                             for continuity with the first two runs of this
                             experiment, but see the report for why it is too
                             weak to support an abduction claim on its own:
                             ANY non-dead latent beats it, whether or not what
                             it carries is path-dependent.

**The headline comparison is abducted vs LASTDAY, not abducted vs zero.**
Beating zero only shows the latent isn't dead (Gate 2b already establishes
this). Beating lastday is the actual claim under test: that the model
recovered something about the window's HISTORY that the final day alone does
not contain -- the definition of the hidden, path-dependent `crystal` state
this experiment is about.

Interpretation (stated plainly, in the printed summary and the JSON output):
abducted clearly better than lastday means the model is doing genuine
path-dependent abduction; abducted no better than lastday means it is not --
it is conditioning on cheap, currently-observable state, and whatever
counterfactual improvement it shows over the zero baseline is fully explained
without invoking any hidden-state inference at all.

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


def flatten_to_last_day(window: np.ndarray) -> np.ndarray:
    """Replace every day in a `[L, F]` window with a copy of its final day.

    Holds the encoder architecture and the per-day information volume fixed
    (still an `L`-day sequence, still the same features) and removes ONLY the
    path dependence: every day now carries the same values, so an encoder
    that reads off "what does the most recent day look like" gets identical
    input to the real window, while an encoder that integrates HOW the
    window got there gets a degenerate, historyless one. This is why (b) --
    re-encoding a flattened window -- was chosen over (a) -- a learned
    last-day-to-latent regressor: (a) would need its own training run and
    would confound "what the map learned to predict" with "what history
    contains", where (b) reuses the SAME trained encoder unmodified and asks
    it to do the one thing that isolates the question this control exists
    for.
    """
    last_day = window[-1:, :]
    return np.repeat(last_day, window.shape[0], axis=0)


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

    factual_errors, cf_errors, lastday_errors, interventional_errors = [], [], [], []
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

            # Last-day-only control: same encoder, a window with all path
            # dependence erased (every day replaced by the final day).
            lastday_actions_t = torch.tensor(
                flatten_to_last_day(standardize(window_actions, action_mean, action_std)),
            ).unsqueeze(0)
            lastday_obs_t = torch.tensor(
                flatten_to_last_day(standardize(window_observables, obs_mean, obs_std)),
            ).unsqueeze(0)
            lastday_mu, _lastday_logvar = model.encode(lastday_actions_t, lastday_obs_t)

            factual_actions_t = torch.tensor(
                standardize(factual_actions_raw, action_mean, action_std),
            ).unsqueeze(0)
            cf_actions_t = torch.tensor(
                standardize(cf_actions_raw, action_mean, action_std),
            ).unsqueeze(0)

            pred_factual = model.rollout(mu, factual_actions_t, prev_obs0)
            pred_cf_abducted = model.rollout(mu, cf_actions_t, prev_obs0)
            pred_cf_lastday = model.rollout(lastday_mu, cf_actions_t, prev_obs0)
            zero_latent = torch.zeros_like(mu)
            pred_cf_interventional = model.rollout(zero_latent, cf_actions_t, prev_obs0)

        true_factual_std = standardize(true_factual_obs_raw, obs_mean, obs_std)
        true_cf_std = standardize(true_cf_obs_raw, obs_mean, obs_std)

        f_err = float(((pred_factual.squeeze(0).numpy() - true_factual_std) ** 2).mean())
        c_err = float(((pred_cf_abducted.squeeze(0).numpy() - true_cf_std) ** 2).mean())
        l_err = float(((pred_cf_lastday.squeeze(0).numpy() - true_cf_std) ** 2).mean())
        i_err = float(((pred_cf_interventional.squeeze(0).numpy() - true_cf_std) ** 2).mean())

        factual_errors.append(f_err)
        cf_errors.append(c_err)
        lastday_errors.append(l_err)
        interventional_errors.append(i_err)
        per_example.append({
            "scenario": name, "end_day": end,
            "true_crystal": float(arr["crystal"][end]),
            "factual_mse": f_err, "counterfactual_mse": c_err,
            "lastday_mse": l_err, "interventional_mse": i_err,
            "abducted_beats_lastday": c_err < l_err,
        })

        if (idx + 1) % 200 == 0 or idx == n - 1:
            print(f"[{idx + 1}/{n}] running means: "
                  f"factual={np.mean(factual_errors):.4f} "
                  f"counterfactual={np.mean(cf_errors):.4f} "
                  f"lastday={np.mean(lastday_errors):.4f} "
                  f"interventional={np.mean(interventional_errors):.4f}", flush=True)

    factual_mean = float(np.mean(factual_errors))
    cf_mean = float(np.mean(cf_errors))
    lastday_mean = float(np.mean(lastday_errors))
    interventional_mean = float(np.mean(interventional_errors))

    # HEADLINE: how much of the factual-to-lastday gap the abducted latent
    # closes. This, not the gap against zero, is the number that isolates
    # genuine path-dependent abduction from cheap current-state conditioning.
    lastday_gap = lastday_mean - factual_mean
    closed_vs_lastday = (
        (lastday_mean - cf_mean) / lastday_gap if abs(lastday_gap) > 1e-12 else float("nan")
    )

    # Kept for continuity with the first two runs -- see module docstring for
    # why this comparison alone cannot support an abduction claim.
    zero_gap = interventional_mean - factual_mean
    closed_vs_zero = (
        (interventional_mean - cf_mean) / zero_gap if abs(zero_gap) > 1e-12 else float("nan")
    )

    win_count = sum(1 for e in per_example if e["abducted_beats_lastday"])
    win_rate_vs_lastday = win_count / n

    # A mean-only comparison is not enough: MSE means are easily dragged by a
    # minority of high-error examples, so "abducted beats lastday on average"
    # can coexist with "abducted loses to lastday on most individual
    # windows" -- that combination is exactly what a skewed, outlier-driven
    # aggregate looks like, not a consistent per-battery effect. The
    # per-example win rate is the check for that: chance is 50%, so it must
    # sit CLEARLY above 50% (not just the mean beating lastday) before this
    # counts as abduction rather than noise.
    WIN_RATE_ABOVE_CHANCE = 0.55
    aggregate_favors_abducted = (
        cf_mean < lastday_mean
        and abs(cf_mean - factual_mean) < abs(lastday_mean - factual_mean)
    )
    is_abducting = aggregate_favors_abducted and win_rate_vs_lastday > WIN_RATE_ABOVE_CHANCE

    # Five-bin breakdown by true crystal value: if abduction is real it
    # should help MOST mid-transition, where crystal is changing but SoC and
    # temperature look ordinary -- exactly where the lastday control has no
    # way to see it and the abducted latent would have to.
    n_bins = 5
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    crystal_bins = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [
            e for e in per_example
            if lo <= e["true_crystal"] < hi or (b == n_bins - 1 and e["true_crystal"] == hi)
        ]
        if not in_bin:
            crystal_bins.append({
                "range": f"[{lo:.1f},{hi:.1f})", "n": 0,
            })
            continue
        f_b = float(np.mean([e["factual_mse"] for e in in_bin]))
        c_b = float(np.mean([e["counterfactual_mse"] for e in in_bin]))
        l_b = float(np.mean([e["lastday_mse"] for e in in_bin]))
        i_b = float(np.mean([e["interventional_mse"] for e in in_bin]))
        gap_b = l_b - f_b
        closed_b = (l_b - c_b) / gap_b if abs(gap_b) > 1e-12 else float("nan")
        wins_b = sum(1 for e in in_bin if e["abducted_beats_lastday"])
        crystal_bins.append({
            "range": f"[{lo:.1f},{hi:.1f})", "n": len(in_bin),
            "factual_mse": f_b, "counterfactual_mse": c_b,
            "lastday_mse": l_b, "interventional_mse": i_b,
            "fraction_of_factual_to_lastday_gap_closed": closed_b,
            "win_rate_abducted_vs_lastday": wins_b / len(in_bin),
        })

    if is_abducting:
        verdict = (
            "GENUINE PATH-DEPENDENT ABDUCTION: counterfactual error with the "
            "abducted latent is close to factual error, clearly below the "
            "last-day-only control on average, AND the abducted latent wins "
            "on a clear majority of individual windows -- the model is using "
            "information about the window's HISTORY that the final day alone "
            "does not contain."
        )
    elif aggregate_favors_abducted:
        verdict = (
            "NOT ABDUCTION -- AGGREGATE IS NOISE: the mean counterfactual "
            f"error looks better than lastday ({closed_vs_lastday:.1%} of the "
            "factual-to-lastday gap closed), but the per-example win rate is "
            f"only {win_rate_vs_lastday:.1%} (chance is 50%), not clearly "
            "above it. A mean improvement with a win rate at or below chance "
            "means a minority of high-error examples is dragging the "
            "average, not that the abducted latent is consistently better "
            "per battery. This is exactly the failure mode the win-rate "
            "check exists to catch, and it fails it here."
        )
    else:
        verdict = (
            "NOT ABDUCTION, STATE-CONDITIONING: counterfactual error with the "
            "abducted latent is no better than the last-day-only control, "
            "which has the same architecture and the same per-day "
            "information but zero path dependence. Whatever counterfactual "
            "improvement the abducted latent shows over the zero baseline is "
            "fully explained by reading off the window's most recent "
            "observable state -- not by inferring anything about its "
            "history."
        )

    result = {
        "n_examples": n,
        "rollout_k_days": k,
        "factual_mse": factual_mean,
        "counterfactual_mse": cf_mean,
        "lastday_mse": lastday_mean,
        "interventional_mse": interventional_mean,
        "fraction_of_factual_to_lastday_gap_closed": closed_vs_lastday,
        "fraction_of_factual_to_zero_gap_closed": closed_vs_zero,
        "win_rate_abducted_vs_lastday": win_rate_vs_lastday,
        "win_count_abducted_vs_lastday": f"{win_count}/{n}",
        "crystal_bin_breakdown": crystal_bins,
        "verdict": verdict,
        "per_example": per_example,
    }
    (base / "abduction_results.json").write_text(json.dumps(result, indent=2))

    print("\nSTEP 3 -- abduction test (four-way comparison):")
    print(f"  n_examples:                {n}")
    print(f"  FACTUAL error:             {factual_mean:.5f}")
    print(f"  COUNTERFACTUAL abducted:   {cf_mean:.5f}")
    print(f"  COUNTERFACTUAL lastday:    {lastday_mean:.5f}   <-- the control that matters")
    print(f"  INTERVENTIONAL zero:       {interventional_mean:.5f}")
    print(f"  fraction of factual->lastday gap closed: "
          f"{closed_vs_lastday:.1%}" if not np.isnan(closed_vs_lastday) else
          "  fraction of factual->lastday gap closed: n/a (no gap)")
    print(f"  fraction of factual->zero gap closed (weak control): "
          f"{closed_vs_zero:.1%}" if not np.isnan(closed_vs_zero) else
          "  fraction of factual->zero gap closed: n/a (no gap)")
    print(f"  win rate, abducted vs lastday: {win_count}/{n} ({win_rate_vs_lastday:.1%})")
    print("\n  crystal-bin breakdown:")
    for row in crystal_bins:
        if row["n"] == 0:
            print(f"    {row['range']}: n=0")
            continue
        closed_str = (
            f"{row['fraction_of_factual_to_lastday_gap_closed']:.1%}"
            if not np.isnan(row["fraction_of_factual_to_lastday_gap_closed"]) else "n/a"
        )
        print(f"    {row['range']}: n={row['n']:4d}  factual={row['factual_mse']:.4f}  "
              f"abducted={row['counterfactual_mse']:.4f}  lastday={row['lastday_mse']:.4f}  "
              f"gap_closed={closed_str}  win_rate={row['win_rate_abducted_vs_lastday']:.1%}")
    print(f"\n  {verdict}")
    print(f"\nwrote {base / 'abduction_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
