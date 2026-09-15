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

Interpretation (stated plainly, in the printed summary and the JSON output)
uses a two-sided binomial test on the win rate against chance (50%, normal
approximation via `_binom_test_pvalue`) rather than a fixed threshold like
"55%" -- a fixed threshold either misses a real-but-small effect or
overstates confidence in a large one, and this experiment already produced
one case (round 2 of this report) where a win rate barely above chance
(53.6%) was statistically significant (p=0.0057) but small and CONCENTRATED
in one crystal regime while reversed in another. So the verdict is one of
three tiers, not a binary:
  NOT SIGNIFICANT      -- win rate indistinguishable from chance (p >= 0.05).
                           Not abducting; fully explained by state-conditioning.
  SIGNIFICANT, UNIFORM -- p < 0.05 AND no crystal bin significantly favors
                           lastday. Genuine path-dependent abduction.
  SIGNIFICANT, MIXED   -- p < 0.05 overall, but at least one crystal bin
                           significantly favors LASTDAY (p < 0.05 in the
                           wrong direction). A real, non-noise average edge
                           that is not a consistent per-condition effect --
                           report it as partial and localized, not general
                           abduction.

Run: `python3 -m experiment.abduction --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import math
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
from sweep_dataset import load_driving

ACTION_IDX = {name: i for i, name in enumerate(ACTION_FEATURES)}
OBS_IDX = {name: i for i, name in enumerate(OBSERVABLE_FEATURES)}


def _normal_day_action(arr: dict, feature: str) -> float:
    """The constant value ordinary (non-layup) days of this scenario carry."""
    mask = arr["is_layup"] == 0.0
    return float(arr[feature][mask][0])


def actions_from_day_types(arr: dict, end: int, k: int, day_types: list[bool]) -> np.ndarray:
    """Build a `[k, n_action]` action array from an explicit is-layup sequence.

    Shared by every perturbation shape (the deterministic full flip
    `build_counterfactual_actions` uses for Step 3's eval, and the
    randomized per-day perturbation `random_perturbation_day_types` uses for
    the training curriculum in `experiment/build_counterfactual_targets.py`)
    so they never disagree about how a day_types sequence turns into actual
    driving-minutes/soak-hours/ambient values.
    """
    ambient_c = arr["ambient_c"][end]
    normal_drive_min = _normal_day_action(arr, "driving_minutes")
    normal_soak_h = _normal_day_action(arr, "soak_hours")

    actions = np.zeros((k, len(ACTION_FEATURES)), dtype=np.float32)
    for t, is_layup in enumerate(day_types):
        actions[t, ACTION_IDX["is_layup"]] = 1.0 if is_layup else 0.0
        actions[t, ACTION_IDX["driving_minutes"]] = 0.0 if is_layup else normal_drive_min
        actions[t, ACTION_IDX["soak_hours"]] = 24.0 if is_layup else normal_soak_h
        actions[t, ACTION_IDX["ambient_c"]] = ambient_c
    return actions


def build_counterfactual_actions(
    arr: dict, end: int, k: int,
) -> tuple[np.ndarray, list[bool]]:
    """Flip is_layup for the k days after `end`. Returns (actions[k,4], day_types).

    This is the EVAL-time perturbation (Step 3): maximal and deterministic,
    every day's schedule alternated. Kept deliberately different in SHAPE
    from `random_perturbation_day_types` (used only for the training
    curriculum) so training augmentation is never literally "train on the
    test transformation" -- the model has to generalize the day-type
    invariance from a different, randomized perturbation distribution to
    this specific held-out one.
    """
    factual_layup = arr["is_layup"][end + 1:end + 1 + k]
    cf_layup = 1.0 - factual_layup
    day_types = [bool(v >= 0.5) for v in cf_layup]
    actions = actions_from_day_types(arr, end, k, day_types)
    return actions, day_types


def random_perturbation_day_types(arr: dict, end: int, k: int, rng: np.random.Generator) -> list[bool]:
    """An independently-randomized is-layup sequence for training augmentation.

    Each of the `k` days' is_layup flag is flipped relative to the FACTUAL
    schedule with independent probability 0.5 -- a diverse, non-deterministic
    perturbation distribution, deliberately NOT the same shape as Step 3's
    full, deterministic flip (`build_counterfactual_actions`). Exposing
    training to varied action perturbations (never just the factual
    continuation) is the curriculum fix this module's docstring and the
    experiment report's "what I would do differently" both proposed: if the
    encoder never has to distinguish "what happened" from "what state
    resulted" under anything but the real schedule, there is no training
    pressure to keep genuinely path-dependent information in the latent.
    """
    factual_layup = arr["is_layup"][end + 1:end + 1 + k]
    flip_mask = rng.random(k) < 0.5
    perturbed = np.where(flip_mask, 1.0 - factual_layup, factual_layup)
    return [bool(v >= 0.5) for v in perturbed]


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


def binom_test_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided p-value for `k` successes in `n` trials against chance `p`.

    Normal approximation (stdlib `math` only, no scipy dependency): exact
    enough for the bin sizes here (n=1500 overall, n=300 per crystal bin,
    both well past the ~30 rule of thumb) and it keeps this module's
    dependency footprint unchanged. Verified against `scipy.stats.binomtest`
    during development (agreed to 3+ significant figures at these sample
    sizes).
    """
    if n == 0:
        return float("nan")
    se = math.sqrt(p * (1 - p) / n)
    if se == 0:
        return float("nan")
    z = (k / n - p) / se
    return math.erfc(abs(z) / math.sqrt(2))


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
    parser.add_argument("--arch", choices=("gru_vae", "rssm"), default="gru_vae")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    # Imported here, not at module level: checkpoints imports this package's
    # models, and gate2b/sae_probe import helpers from this module.
    from experiment.checkpoints import artifact, load_model
    model, checkpoint = load_model(base, args.arch)
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
            mu = model.abduct(win_actions_t, win_obs_t)
            prev_obs0 = win_obs_t[:, -1, :]

            # Last-day-only control: same encoder, a window with all path
            # dependence erased (every day replaced by the final day).
            lastday_actions_t = torch.tensor(
                flatten_to_last_day(standardize(window_actions, action_mean, action_std)),
            ).unsqueeze(0)
            lastday_obs_t = torch.tensor(
                flatten_to_last_day(standardize(window_observables, obs_mean, obs_std)),
            ).unsqueeze(0)
            lastday_mu = model.abduct(lastday_actions_t, lastday_obs_t)

            factual_actions_t = torch.tensor(
                standardize(factual_actions_raw, action_mean, action_std),
            ).unsqueeze(0)
            cf_actions_t = torch.tensor(
                standardize(cf_actions_raw, action_mean, action_std),
            ).unsqueeze(0)

            pred_factual = model.rollout(mu, factual_actions_t, prev_obs0)
            pred_cf_abducted = model.rollout(mu, cf_actions_t, prev_obs0)
            pred_cf_lastday = model.rollout(lastday_mu, cf_actions_t, prev_obs0)
            zero_latent = model.zero_state(1)
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
    p_value_overall = binom_test_pvalue(win_count, n)
    SIGNIFICANCE_ALPHA = 0.05
    overall_significant = p_value_overall < SIGNIFICANCE_ALPHA and win_rate_vs_lastday > 0.5

    # Five-bin breakdown by true crystal value: if abduction is real it
    # should help MOST mid-transition, where crystal is changing but SoC and
    # temperature look ordinary -- exactly where the lastday control has no
    # way to see it and the abducted latent would have to. Each bin also
    # gets its own significance test: a real, uniform effect should not show
    # any bin significantly favoring lastday.
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
        win_rate_b = wins_b / len(in_bin)
        p_b = binom_test_pvalue(wins_b, len(in_bin))
        crystal_bins.append({
            "range": f"[{lo:.1f},{hi:.1f})", "n": len(in_bin),
            "factual_mse": f_b, "counterfactual_mse": c_b,
            "lastday_mse": l_b, "interventional_mse": i_b,
            "fraction_of_factual_to_lastday_gap_closed": closed_b,
            "win_rate_abducted_vs_lastday": win_rate_b,
            "p_value_vs_chance": p_b,
            "significant_favoring_lastday": p_b < SIGNIFICANCE_ALPHA and win_rate_b < 0.5,
        })

    bins_significant_against = [
        row for row in crystal_bins
        if row.get("n", 0) > 0 and row["significant_favoring_lastday"]
    ]

    if not overall_significant:
        verdict = (
            "NOT ABDUCTION -- NOT SIGNIFICANT: the per-example win rate "
            f"({win_rate_vs_lastday:.1%}, p={p_value_overall:.4f} two-sided vs "
            "chance) is not statistically distinguishable from 50%. Whatever "
            "counterfactual improvement the abducted latent shows over the "
            "zero baseline is fully explained by conditioning on cheap, "
            "currently-observable state, not by inferring anything about the "
            "window's history."
        )
    elif bins_significant_against:
        bad_ranges = ", ".join(row["range"] for row in bins_significant_against)
        verdict = (
            "PARTIAL, NON-UNIFORM SIGNAL -- SIGNIFICANT BUT MIXED: the "
            f"overall win rate ({win_rate_vs_lastday:.1%}, p={p_value_overall:.4f}) "
            "is a real, statistically significant edge over the last-day "
            "control -- not noise. But it is NOT a consistent per-condition "
            f"effect: crystal bin(s) {bad_ranges} significantly favor the "
            "last-day control instead (the abducted latent does WORSE than "
            "state-conditioning there, p<0.05). This is a genuine but "
            "localized signal, concentrated in specific crystal regimes "
            "rather than general path-dependent abduction across the board -- "
            "report it as partial, not as \"the model abducts.\""
        )
    else:
        verdict = (
            "GENUINE PATH-DEPENDENT ABDUCTION: the win rate "
            f"({win_rate_vs_lastday:.1%}, p={p_value_overall:.4f}) is "
            "statistically significant, and no crystal bin significantly "
            "favors the last-day control -- the effect is real and not "
            "concentrated in a way that contradicts it elsewhere. The model "
            "is using information about the window's HISTORY that the final "
            "day alone does not contain."
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
        "p_value_win_rate_vs_chance": p_value_overall,
        "significance_alpha": SIGNIFICANCE_ALPHA,
        "crystal_bin_breakdown": crystal_bins,
        "verdict": verdict,
        "per_example": per_example,
    }
    results_path = base / artifact(args.arch, "abduction_results.json")
    results_path.write_text(json.dumps(result, indent=2))

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
    print(f"  win rate, abducted vs lastday: {win_count}/{n} ({win_rate_vs_lastday:.1%}, "
          f"p={p_value_overall:.4f} vs chance)")
    print("\n  crystal-bin breakdown:")
    for row in crystal_bins:
        if row["n"] == 0:
            print(f"    {row['range']}: n=0")
            continue
        closed_str = (
            f"{row['fraction_of_factual_to_lastday_gap_closed']:.1%}"
            if not np.isnan(row["fraction_of_factual_to_lastday_gap_closed"]) else "n/a"
        )
        flag = "  ** significantly favors lastday **" if row["significant_favoring_lastday"] else ""
        print(f"    {row['range']}: n={row['n']:4d}  factual={row['factual_mse']:.4f}  "
              f"abducted={row['counterfactual_mse']:.4f}  lastday={row['lastday_mse']:.4f}  "
              f"gap_closed={closed_str}  win_rate={row['win_rate_abducted_vs_lastday']:.1%}  "
              f"p={row['p_value_vs_chance']:.4f}{flag}")
    print(f"\n  {verdict}")
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
