#!/usr/bin/env python3
"""Step 1b: turn the daily calendar trajectory into observation windows.

Each example is a `WINDOW_LEN`-day (60-day) span of `ACTION_FEATURES` and
`OBSERVABLE_FEATURES`, paired with the true `crystal` at the window's end
(the abduction target), the next day's action/observable pair (the Gate-2
next-step-prediction target), and `corrosion_hours`/`shedding`/`soh` retained
as ground truth for later scoring -- never as model input.

**Target flattening.** The brief's own prior sweep found the terminal crystal
distribution is a mostly-saturated, mostly-empty state when sampled only at
a scenario's END (or at a fixed 10-year horizon): the *transition* through
the middle happens over roughly 540-1170 days but any fixed sampling point
lands past it almost every time. Sampling a window end day UNIFORMLY at
random over a scenario's life reproduces exactly that skew. So windows here
are drawn by STRATIFIED sampling on the crystal target itself: bin the
candidate (scenario, end_day) pool into 10 crystal bins and cap how many are
drawn from each bin, rather than drawing candidates uniformly and hoping the
transition band is represented. See Gate 1's histogram output for whether
this worked.

**Scenario-level holdout.** The train/test split happens on whole scenarios,
before any window is drawn, so no window in `windows_test.npz` shares a
scenario with any window in `windows_train.npz` -- the abduction test in
`experiment/abduction.py` needs genuinely unseen batteries, not unseen days
of a battery the model already saw part of.

Run: `python3 -m experiment.windows --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.data import load_daily_trajectory

#: 60 days of daily summaries per window -- long enough to span several
#: trip/layup cycles (layup_gap_days ranges 20-120), short enough that a
#: modest GRU encoder is the right tool.
WINDOW_LEN = 60

#: How many days of clean runway every window is guaranteed beyond its end
#: day, so the SAME windows (train or test) could in principle support a
#: multi-day rollout probe -- the abduction test in Step 3 uses this for its
#: held-out scenarios.
ROLLOUT_K = 30

N_BINS = 10

#: Sampling ceilings, not guarantees -- a bin with fewer candidates than this
#: contributes everything it has. See Gate 1's printed histogram for the
#: actual counts.
TARGET_PER_SPLIT = {"train": 6000, "val": 1000, "test": 1500}

SPLIT_SEED = 7
TRAIN_FRACTION = 0.8

#: Share of the NON-held-out batteries kept back for best-epoch selection
#: when a climate is held out (round 6). Split by battery, like test.
VAL_FRACTION = 0.15


def scenario_split(
    scenario_names: list[str], train_fraction: float = TRAIN_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[set[str], set[str]]:
    names = sorted(scenario_names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_train = int(round(len(names) * train_fraction))
    return set(names[:n_train]), set(names[n_train:])


def climate_of(base: Path, scenarios: dict, order: list[str]) -> dict[str, float]:
    """Each battery's climate, for splitting folds by it.

    Rounds 4-11 held `ambient_c` constant for a battery's life, so day 0's
    value WAS the climate. Round 12 varies it seasonally -- 79 distinct day-0
    values across 80 batteries, and the 42 C battery starts its year at
    30.2 C -- so the climate has to come from the recorded annual mean
    instead. Falls back to the trajectory column when that metadata is
    absent, which keeps every earlier dataset splitting exactly as before.
    """
    meta_path = base / "scenarios.json"
    if meta_path.exists():
        meta = {m["scenario"]: m for m in json.loads(meta_path.read_text())}
        if all("annual_mean_c" in meta.get(n, {}) for n in order):
            return {n: float(meta[n]["annual_mean_c"]) for n in order}
    return {n: float(scenarios[n]["ambient_c"][0]) for n in order}


def holdout_split(
    scenario_names: list[str], ambient_by_scenario: dict[str, float],
    holdout_ambient: float, val_fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED,
) -> tuple[set[str], set[str], set[str]]:
    """(train, val, test) with every battery at `holdout_ambient` in test only.

    Round 6's generalization test: the models never see the held-out climate,
    so test is a genuinely new condition rather than an interpolation between
    batteries drawn from the same four climates. Validation comes from the
    remaining climates, by battery, so choosing the best epoch never looks at
    the held-out one.
    """
    test = {n for n in scenario_names if ambient_by_scenario[n] == holdout_ambient}
    if not test:
        raise ValueError(f"no scenario has ambient {holdout_ambient}")
    rest = sorted(set(scenario_names) - test)
    random.Random(seed).shuffle(rest)
    n_val = int(round(len(rest) * val_fraction))
    return set(rest[n_val:]), set(rest[:n_val]), test


def _candidates(
    scenarios: dict, names: set[str],
) -> list[tuple[str, int, float]]:
    """Every (scenario, end_day, crystal_target) with a full window and runway."""
    out = []
    for name in names:
        arr = scenarios[name]
        n_days = arr["crystal"].shape[0]
        last_end = n_days - 1 - ROLLOUT_K
        for end in range(WINDOW_LEN - 1, last_end + 1):
            out.append((name, end, float(arr["crystal"][end])))
    return out


def _stratified_sample(
    candidates: list[tuple[str, int, float]], desired_total: int,
    rng: random.Random,
) -> list[tuple[str, int, float]]:
    bins: list[list[tuple[str, int, float]]] = [[] for _ in range(N_BINS)]
    for item in candidates:
        b = min(N_BINS - 1, int(item[2] * N_BINS))
        bins[b].append(item)
    per_bin = max(1, desired_total // N_BINS)
    chosen: list[tuple[str, int, float]] = []
    for bucket in bins:
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bin])
    rng.shuffle(chosen)
    return chosen


def _extract(scenarios: dict, items: list[tuple[str, int, float]]) -> dict[str, np.ndarray]:
    n = len(items)
    n_action = len(ACTION_FEATURES)
    n_obs = len(OBSERVABLE_FEATURES)
    actions = np.zeros((n, WINDOW_LEN, n_action), dtype=np.float32)
    observables = np.zeros((n, WINDOW_LEN, n_obs), dtype=np.float32)
    next_action = np.zeros((n, n_action), dtype=np.float32)
    next_observable = np.zeros((n, n_obs), dtype=np.float32)
    target_crystal = np.zeros(n, dtype=np.float32)
    target_corrosion_hours = np.zeros(n, dtype=np.float32)
    target_shedding = np.zeros(n, dtype=np.float32)
    target_soh = np.zeros(n, dtype=np.float32)
    target_soc = np.zeros(n, dtype=np.float32)
    end_day = np.zeros(n, dtype=np.int32)
    scenario_names: list[str] = []

    for idx, (name, end, _) in enumerate(items):
        arr = scenarios[name]
        start = end - WINDOW_LEN + 1
        for j, f in enumerate(ACTION_FEATURES):
            actions[idx, :, j] = arr[f][start:end + 1]
        for j, f in enumerate(OBSERVABLE_FEATURES):
            observables[idx, :, j] = arr[f][start:end + 1]
        for j, f in enumerate(ACTION_FEATURES):
            next_action[idx, j] = arr[f][end + 1]
        for j, f in enumerate(OBSERVABLE_FEATURES):
            next_observable[idx, j] = arr[f][end + 1]
        target_crystal[idx] = arr["crystal"][end]
        target_corrosion_hours[idx] = arr["corrosion_hours"][end]
        target_shedding[idx] = arr["shedding"][end]
        target_soh[idx] = arr["soh"][end]
        target_soc[idx] = arr["soc"][end]
        end_day[idx] = end
        scenario_names.append(name)

    return dict(
        actions=actions, observables=observables,
        next_action=next_action, next_observable=next_observable,
        target_crystal=target_crystal,
        target_corrosion_hours=target_corrosion_hours,
        target_shedding=target_shedding, target_soh=target_soh,
        target_soc=target_soc,
        end_day=end_day, scenario=np.array(scenario_names),
    )


def histogram(values: np.ndarray, n_bins: int = N_BINS) -> dict[str, int]:
    bins = [0] * n_bins
    for v in values:
        b = min(n_bins - 1, int(float(v) * n_bins))
        bins[b] += 1
    return {f"[{i/n_bins:.1f},{(i+1)/n_bins:.1f})": bins[i] for i in range(n_bins)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--holdout-ambient", type=float, default=None,
                        help="test on every battery at this ambient, train/val on the rest")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    scenarios, order = load_daily_trajectory(base / "daily_trajectory.csv")
    if args.holdout_ambient is None:
        train_names, test_names = scenario_split(order)
        splits = {"train": train_names, "test": test_names}
    else:
        ambient = climate_of(base, scenarios, order)
        train_names, val_names, test_names = holdout_split(order, ambient, args.holdout_ambient)
        splits = {"train": train_names, "val": val_names, "test": test_names}

    rng = random.Random(SPLIT_SEED)
    counts = {}
    hist = {}
    for split, names in splits.items():
        desired = TARGET_PER_SPLIT[split]
        candidates = _candidates(scenarios, names)
        chosen = _stratified_sample(candidates, desired, rng)
        data = _extract(scenarios, chosen)
        np.savez(base / f"windows_{split}.npz", **data)
        hist[split] = histogram(data["target_crystal"])
        counts[split] = {
            "n_windows": len(chosen), "n_scenarios": len(names),
            "n_candidates": len(candidates),
        }
        print(
            f"{split}: {len(chosen)} windows from {len(names)} scenarios "
            f"({len(candidates)} candidates available)"
        )

    (base / "windows_meta.json").write_text(json.dumps({
        "window_len": WINDOW_LEN, "rollout_k": ROLLOUT_K,
        "action_features": list(ACTION_FEATURES),
        "observable_features": list(OBSERVABLE_FEATURES),
        "train_scenarios": sorted(train_names),
        **({"val_scenarios": sorted(splits["val"])} if "val" in splits else {}),
        "test_scenarios": sorted(test_names),
        "holdout_ambient": args.holdout_ambient,
        "split_seed": SPLIT_SEED,
        "counts": counts,
        "target_crystal_histogram": hist,
    }, indent=2))

    max_frac = 0.0
    print("\nGATE 1 -- crystal target histogram (10 bins):")
    for split in splits:
        print(f"  {split}:")
        n = counts[split]["n_windows"]
        for label, count in hist[split].items():
            frac = count / max(1, n)
            max_frac = max(max_frac, frac)
            print(f"    {label}: {count:5d}  ({frac:5.1%})")
    print(f"\nlargest single bin fraction across both splits: {max_frac:.1%} "
          f"({'PASS' if max_frac <= 0.27 else 'FAIL'} -- gate wants <= ~25%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
