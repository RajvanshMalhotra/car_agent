#!/usr/bin/env python3
"""Round 6 step 0: copy a dataset directory with sensor noise on the measurable observables.

Reads `<src>/daily_trajectory.csv`, applies `experiment.noise.add_daily_noise`,
and writes it to `<out>/daily_trajectory.csv` beside copies of
`scenarios.json` and `daily_dataset.json`. Hidden, resume, action and latent
columns are carried over bit-exact, so the abduction test can still restart
the ODE from the true state of any day. `noise.json` records what was applied
and the realised noise size per column.

Then, in `<out>`:
    python3 -m experiment.windows --dataset-dir <out> --holdout-ambient 42
    python3 -m experiment.build_counterfactual_targets --dataset-dir <out>

Run: `python3 -m experiment.make_noisy_variant --src runs/experiment --out runs/experiment_hot_noisy`
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from cell.sensors import SensorNoise
from experiment.data import load_daily_trajectory, write_daily_trajectory
from experiment.noise import DAILY_NOISE_CHANNEL, add_daily_noise

NOISE_SEED = 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="runs/experiment")
    parser.add_argument("--out", default="runs/experiment_hot_noisy")
    parser.add_argument("--seed", type=int, default=NOISE_SEED)
    args = parser.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean, order = load_daily_trajectory(src / "daily_trajectory.csv")
    noisy = add_daily_noise(clean, order, seed=args.seed)
    write_daily_trajectory(out / "daily_trajectory.csv", noisy, order)
    for name in ("scenarios.json", "daily_dataset.json"):
        shutil.copy2(src / name, out / name)

    realised = {
        column: float(np.concatenate([noisy[n][column] - clean[n][column] for n in order]).std())
        for column in DAILY_NOISE_CHANNEL
    }
    (out / "noise.json").write_text(json.dumps({
        "source": str(src / "daily_trajectory.csv"),
        "seed": args.seed,
        "daily_noise_channel": DAILY_NOISE_CHANNEL,
        "sensor_spec": SensorNoise().manifest(),
        "realised_noise_std": realised,
    }, indent=2))

    print(f"wrote {out / 'daily_trajectory.csv'} ({len(order)} scenarios)")
    for column, std in realised.items():
        print(f"  {column:11s} realised noise std {std:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
