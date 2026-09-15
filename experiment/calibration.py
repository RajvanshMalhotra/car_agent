#!/usr/bin/env python3
"""Step 5: do the model's prediction intervals contain the truth as often as they claim?

The output contract (CLAUDE.md) is a calibrated interval, never a bare
point: "847 days (90% interval 620-1150), plan for 620". What gets validated
is calibration -- if a 90% interval is claimed, the truth must land inside it
about 90% of the time on held-out data. This measures that on the factual
30-day continuation of every test window.

For each window, `n` stochastic futures are drawn with `sample_rollouts`
(gru_vae: `n` posterior draws of its one frozen `z`; RSSM: sampled filtering
and a fresh prior draw every imagined day). The central `level` band of those
samples is the claimed interval, per window, per day, per observable.
Coverage is the fraction of (window, day, observable) cells whose true value
falls inside it.

Read the result both ways: coverage well below `level` is overconfidence,
well above is intervals too wide to be useful. Width by horizon day shows
whether uncertainty grows as it should.

Coverage is computed in standardized units; a per-feature affine rescale
moves the band and the truth together, so it does not change coverage.

Run: `python3 -m experiment.calibration --dataset-dir runs/experiment --arch rssm`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiment.abduction import standardize
from experiment.calendar_sim import OBSERVABLE_FEATURES
from experiment.checkpoints import ARCHS, artifact, load_model
from experiment.data import load_daily_trajectory
from experiment.train_world_model import SEED, build_future_arrays

N_SAMPLES = 100
LEVEL = 0.9
CHUNK = 100


def _band(samples: torch.Tensor, level: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Central `level` band over the sample axis of `[n, B, K, F]`."""
    tail = (1.0 - level) / 2.0
    lo = torch.quantile(samples, tail, dim=0)
    hi = torch.quantile(samples, 1.0 - tail, dim=0)
    return lo, hi


def _summarise(inside: torch.Tensor, width: torch.Tensor) -> dict:
    """Aggregate per-cell `[B, K, F]` inside flags and band widths."""
    inside = inside.float()
    return {
        "coverage": float(inside.mean()),
        "coverage_by_day": [float(v) for v in inside.mean(dim=(0, 2))],
        "coverage_by_feature": [float(v) for v in inside.mean(dim=(0, 1))],
        "mean_width_by_day": [float(v) for v in width.mean(dim=(0, 2))],
    }


def interval_coverage(samples: torch.Tensor, truth: torch.Tensor, level: float = LEVEL) -> dict:
    """Coverage of `truth` `[B, K, F]` by the central `level` band of `samples` `[n, B, K, F]`."""
    lo, hi = _band(samples, level)
    return _summarise((truth >= lo) & (truth <= hi), hi - lo)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--arch", choices=ARCHS, default="gru_vae")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--level", type=float, default=LEVEL)
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(SEED)
    model, checkpoint = load_model(base, args.arch)
    horizon = json.loads((base / "windows_meta.json").read_text())["rollout_k"]
    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))
    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")
    future_actions_raw, future_obs_raw = build_future_arrays(test, scenarios, horizon)

    a_mean, a_std = checkpoint["action_mean"], checkpoint["action_std"]
    o_mean, o_std = checkpoint["obs_mean"], checkpoint["obs_std"]
    actions = torch.tensor(standardize(test["actions"], a_mean, a_std))
    observables = torch.tensor(standardize(test["observables"], o_mean, o_std))
    future_actions = torch.tensor(standardize(future_actions_raw, a_mean, a_std))
    truth = torch.tensor(standardize(future_obs_raw, o_mean, o_std))

    inside_parts, width_parts = [], []
    with torch.no_grad():
        for start in range(0, actions.shape[0], CHUNK):
            sl = slice(start, start + CHUNK)
            samples = model.sample_rollouts(
                actions[sl], observables[sl], future_actions[sl], n=args.n_samples,
            )
            lo, hi = _band(samples, args.level)
            inside_parts.append((truth[sl] >= lo) & (truth[sl] <= hi))
            width_parts.append(hi - lo)

    summary = _summarise(torch.cat(inside_parts), torch.cat(width_parts))
    result = {
        "arch": args.arch, "level": args.level, "n_samples": args.n_samples,
        "n_windows": int(actions.shape[0]), "horizon_days": horizon,
        **summary,
        "coverage_by_feature": dict(zip(OBSERVABLE_FEATURES, summary["coverage_by_feature"])),
    }
    results_path = base / artifact(args.arch, "calibration_results.json")
    results_path.write_text(json.dumps(result, indent=2))

    by_day = summary["coverage_by_day"]
    width = summary["mean_width_by_day"]
    print(f"STEP 5 -- {args.level:.0%} interval calibration ({args.arch}, "
          f"{args.n_samples} samples, {result['n_windows']} test windows)")
    print(f"  overall coverage: {summary['coverage']:.1%}   (claimed {args.level:.0%})")
    for day in (0, horizon // 3 - 1, horizon - 1):
        print(f"  day {day + 1:2d}: coverage {by_day[day]:.1%}   mean width {width[day]:.4f}")
    print("  by observable:")
    for name, cov in result["coverage_by_feature"].items():
        print(f"    {name:12s} {cov:.1%}")
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
