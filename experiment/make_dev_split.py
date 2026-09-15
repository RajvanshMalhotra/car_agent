#!/usr/bin/env python3
"""Round 7 development split: never tune on the 42 C test.

Builds a dataset directory from round 6's (`runs/experiment_hot_noisy`) where
the models train on `--train-ambients` batteries only and every window from an
`--unseen-ambient` battery becomes the test set. Round 6 already held 42 C
out, so with the defaults (-10/10 train, 25 unseen) this is the same kind of
test -- extrapolating to a hotter climate -- run one step down, where design
choices can be compared freely.

Everything is filtered, not rebuilt:
- train and its precomputed counterfactual targets: kept rows are the
  train-ambient batteries; `filter_windows` keeps the two files row-aligned,
  asserted before writing.
- val: round 6's validation batteries at the train ambients.
- test: every round 6 train or val window from an unseen-ambient battery.
  None of those batteries is in the dev train or val set.

`daily_trajectory.csv`, `scenarios.json`, `daily_dataset.json` and
`noise.json` are symlinked, so the noisy data is identical to round 6's.

Run: `python3 -m experiment.make_dev_split --src runs/experiment_hot_noisy --out runs/r7_dev`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiment.representation import filter_windows


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="runs/experiment_hot_noisy")
    parser.add_argument("--out", default="runs/r7_dev")
    parser.add_argument("--train-ambients", type=float, nargs="+", default=[-10.0, 10.0])
    parser.add_argument("--unseen-ambient", type=float, default=25.0)
    args = parser.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ambient = {s["scenario"]: float(s["ambient_c"]) for s in json.loads((src / "scenarios.json").read_text())}
    train_keep = {n for n, a in ambient.items() if a in args.train_ambients}
    unseen = {n for n, a in ambient.items() if a == args.unseen_ambient}
    assert not (train_keep & unseen)

    train, train_cf, val = _load(src / "windows_train.npz"), _load(src / "windows_train_cf.npz"), _load(src / "windows_val.npz")
    dev_train = filter_windows(train, train_keep)
    dev_cf = filter_windows(train_cf, train_keep)
    assert np.array_equal(dev_train["scenario"], dev_cf["scenario"]) and \
        np.array_equal(dev_train["end_day"], dev_cf["end_day"])
    dev_val = filter_windows(val, train_keep)

    parts = [filter_windows(s, unseen) for s in (train, val) if np.isin(s["scenario"], sorted(unseen)).any()]
    dev_test = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if parts[0][key].ndim > 0}

    np.savez(out / "windows_train.npz", **dev_train)
    np.savez(out / "windows_train_cf.npz", **dev_cf)
    np.savez(out / "windows_val.npz", **dev_val)
    np.savez(out / "windows_test.npz", **dev_test)
    for name in ("daily_trajectory.csv", "scenarios.json", "daily_dataset.json", "noise.json"):
        link = out / name
        if not link.exists():
            link.symlink_to((src / name).resolve())

    meta = json.loads((src / "windows_meta.json").read_text())
    meta.update({
        "dev_split_of": str(src),
        "train_ambients": args.train_ambients, "unseen_ambient": args.unseen_ambient,
        "train_scenarios": sorted(set(dev_train["scenario"].tolist())),
        "val_scenarios": sorted(set(dev_val["scenario"].tolist())),
        "test_scenarios": sorted(set(dev_test["scenario"].tolist())),
        "counts": {k: {"n_windows": int(v["scenario"].shape[0]), "n_scenarios": len(set(v["scenario"].tolist()))}
                   for k, v in (("train", dev_train), ("val", dev_val), ("test", dev_test))},
    })
    meta.pop("target_crystal_histogram", None)
    (out / "windows_meta.json").write_text(json.dumps(meta, indent=2))
    for k, v in meta["counts"].items():
        print(f"{k}: {v['n_windows']} windows from {v['n_scenarios']} batteries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
