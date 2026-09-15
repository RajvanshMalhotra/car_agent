"""Shared loader: `daily_trajectory.csv` into per-scenario numpy arrays.

Used by both `experiment/windows.py` (to build training windows) and
`experiment/abduction.py` (to look up a test scenario's true continuation and
resume state for the counterfactual rerun) so the two never disagree about
how the raw table is laid out.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_daily_trajectory(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Read `daily_trajectory.csv` into `{scenario: {column: array}}`.

    Rows for a given scenario are assumed contiguous and ordered by `day`
    starting at 0 -- true by construction of `experiment/build_dataset.py` --
    so `arr[column][day]` is a valid positional index with no separate lookup.
    """
    buffers: dict[str, dict[str, list]] = {}
    order: list[str] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in reader.fieldnames if c != "scenario"]
        for row in reader:
            name = row["scenario"]
            if name not in buffers:
                buffers[name] = {c: [] for c in columns}
                order.append(name)
            bucket = buffers[name]
            for c in columns:
                bucket[c].append(float(row[c]))

    scenarios = {
        name: {c: np.asarray(vals, dtype=np.float64) for c, vals in cols.items()}
        for name, cols in buffers.items()
    }
    return scenarios, order


def write_daily_trajectory(
    path: Path, scenarios: dict[str, dict[str, np.ndarray]], order: list[str],
) -> None:
    """Inverse of `load_daily_trajectory`: one row per scenario per day, in `order`.

    Values are written with `repr`, which round-trips a float64 exactly, so a
    variant dataset (e.g. `experiment/make_noisy_variant.py`) differs from its
    source only where it was deliberately changed.
    """
    columns = list(scenarios[order[0]])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", *columns])
        for name in order:
            arr = scenarios[name]
            for day in range(len(arr[columns[0]])):
                writer.writerow([name, *(repr(float(arr[c][day])) for c in columns)])
