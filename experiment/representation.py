"""Round 7 representation: anchors, temperature offsets, and development subsets.

Round 6 showed both world models regress toward their training climates on an
unseen 42 C climate, scoring ~30% tolerance accuracy where copying "the most
recent observed day of the same type" scores 83%. Two fixes live here, both
pure numpy so they are tested without torch:

- **Anchors.** `same_type_anchors` is that 83% baseline, per future day:
  the latest window day whose `is_layup` matches the day being predicted
  (the window's last day if that type never occurred). An anchored model
  predicts `anchor + correction`, so learning nothing reproduces the
  baseline. It reads only the observed window and the action sequence being
  asked about -- under a counterfactual schedule the anchors follow the
  counterfactual day types. `window_anchors` is the in-window equivalent
  the RSSM's filtering reconstruction needs: only EARLIER days, so
  reconstructing day t never peeks at day t.
- **Temperature offsets.** Absolute temperatures at 42 C lie outside every
  training climate; the offset above ambient does not (battery mean +1.4 C
  median at 42 C, inside the colder climates' range). `to_offsets` /
  `from_offsets` convert the four temperature observables using the day's
  `ambient_c` action.

`filter_windows` subsets a windows split by scenario while keeping every
per-window array row-aligned, so the round 6 counterfactual targets can be
reused for the round 7 development split (train -10/10 C, 25 C as the
stand-in unseen climate) without rebuilding them.
"""

from __future__ import annotations

import numpy as np

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES

TEMPERATURE_FEATURES: tuple[str, ...] = ("t_bat_mean", "t_bat_max", "t_bay_mean", "t_bay_max")

_TEMP_IDX = [OBSERVABLE_FEATURES.index(f) for f in TEMPERATURE_FEATURES]
_LAYUP = ACTION_FEATURES.index("is_layup")
_AMBIENT = ACTION_FEATURES.index("ambient_c")


def to_offsets(observables: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Temperatures become `value - ambient_c`; everything else is copied."""
    out = observables.astype(np.float32, copy=True)
    out[..., _TEMP_IDX] = out[..., _TEMP_IDX] - actions[..., _AMBIENT:_AMBIENT + 1]
    return out


def from_offsets(observables: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Inverse of `to_offsets`."""
    out = observables.astype(np.float32, copy=True)
    out[..., _TEMP_IDX] = out[..., _TEMP_IDX] + actions[..., _AMBIENT:_AMBIENT + 1]
    return out


def same_type_anchors(
    window_actions: np.ndarray, window_observables: np.ndarray, future_actions: np.ndarray,
) -> np.ndarray:
    """`[B, K, F]`: for each future day, the latest window day of the same type."""
    is_layup = window_actions[..., _LAYUP] >= 0.5
    length = is_layup.shape[1]
    days = np.arange(length)
    last_layup = np.where(is_layup, days, -1).max(axis=1)
    last_drive = np.where(~is_layup, days, -1).max(axis=1)
    future_layup = future_actions[..., _LAYUP] >= 0.5
    source = np.where(future_layup, last_layup[:, None], last_drive[:, None])
    source = np.where(source < 0, length - 1, source)
    index = np.broadcast_to(source[..., None], (*source.shape, window_observables.shape[-1]))
    return np.take_along_axis(window_observables, index, axis=1)


def window_anchors(window_actions: np.ndarray, window_observables: np.ndarray) -> np.ndarray:
    """`[B, L, F]`: for each window day, the latest EARLIER day of the same type.

    Falls back to the previous day, and day 0 anchors on itself.
    """
    is_layup = window_actions[..., _LAYUP] >= 0.5
    batch, length = is_layup.shape
    last_layup = np.full(batch, -1)
    last_drive = np.full(batch, -1)
    source = np.zeros((batch, length), dtype=np.int64)
    for t in range(length):
        current = is_layup[:, t]
        previous_same = np.where(current, last_layup, last_drive)
        source[:, t] = np.where(previous_same >= 0, previous_same, max(t - 1, 0))
        last_layup = np.where(current, t, last_layup)
        last_drive = np.where(~current, t, last_drive)
    index = np.broadcast_to(source[..., None], (*source.shape, window_observables.shape[-1]))
    return np.take_along_axis(window_observables, index, axis=1)


def filter_windows(split: dict[str, np.ndarray], keep_scenarios: set[str]) -> dict[str, np.ndarray]:
    """Rows whose scenario is in `keep_scenarios`, every per-window array aligned."""
    scenario = split["scenario"]
    mask = np.isin(scenario, sorted(keep_scenarios))
    if not mask.any():
        raise ValueError("no window belongs to the requested scenarios")
    n = scenario.shape[0]
    return {
        key: value[mask] if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == n else value
        for key, value in split.items()
    }
