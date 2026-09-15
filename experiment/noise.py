"""Sensor noise on the daily observables, following `cell/sensors.py`'s rules.

`experiment/calendar_sim.py` emits exact ODE aggregates. A real logger's
daily summaries would not be exact, so round 6 adds noise before the models
see anything -- and, as `cell/sensors.py` insists, ONLY on channels a real
car measures. Battery voltage, current and case temperature get noise.
State of charge and engine-bay temperature are latent there, so they stay
clean here too, as do every hidden and resume field.

Each daily column is perturbed independently at the per-sample sigma of its
sensor. That is a simplification: white noise would shrink when averaged
over a day, but calibration offset and drift do not, and those dominate a
real logger's daily error. Current noise scales with the reading
(`cell/sensors.py`: 0.5% with a 0.1 A floor); `i_mean_abs` is clamped at 0
because an absolute value cannot be negative.

Deterministic given `(seed, order)`: one generator walks scenarios in `order`
and columns in `DAILY_NOISE_CHANNEL` order.
"""

from __future__ import annotations

import numpy as np

from cell.sensors import NOISE

#: Daily observable column -> the `cell/sensors.py` channel whose sigma it uses.
DAILY_NOISE_CHANNEL: dict[str, str] = {
    "v_mean": "v_bat_v", "v_min": "v_bat_v", "v_max": "v_bat_v",
    "i_mean": "i_bat_a", "i_mean_abs": "i_bat_a",
    "t_bat_mean": "t_bat_c", "t_bat_max": "t_bat_c",
}

NON_NEGATIVE = frozenset({"i_mean_abs"})


def _sigma(channel: str, values: np.ndarray) -> np.ndarray:
    spec = NOISE[channel]["sigma"]
    if callable(spec):
        return np.array([spec(float(v)) for v in values])
    return np.full(values.shape, float(spec))


def add_daily_noise(
    scenarios: dict[str, dict[str, np.ndarray]], order: list[str], seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """A noisy copy of `scenarios`; the input is left untouched."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in order:
        arr = {column: values.copy() for column, values in scenarios[name].items()}
        for column, channel in DAILY_NOISE_CHANNEL.items():
            if column not in arr:
                continue
            clean = arr[column]
            noisy = clean + rng.normal(0.0, 1.0, clean.shape) * _sigma(channel, clean)
            if column in NON_NEGATIVE:
                noisy = np.maximum(noisy, 0.0)
            arr[column] = noisy
        out[name] = arr
    return out
