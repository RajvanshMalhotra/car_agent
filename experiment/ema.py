"""Exponential moving averages of the observed window, expressed relative to the anchor.

Rounds 8 and 10 both failed the same way: what they injected into the decoder
was outside the training range on an unseen climate. Measured on the dev
split, the share of unseen-climate values outside the training range was 26.6%
for raw values (battery temperature 100%) and 24.9-26.4% for raw EMAs -- an
EMA is a convex mix of that battery's OWN days, and on a hotter climate every
one of those days is hotter than anything trained on.

Differences transfer where levels do not: EMA minus the anchor is 0.1-0.2%
outside the training range, and fast-EMA minus slow-EMA 0.4%. So the features
here are always differences:

    skip_t = [ e_2 - anchor_t, e_7 - anchor_t, e_30 - anchor_t, e_2 - e_30 ]

giving the decoder the level of this battery's recent history relative to the
day being copied, plus its trend, and never an absolute reading.

`ema_state` freezes the EMA at the end of the observed window (rollout);
`ema_series(exclude_current=True)` gives day t's EMA over days strictly
before t (filtering reconstruction, so day t never sees itself).
"""

from __future__ import annotations

import numpy as np

#: Fast, medium and slow. 2 days catches the last trip or two, 30 days spans a
#: layup cycle (the dataset's layup gaps run 20-120 days).
HALF_LIVES: tuple[float, ...] = (2.0, 7.0, 30.0)


def _alpha(half_life: float) -> float:
    return 1.0 - 0.5 ** (1.0 / half_life)


def ema_series(
    observables: np.ndarray, half_life: float, exclude_current: bool = False,
) -> np.ndarray:
    """`[B, L, F]`: the EMA after each window day (or before it).

    Seeded with day 0's value, so a constant window gives exactly that
    constant. With `exclude_current`, day t's entry covers days `< t` and day
    0 falls back to its own value, having no history.
    """
    alpha = _alpha(half_life)
    out = np.zeros_like(observables, dtype=np.float32)
    accumulator = observables[:, 0].astype(np.float32)
    for t in range(observables.shape[1]):
        if exclude_current:
            out[:, t] = accumulator
        accumulator = alpha * observables[:, t] + (1.0 - alpha) * accumulator
        if not exclude_current:
            out[:, t] = accumulator
    return out


def ema_state(observables: np.ndarray, half_life: float) -> np.ndarray:
    """`[B, F]`: the EMA over the whole observed window."""
    return ema_series(observables, half_life)[:, -1]


def _stack(levels: list[np.ndarray], trend: np.ndarray) -> np.ndarray:
    return np.concatenate([*levels, trend], axis=-1).astype(np.float32)


def relative_ema_features(observables: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """`[B, T, F * (len(HALF_LIVES) + 1)]` for rollout: EMA frozen at the window's end.

    `anchors` is `[B, T, F]`; each half-life contributes `ema - anchor_t`, and
    the last block is the fast-minus-slow trend (which the anchor cannot move).
    """
    states = [ema_state(observables, h)[:, None, :] for h in HALF_LIVES]
    levels = [state - anchors for state in states]
    trend = np.broadcast_to(states[0] - states[-1], anchors.shape)
    return _stack(levels, trend)


def relative_ema_features_causal(observables: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """`[B, L, F * (len(HALF_LIVES) + 1)]` for filtering: day t sees only days before t."""
    series = [ema_series(observables, h, exclude_current=True) for h in HALF_LIVES]
    levels = [s - anchors for s in series]
    return _stack(levels, series[0] - series[-1])


def feature_dim(n_obs: int) -> int:
    return n_obs * (len(HALF_LIVES) + 1)
