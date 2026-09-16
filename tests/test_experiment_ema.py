"""Round 11: stacked relative-EMA features for the decoder skip."""

import numpy as np
import pytest

from experiment.calendar_sim import OBSERVABLE_FEATURES
from experiment.ema import HALF_LIVES, ema_series, ema_state, relative_ema_features

N_OBS = len(OBSERVABLE_FEATURES)


def _window(values):
    """[1, L, N_OBS] with every feature equal to the given per-day series."""
    return np.repeat(np.asarray(values, np.float32)[None, :, None], N_OBS, axis=2)


def test_ema_of_a_constant_series_is_that_constant():
    state = ema_state(_window([7.0] * 20), half_life=7.0)
    assert state.shape == (1, N_OBS)
    np.testing.assert_allclose(state, 7.0, rtol=0, atol=1e-5)


def test_half_life_covers_half_the_gap_after_that_many_days():
    # Starts at 0, then a step to 1: after `h` days the EMA sits near 0.5.
    h = 5
    series = _window([0.0] + [1.0] * h)
    np.testing.assert_allclose(ema_state(series, half_life=float(h))[0, 0], 0.5, atol=0.02)


def test_ema_series_is_causal_and_starts_at_day_zero():
    series = _window([0.0, 10.0, 10.0, 10.0])
    per_day = ema_series(series, half_life=2.0, exclude_current=True)
    assert per_day.shape == (1, 4, N_OBS)
    # Day 0 has no history: it falls back to its own value.
    assert per_day[0, 0, 0] == pytest.approx(0.0)
    # Day 1 may only see day 0.
    assert per_day[0, 1, 0] == pytest.approx(0.0)
    # Day 2 sees days 0-1, so it has moved toward 10 but not reached it.
    assert 0.0 < per_day[0, 2, 0] < 10.0


def test_features_are_relative_to_the_anchor_and_include_a_trend():
    window = _window(list(range(30)))
    anchors = np.full((1, 4, N_OBS), 5.0, np.float32)
    feats = relative_ema_features(window, anchors)
    assert feats.shape == (1, 4, N_OBS * (len(HALF_LIVES) + 1))
    fast = ema_state(window, HALF_LIVES[0])[0, 0]
    slow = ema_state(window, HALF_LIVES[-1])[0, 0]
    assert feats[0, 0, 0] == pytest.approx(fast - 5.0, abs=1e-4)
    assert feats[0, 0, N_OBS * len(HALF_LIVES)] == pytest.approx(fast - slow, abs=1e-4)


def test_shifting_the_anchor_shifts_the_features_by_the_same_amount():
    window = _window(list(range(30)))
    anchors = np.zeros((1, 3, N_OBS), np.float32)
    base = relative_ema_features(window, anchors)
    moved = relative_ema_features(window, anchors + 4.0)
    level_part = N_OBS * len(HALF_LIVES)
    np.testing.assert_allclose(moved[..., :level_part], base[..., :level_part] - 4.0, atol=1e-4)
    # The trend feature is a difference of two EMAs, so the anchor cannot touch it.
    np.testing.assert_allclose(moved[..., level_part:], base[..., level_part:], atol=1e-6)


def test_hotter_history_leaves_the_relative_features_unchanged():
    """A battery 6 C hotter throughout looks identical in relative terms."""
    window = _window(list(range(30)))
    anchors = np.full((1, 2, N_OBS), 20.0, np.float32)
    cool = relative_ema_features(window, anchors)
    hot = relative_ema_features(window + 6.0, anchors + 6.0)
    np.testing.assert_allclose(hot, cool, atol=1e-4)
