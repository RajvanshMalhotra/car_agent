"""experiment/pipeline.py: one path from a windows split to model arrays and back."""

import numpy as np
import pytest

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.pipeline import prepare, real_units, standardize_prepared

LAY = ACTION_FEATURES.index("is_layup")
AMB = ACTION_FEATURES.index("ambient_c")
T_BAT = OBSERVABLE_FEATURES.index("t_bat_mean")
V_MEAN = OBSERVABLE_FEATURES.index("v_mean")
L, K, A, F = 6, 4, len(ACTION_FEATURES), len(OBSERVABLE_FEATURES)


def _fixture():
    """One scenario, 20 days; one window ending on day 9 (so days 10-13 are the future)."""
    days = 20
    arr = {f: np.zeros(days) for f in (*ACTION_FEATURES, *OBSERVABLE_FEATURES)}
    arr["ambient_c"][:] = 30.0
    arr["is_layup"][:] = [0, 1] * 10
    arr["t_bat_mean"][:] = 30.0 + np.arange(days)  # offset above ambient = day index
    arr["v_mean"][:] = 12.0 + 0.01 * np.arange(days)
    end = 9
    split = {
        "scenario": np.array(["s"]), "end_day": np.array([end], np.int32),
        "actions": np.stack([arr[f][end - L + 1:end + 1] for f in ACTION_FEATURES], -1)[None].astype(np.float32),
        "observables": np.stack([arr[f][end - L + 1:end + 1] for f in OBSERVABLE_FEATURES], -1)[None].astype(np.float32),
    }
    cf = {
        "cf_actions": np.stack([arr[f][end + 1:end + 1 + K] for f in ACTION_FEATURES], -1)[None].astype(np.float32),
        "cf_observables": np.zeros((1, K, F), np.float32),
    }
    cf["cf_actions"][0, :, LAY] = 1.0  # every counterfactual day is a layup
    return {"s": arr}, split, cf


def test_future_arrays_are_the_days_after_the_window():
    scenarios, split, _ = _fixture()
    out = prepare(split, scenarios, K, offsets=False)
    assert out["future_observables"][0, :, V_MEAN] == pytest.approx(12.0 + 0.01 * np.arange(10, 14))


def test_offsets_subtract_each_days_own_ambient_from_temperatures_only():
    scenarios, split, _ = _fixture()
    out = prepare(split, scenarios, K, offsets=True)
    assert out["observables"][0, :, T_BAT] == pytest.approx(np.arange(4, 10))
    assert out["future_observables"][0, :, T_BAT] == pytest.approx(np.arange(10, 14))
    assert out["observables"][0, :, V_MEAN] == pytest.approx(12.0 + 0.01 * np.arange(4, 10))


def test_future_anchors_follow_day_type_in_the_prepared_representation():
    scenarios, split, _ = _fixture()
    out = prepare(split, scenarios, K, offsets=True)
    # window days 4..9 alternate drive/layup: last drive = day 8, last layup = day 9.
    # future days 10..13 are drive, layup, drive, layup.
    assert out["future_anchors"][0, :, T_BAT] == pytest.approx([8, 9, 8, 9])
    assert out["window_anchors"].shape == (1, L, F)


def test_counterfactual_anchors_follow_the_counterfactual_schedule():
    scenarios, split, cf = _fixture()
    out = prepare(split, scenarios, K, offsets=True, cf=cf)
    assert out["cf_anchors"][0, :, T_BAT] == pytest.approx([9, 9, 9, 9])
    assert out["cf_observables"].shape == (1, K, F)


def test_standardize_uses_action_stats_for_actions_and_obs_stats_for_the_rest():
    scenarios, split, cf = _fixture()
    out = prepare(split, scenarios, K, offsets=False, cf=cf)
    a_mean, a_std = np.full(A, 1.0, np.float32), np.full(A, 2.0, np.float32)
    o_mean, o_std = np.full(F, 3.0, np.float32), np.full(F, 4.0, np.float32)
    std = standardize_prepared(out, a_mean, a_std, o_mean, o_std)
    for key in ("actions", "future_actions", "cf_actions"):
        np.testing.assert_allclose(std[key], (out[key] - 1.0) / 2.0, rtol=1e-6)
    for key in ("observables", "future_observables", "future_anchors", "window_anchors",
                "cf_observables", "cf_anchors"):
        np.testing.assert_allclose(std[key], (out[key] - 3.0) / 4.0, rtol=1e-6)


def test_real_units_undoes_standardization_and_offsets():
    scenarios, split, _ = _fixture()
    out = prepare(split, scenarios, K, offsets=True)
    o_mean, o_std = np.full(F, 3.0, np.float32), np.full(F, 4.0, np.float32)
    std = (out["future_observables"] - o_mean) / o_std
    back = real_units(std, out["future_actions"], o_mean, o_std, offsets=True)
    assert back[0, :, T_BAT] == pytest.approx(30.0 + np.arange(10, 14), abs=1e-4)
    assert back[0, :, V_MEAN] == pytest.approx(12.0 + 0.01 * np.arange(10, 14), abs=1e-5)
