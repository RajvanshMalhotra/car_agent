"""Round 7: anchored residual prediction, temperature offsets, dev subsets."""

import numpy as np
import pytest

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.representation import (
    TEMPERATURE_FEATURES,
    filter_windows,
    from_offsets,
    same_type_anchors,
    to_offsets,
    window_anchors,
)

LAY = ACTION_FEATURES.index("is_layup")
AMB = ACTION_FEATURES.index("ambient_c")
N_OBS = len(OBSERVABLE_FEATURES)


def _window(layup_days, obs_values):
    """One window: day types and a per-day scalar broadcast over every observable."""
    days = len(layup_days)
    actions = np.zeros((1, days, len(ACTION_FEATURES)), np.float32)
    actions[0, :, LAY] = layup_days
    obs = np.repeat(np.asarray(obs_values, np.float32)[None, :, None], N_OBS, axis=2)
    return actions, obs


def test_anchor_is_the_latest_window_day_of_the_same_type():
    actions, obs = _window([0, 1, 0, 1, 1], [10, 20, 30, 40, 50])
    future = np.zeros((1, 3, len(ACTION_FEATURES)), np.float32)
    future[0, :, LAY] = [0, 1, 0]
    anchors = same_type_anchors(actions, obs, future)
    assert anchors.shape == (1, 3, N_OBS)
    assert anchors[0, :, 0].tolist() == [30, 50, 30]


def test_anchor_falls_back_to_the_last_day_when_the_type_never_occurred():
    actions, obs = _window([0, 0, 0], [1, 2, 3])
    future = np.zeros((1, 2, len(ACTION_FEATURES)), np.float32)
    future[0, :, LAY] = [1, 0]
    assert same_type_anchors(actions, obs, future)[0, :, 0].tolist() == [3, 3]


def test_window_anchor_uses_only_earlier_days():
    actions, obs = _window([0, 1, 0, 0, 1], [10, 20, 30, 40, 50])
    anchors = window_anchors(actions, obs)
    # day 0 has no earlier day: itself. day1 layup, none earlier: previous day.
    assert anchors[0, :, 0].tolist() == [10, 10, 10, 30, 20]


def test_temperature_offsets_round_trip_and_touch_only_temperatures():
    rng = np.random.default_rng(0)
    obs = rng.normal(20, 5, (2, 4, N_OBS)).astype(np.float32)
    actions = np.zeros((2, 4, len(ACTION_FEATURES)), np.float32)
    actions[..., AMB] = 42.0
    off = to_offsets(obs, actions)
    for j, name in enumerate(OBSERVABLE_FEATURES):
        expected = obs[..., j] - 42.0 if name in TEMPERATURE_FEATURES else obs[..., j]
        np.testing.assert_allclose(off[..., j], expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(from_offsets(off, actions), obs, rtol=0, atol=1e-4)
    assert set(TEMPERATURE_FEATURES) == {"t_bat_mean", "t_bat_max", "t_bay_mean", "t_bay_max"}


def test_filter_windows_keeps_rows_aligned_across_arrays():
    split = {
        "scenario": np.array(["a", "b", "a", "c"]),
        "end_day": np.array([1, 2, 3, 4], np.int32),
        "actions": np.arange(4)[:, None, None] * np.ones((4, 2, 3)),
        "seed": np.array(11),  # scalar entries pass through untouched
    }
    out = filter_windows(split, {"a", "c"})
    assert out["scenario"].tolist() == ["a", "a", "c"]
    assert out["end_day"].tolist() == [1, 3, 4]
    assert out["actions"][:, 0, 0].tolist() == [0, 2, 3]
    assert int(out["seed"]) == 11


def test_filter_windows_rejects_an_empty_result():
    with pytest.raises(ValueError):
        filter_windows({"scenario": np.array(["a"])}, {"z"})


@pytest.mark.parametrize("arch", ["gru_vae", "rssm"])
def test_zeroed_anchored_model_reproduces_the_anchor(arch):
    torch = pytest.importorskip("torch")
    from experiment.model import WorldModel
    from experiment.rssm import RSSM

    torch.manual_seed(0)
    model = (RSSM(4, N_OBS, anchored=True) if arch == "rssm" else WorldModel(4, N_OBS, anchored=True)).eval()
    actions = torch.randn(2, 6, 4)
    obs = torch.randn(2, 6, N_OBS)
    future = torch.randn(2, 5, 4)
    anchors = torch.randn(2, 5, N_OBS)
    with torch.no_grad():
        state = model.abduct(actions, obs)
        pred = model.rollout(state, future, obs[:, -1, :], anchors=anchors)
    torch.testing.assert_close(pred, anchors)


@pytest.mark.parametrize("arch", ["gru_vae", "rssm"])
def test_unanchored_models_still_reject_nothing_and_ignore_no_anchor(arch):
    torch = pytest.importorskip("torch")
    from experiment.model import WorldModel
    from experiment.rssm import RSSM

    torch.manual_seed(0)
    model = (RSSM(4, N_OBS) if arch == "rssm" else WorldModel(4, N_OBS)).eval()
    actions, obs, future = torch.randn(2, 6, 4), torch.randn(2, 6, N_OBS), torch.randn(2, 5, 4)
    with torch.no_grad():
        state = model.abduct(actions, obs)
        a = model.rollout(state, future, obs[:, -1, :])
        b = model.rollout(state, future, obs[:, -1, :], anchors=None)
    torch.testing.assert_close(a, b)


def test_anchored_checkpoint_round_trips():
    torch = pytest.importorskip("torch")
    from experiment.checkpoints import build_model, model_config
    from experiment.rssm import RSSM

    model = RSSM(4, N_OBS, anchored=True)
    rebuilt = build_model({"state_dict": model.state_dict(), **model_config(model)})
    assert rebuilt.anchored is True
