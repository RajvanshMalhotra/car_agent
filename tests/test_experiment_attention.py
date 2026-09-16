"""Round 10: the attention-picked anchor."""

import pytest

torch = pytest.importorskip("torch")

from experiment.attention import AttentionAnchor  # noqa: E402
from experiment.calendar_sim import ACTION_FEATURES  # noqa: E402

LAY = ACTION_FEATURES.index("is_layup")
N_ACTION, N_OBS, L, K = len(ACTION_FEATURES), 10, 8, 4


def _window(layup_days, values):
    actions = torch.zeros(1, len(layup_days), N_ACTION)
    actions[0, :, LAY] = torch.tensor(layup_days, dtype=torch.float32)
    obs = torch.tensor(values, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, N_OBS)
    return actions, obs


def _future(layup_days):
    actions = torch.zeros(1, len(layup_days), N_ACTION)
    actions[0, :, LAY] = torch.tensor(layup_days, dtype=torch.float32)
    return actions


def _anchor(deter=6):
    torch.manual_seed(0)
    return AttentionAnchor(n_action=N_ACTION, n_obs=N_OBS, deter=deter).eval()


def test_untrained_attention_copies_the_latest_same_type_day():
    module = _anchor()
    actions, obs = _window([0, 1, 0, 1, 1, 0, 1, 1], list(range(8)))
    future = _future([0, 1])
    with torch.no_grad():
        weights = module.weights(actions, torch.zeros(1, K, module.deter)[:, :2], future)
    assert weights.shape == (1, 2, L)
    # latest drive day is index 5, latest layup day index 7
    assert weights[0, 0].argmax().item() == 5 and weights[0, 0, 5] > 0.8
    assert weights[0, 1].argmax().item() == 7 and weights[0, 1, 7] > 0.8
    # almost no mass on the opposite type
    assert weights[0, 0, [1, 3, 4, 6, 7]].sum() < 0.01


def test_anchor_is_a_convex_combination_of_observed_days():
    module = _anchor()
    actions, obs = _window([0, 1] * 4, [5.0, -3.0, 11.0, 2.0, 0.0, 7.0, 4.0, 9.0])
    future = _future([0, 1, 0, 1])
    with torch.no_grad():
        weights = module.weights(actions, torch.randn(1, K, module.deter), future)
        anchors = module(actions, obs, torch.randn(1, K, module.deter), future)
    assert torch.allclose(weights.sum(-1), torch.ones(1, K), atol=1e-5)
    assert (weights >= 0).all()
    assert anchors.shape == (1, K, N_OBS)
    assert anchors.max() <= obs.max() + 1e-5 and anchors.min() >= obs.min() - 1e-5


def test_anchor_follows_a_counterfactual_schedule():
    module = _anchor()
    actions, obs = _window([0, 1] * 4, list(range(8)))
    belief = torch.randn(1, 2, module.deter)
    with torch.no_grad():
        drive_first = module(actions, obs, belief, _future([0, 0]))
        layup_first = module(actions, obs, belief, _future([1, 1]))
    assert not torch.allclose(drive_first, layup_first)


def test_window_anchors_are_causal():
    module = _anchor()
    actions, obs = _window([0, 1] * 4, list(range(8)))
    belief = torch.randn(1, L, module.deter)
    with torch.no_grad():
        weights = module.weights(actions, belief, actions, causal=True)
    for t in range(L):
        assert weights[0, t, t:].sum() < 1e-6 or t == 0, f"day {t} attended to itself or later"
    assert torch.allclose(weights[0, 1:].sum(-1), torch.ones(L - 1), atol=1e-5)


def test_changing_a_late_window_value_changes_the_anchor():
    module = _anchor()
    actions, obs = _window([0, 1] * 4, list(range(8)))
    other = obs.clone()
    other[0, 6, :] += 10.0
    belief = torch.randn(1, 2, module.deter)
    with torch.no_grad():
        assert not torch.allclose(
            module(actions, obs, belief, _future([0, 0])),
            module(actions, other, belief, _future([0, 0])),
        )
