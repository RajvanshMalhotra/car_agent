"""Every caller of rollout must pass what the model in hand actually needs.

Round 10's attention anchor needs the observed window at rollout time. The
training script passed it; experiment/accuracy.py did not, and a finished
100-epoch dev run could not be scored. One helper now answers "what extra
arguments does THIS model need", and every caller uses it.
"""

import pytest

torch = pytest.importorskip("torch")

from experiment.checkpoints import rollout_kwargs  # noqa: E402
from experiment.model import WorldModel  # noqa: E402
from experiment.rssm import RSSM  # noqa: E402

N_ACTION, N_OBS = 4, 10


def _window():
    return torch.randn(2, 6, N_ACTION), torch.randn(2, 6, N_OBS)


def test_plain_models_need_nothing_extra():
    actions, observables = _window()
    assert rollout_kwargs(WorldModel(N_ACTION, N_OBS), actions, observables) == {}
    assert rollout_kwargs(RSSM(N_ACTION, N_OBS), actions, observables) == {}


def test_anchored_model_without_attention_needs_nothing_extra():
    actions, observables = _window()
    assert rollout_kwargs(RSSM(N_ACTION, N_OBS, anchored=True), actions, observables) == {}


def test_attention_model_gets_the_observed_window():
    actions, observables = _window()
    kwargs = rollout_kwargs(RSSM(N_ACTION, N_OBS, anchored=True, attn_anchor=True), actions, observables)
    assert set(kwargs) == {"window_actions", "window_observables"}
    assert kwargs["window_actions"] is actions and kwargs["window_observables"] is observables


def test_the_helper_makes_an_attention_rollout_work():
    torch.manual_seed(0)
    model = RSSM(N_ACTION, N_OBS, anchored=True, attn_anchor=True).eval()
    actions, observables = _window()
    future = torch.randn(2, 5, N_ACTION)
    with torch.no_grad():
        state = model.abduct(actions, observables)
        pred = model.rollout(state, future, observables[:, -1, :],
                             **rollout_kwargs(model, actions, observables))
    assert pred.shape == (2, 5, N_OBS)
