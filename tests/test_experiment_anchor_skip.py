"""Round 8: residual skip connections injecting the anchor into every RSSM decoder layer."""

import pytest

torch = pytest.importorskip("torch")

from experiment.checkpoints import build_model, model_config  # noqa: E402
from experiment.rssm import RSSM  # noqa: E402

N_ACTION, N_OBS, WINDOW, HORIZON = 4, 10, 6, 5


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, WINDOW, N_ACTION, generator=g),
        torch.randn(2, WINDOW, N_OBS, generator=g),
        torch.randn(2, HORIZON, N_ACTION, generator=g),
        torch.randn(2, HORIZON, N_OBS, generator=g),
    )


def _model(**kwargs):
    torch.manual_seed(0)
    return RSSM(N_ACTION, N_OBS, anchored=True, **kwargs).eval()


def _randomise_head(model):
    with torch.no_grad():
        for p in model.decoder_head.parameters():
            p.normal_(0.0, 0.5)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_skip_decoder_starts_exactly_at_the_anchor(layers):
    model = _model(decoder_layers=layers, anchor_skip=True)
    actions, obs, future, anchors = _inputs()
    with torch.no_grad():
        pred = model.rollout(model.abduct(actions, obs), future, anchors=anchors)
    torch.testing.assert_close(pred, anchors)


def test_with_skips_the_correction_depends_on_the_anchor():
    model = _model(decoder_layers=2, anchor_skip=True)
    _randomise_head(model)
    actions, obs, future, anchors = _inputs()
    with torch.no_grad():
        state = model.abduct(actions, obs)
        correction_a = model.rollout(state, future, anchors=anchors) - anchors
        correction_b = model.rollout(state, future, anchors=anchors + 5.0) - (anchors + 5.0)
    assert not torch.allclose(correction_a, correction_b)


def test_without_skips_the_correction_ignores_the_anchor():
    model = _model(decoder_layers=2, anchor_skip=False)
    _randomise_head(model)
    actions, obs, future, anchors = _inputs()
    with torch.no_grad():
        state = model.abduct(actions, obs)
        correction_a = model.rollout(state, future, anchors=anchors) - anchors
        correction_b = model.rollout(state, future, anchors=anchors + 5.0) - (anchors + 5.0)
    torch.testing.assert_close(correction_a, correction_b)


def test_filtering_reconstruction_uses_the_skips_too():
    model = _model(decoder_layers=2, anchor_skip=True)
    _randomise_head(model)
    actions, obs, _, _ = _inputs()
    window_anchors = torch.randn(2, WINDOW, N_OBS)
    with torch.no_grad():
        a = model.filter(actions, obs, sample=False, anchors=window_anchors)["recon"] - window_anchors
        b = model.filter(actions, obs, sample=False, anchors=window_anchors + 5.0)["recon"] - (window_anchors + 5.0)
    assert not torch.allclose(a, b)


def test_skips_add_one_projection_per_hidden_layer():
    plain = _model(decoder_layers=2, anchor_skip=False)
    skip = _model(decoder_layers=2, anchor_skip=True)
    extra = sum(p.numel() for p in skip.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * (N_OBS * skip.hidden + skip.hidden)


def test_default_decoder_is_the_round7_decoder():
    # Same parameter names and count as before, so round 7 checkpoints load unchanged.
    torch.manual_seed(0)
    model = RSSM(N_ACTION, N_OBS, anchored=True)
    assert model.decoder_layers == 1 and model.anchor_skip is False
    assert sum(p.numel() for p in model.parameters()) == 10914


def test_anchor_skip_requires_anchoring():
    with pytest.raises(ValueError):
        RSSM(N_ACTION, N_OBS, anchored=False, anchor_skip=True)


def test_checkpoint_round_trips_skip_options():
    model = _model(decoder_layers=2, anchor_skip=True)
    rebuilt = build_model({"state_dict": model.state_dict(), **model_config(model)})
    assert rebuilt.decoder_layers == 2 and rebuilt.anchor_skip is True
    actions, obs, future, anchors = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            rebuilt.rollout(rebuilt.abduct(actions, obs), future, anchors=anchors),
            model.rollout(model.abduct(actions, obs), future, anchors=anchors),
        )
