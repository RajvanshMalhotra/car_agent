"""The RSSM world model and the protocol both world models share."""

import pytest

torch = pytest.importorskip("torch")

from experiment.model import WorldModel  # noqa: E402
from experiment.rssm import RSSM, balanced_kl  # noqa: E402

N_ACTION, N_OBS, WINDOW, HORIZON = 4, 10, 12, 7


def _batch(batch=3, days=WINDOW, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, days, N_ACTION, generator=g),
        torch.randn(batch, days, N_OBS, generator=g),
    )


def _rssm():
    torch.manual_seed(0)
    return RSSM(n_action=N_ACTION, n_obs=N_OBS).eval()


def test_filter_returns_per_day_prior_posterior_and_reconstruction():
    model = _rssm()
    actions, observables = _batch()
    out = model.filter(actions, observables)
    for key in ("prior_mu", "prior_logvar", "post_mu", "post_logvar"):
        assert out[key].shape == (3, WINDOW, model.stoch)
    assert out["recon"].shape == (3, WINDOW, N_OBS)
    assert out["state"].shape == (3, model.state_dim)


def test_abducted_state_depends_on_early_history_not_only_the_last_day():
    model = _rssm()
    actions, observables = _batch()
    changed = observables.clone()
    changed[:, 0, :] += 5.0  # only day 1 differs
    with torch.no_grad():
        a = model.abduct(actions, observables)
        b = model.abduct(actions, changed)
    assert not torch.allclose(a, b)


def test_rollout_depends_on_the_action_sequence():
    model = _rssm()
    actions, observables = _batch()
    future_a, _ = _batch(days=HORIZON, seed=1)
    with torch.no_grad():
        state = model.abduct(actions, observables)
        prev = observables[:, -1, :]
        p1 = model.rollout(state, future_a, prev)
        p2 = model.rollout(state, future_a + 3.0, prev)
    assert p1.shape == (3, HORIZON, N_OBS)
    assert not torch.allclose(p1, p2)


def test_rollout_is_deterministic_by_default():
    model = _rssm()
    actions, observables = _batch()
    future_a, _ = _batch(days=HORIZON, seed=1)
    with torch.no_grad():
        state = model.abduct(actions, observables)
        p1 = model.rollout(state, future_a, observables[:, -1, :])
        p2 = model.rollout(state, future_a, observables[:, -1, :])
    assert torch.equal(p1, p2)


def test_zero_state_matches_abducted_state_shape():
    model = _rssm()
    actions, observables = _batch()
    with torch.no_grad():
        assert model.zero_state(3).shape == model.abduct(actions, observables).shape


def _final_day_spread(prior_logvar_bias):
    model = _rssm()
    with torch.no_grad():
        last = model.prior_head[-1]
        last.weight.zero_()
        last.bias.zero_()
        last.bias[model.stoch:] = prior_logvar_bias
    actions, observables = _batch(batch=2)
    future_a, _ = _batch(batch=2, days=HORIZON, seed=1)
    torch.manual_seed(1)
    with torch.no_grad():
        samples = model.sample_rollouts(actions, observables, future_a, n=64)
    assert samples.shape == (64, 2, HORIZON, N_OBS)
    return samples.std(dim=0)[:, -1, :].mean()


def test_imagination_draws_fresh_noise_every_day():
    # Same posterior noise both times; only the imagined prior's width changes.
    # A frozen-latent model would show no difference here.
    assert _final_day_spread(0.0) > 1.5 * _final_day_spread(-8.0)


def test_posterior_is_the_final_day_belief_for_both_models():
    actions, observables = _batch()
    torch.manual_seed(0)
    for model, dim in ((_rssm(), 8), (WorldModel(n_action=N_ACTION, n_obs=N_OBS).eval(), 8)):
        with torch.no_grad():
            mu, logvar = model.posterior(actions, observables)
        assert mu.shape == logvar.shape == (3, dim)


@pytest.mark.parametrize("arch", ["gru_vae", "rssm"])
def test_checkpoint_round_trip_rebuilds_the_right_architecture(arch):
    from experiment.checkpoints import build_model, model_config

    torch.manual_seed(0)
    model = RSSM(N_ACTION, N_OBS) if arch == "rssm" else WorldModel(N_ACTION, N_OBS)
    checkpoint = {"state_dict": model.state_dict(), **model_config(model)}
    rebuilt = build_model(checkpoint)
    assert type(rebuilt) is type(model)
    actions, observables = _batch()
    with torch.no_grad():
        assert torch.equal(rebuilt.abduct(actions, observables), model.eval().abduct(actions, observables))


def test_checkpoint_without_arch_is_the_original_gru_vae():
    from experiment.checkpoints import build_model

    model = WorldModel(N_ACTION, N_OBS)
    legacy = {
        "state_dict": model.state_dict(), "n_action": N_ACTION, "n_obs": N_OBS,
        "enc_hidden": 32, "latent_dim": 8, "dec_hidden": 32,
    }
    assert isinstance(build_model(legacy), WorldModel)


def test_interval_coverage_counts_truth_inside_the_sampled_band():
    from experiment.calibration import interval_coverage

    # 101 samples per cell spread evenly over 0..100: the 90% band is ~[5, 95].
    samples = torch.arange(101.0).view(101, 1, 1, 1).expand(101, 2, 3, 1)
    truth = torch.tensor([[[50.0], [50.0], [99.0]], [[50.0], [2.0], [50.0]]])
    result = interval_coverage(samples, truth, level=0.9)
    assert result["coverage_by_day"] == pytest.approx([1.0, 0.5, 0.5])
    assert result["coverage"] == pytest.approx(4 / 6)
    assert result["mean_width_by_day"][0] == pytest.approx(90.0)


def test_rssm_artifacts_never_overwrite_the_gru_vae_ones():
    from experiment.checkpoints import artifact

    assert artifact("gru_vae", "abduction_results.json") == "abduction_results.json"
    assert artifact("rssm", "abduction_results.json") == "rssm_abduction_results.json"
    with pytest.raises(ValueError):
        artifact("transformer", "world_model.pt")


def test_balanced_kl_is_zero_when_prior_equals_posterior_and_floored():
    mu = torch.zeros(2, 5, 8)
    logvar = torch.zeros(2, 5, 8)
    assert balanced_kl(mu, logvar, mu, logvar, free_bits=0.0).item() == pytest.approx(0.0)
    assert balanced_kl(mu, logvar, mu, logvar, free_bits=1.0).item() == pytest.approx(1.0)


def test_balanced_kl_positive_when_posterior_differs():
    post_mu = torch.ones(2, 5, 8)
    zeros = torch.zeros(2, 5, 8)
    assert balanced_kl(post_mu, zeros, zeros, zeros, free_bits=0.0).item() > 0.0


def test_gru_vae_speaks_the_same_protocol():
    torch.manual_seed(0)
    model = WorldModel(n_action=N_ACTION, n_obs=N_OBS).eval()
    actions, observables = _batch()
    future_a, _ = _batch(days=HORIZON, seed=1)
    with torch.no_grad():
        state = model.abduct(actions, observables)
        assert state.shape == model.zero_state(3).shape
        pred = model.rollout(state, future_a, observables[:, -1, :])
        samples = model.sample_rollouts(actions, observables, future_a, n=5)
    assert pred.shape == (3, HORIZON, N_OBS)
    assert samples.shape == (5, 3, HORIZON, N_OBS)
