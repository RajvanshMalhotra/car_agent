# RSSM world model, head to head with the frozen-latent model

**Date:** 2026-09-15
**Status:** agreed in conversation, being implemented

## Why

The current world model (`experiment/model.py`, "gru_vae") squeezes a 60-day
window into ONE latent `z` that stays frozen for the whole rollout. Sulfation
is not frozen -- it accumulates day by day. On the held-out abduction test it
beats the last-day-only control only 53.6% of the time, loses significantly
in the lowest crystal bin, and its counterfactual error is 2x its factual
error. Its uncertainty is drawn once, so its spread cannot widen with
horizon, which the calibrated-interval output contract needs.

A Recurrent State-Space Model (Hafner et al., PlaNet 2019 / Dreamer) keeps a
belief state that is updated every day, filtered against each day's
measurement, and imagined forward with a learned prior.

## Model (`experiment/rssm.py`)

Per day `t`, with action `a_t` (that day's schedule) and observable `o_t`
(that day's aggregates):

    h_t      = GRUCell([s_{t-1}, a_t], h_{t-1})         deterministic path
    prior    p(s_t | h_t)        = N(mu_p, sigma_p)     imagination
    posterior q(s_t | h_t, o_t)  = N(mu_q, sigma_q)     filtering (abduction)
    o_t_hat  = MLP([h_t, s_t])

- **Abduct** = filter over the 60-day window; the belief state is `[h_60, s_60]`.
- **Act / predict** = imagine 30 days from that state under any action
  sequence, using prior means (or prior samples for intervals).

Sized to roughly match the gru_vae's ~11k parameters so the comparison is
architecture, not capacity: `deter=32`, `stoch=8`, MLP hidden 32.

## Training (`experiment/train_rssm.py`)

Same data, same split, same standardisation, same counterfactual curriculum
as gru_vae, so the only difference is the architecture:

    loss = window reconstruction (every day, from the posterior state)
         + KL_WEIGHT * balanced KL(q || p) per day, free bits per day
         + 30-day imagined rollout MSE, factual actions
         + 30-day imagined rollout MSE, randomized counterfactual actions

KL balancing (alpha 0.8, DreamerV2) trains the prior toward the posterior
faster than the reverse, which is what makes imagination usable.

## Evaluation

The existing gate2b, abduction and SAE-probe scripts gain `--arch
{gru_vae,rssm}`. gru_vae artifacts keep their current filenames; rssm writes
`rssm_*` next to them, so nothing already reported is overwritten. A shared
protocol lets one script drive both models:

    model.abduct(actions, observables)          -> state
    model.rollout(state, actions, prev_obs0)    -> predictions
    model.zero_state(batch)                     -> uninformed state
    model.sample_rollouts(actions, observables, future_actions, n) -> samples

New: `experiment/calibration.py` -- 90% interval coverage of the factual
30-day rollout on the test set, by horizon day, for both models.

## Success criteria (decided before running)

| Test | gru_vae | RSSM must reach |
|---|---|---|
| Win rate vs last-day control | 53.6% | clearly higher, significant |
| Crystal bin [0.0, 0.2) | loses (39%) | no longer significantly loses |
| Counterfactual / factual error | 1.9x | smaller ratio |
| 90% interval coverage | not measured | near 90% |

If RSSM does not beat these, that is reported as the result, and it points at
the data (one recorded drive, reused) rather than the architecture.

Hyperparameters are fixed before the test-set run. Any retuning after seeing
test results is reported as such.
