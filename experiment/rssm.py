"""A Recurrent State-Space Model (Hafner et al. 2019, PlaNet; Dreamer) over daily windows.

The alternative to `experiment/model.py`'s WorldModel ("gru_vae"), which
squeezes a whole 60-day window into ONE latent `z` that stays frozen for the
entire rollout. Sulfation is not frozen -- it builds up day by day -- so this
model keeps a belief state that is updated every calendar day:

    h_t       = GRUCell([s_{t-1}, a_t], h_{t-1})      deterministic path
    prior     p(s_t | h_t)       = N(mu_p, sigma_p)   used to IMAGINE forward
    posterior q(s_t | h_t, o_t)  = N(mu_q, sigma_q)   used to FILTER (abduct)
    o_hat_t   = decoder([h_t, s_t])

`a_t` is day t's schedule and `o_t` that day's aggregates, so the action
enters the transition INTO day t (the driving on day t produces day t's
observables).

Abduction is filtering: run the posterior over the observed window and keep
the final `[h, s]`. Counterfactual prediction is imagination: step the prior
forward from that state under whatever action sequence is being asked about.
Because a fresh `s_t` is drawn every imagined day, sampled rollouts spread
wider with horizon, which a single frozen `z` cannot do.

Speaks the same protocol as WorldModel -- `abduct`, `rollout`, `zero_state`,
`sample_rollouts` -- so gate2b, abduction, the SAE probe and calibration
drive both models through one code path. `rollout` accepts `prev_obs0` for
that reason and ignores it: an RSSM decodes from its state, not from the
previous observable.

Sized (deter 36, stoch 8, hidden 32: 10,914 parameters against the gru_vae's
11,130) so a difference between them is architecture, not capacity.
"""

from __future__ import annotations

import torch
from torch import nn

from experiment.attention import AttentionAnchor
from experiment.ema import feature_dim, relative_ema_features, relative_ema_features_causal

#: Keeps a freshly initialised or badly trained head from producing an
#: exp(logvar) that overflows or underflows in the KL.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


def _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the last dimension."""
    return 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    ).sum(dim=-1)


def balanced_kl(
    post_mu: torch.Tensor, post_logvar: torch.Tensor,
    prior_mu: torch.Tensor, prior_logvar: torch.Tensor,
    alpha: float = 0.8, free_bits: float = 1.0,
) -> torch.Tensor:
    """DreamerV2 KL balancing, floored at `free_bits` nats per day.

    `alpha` of the gradient moves the PRIOR toward the posterior and
    `1 - alpha` moves the posterior toward the prior. Imagination runs on the
    prior alone, so it has to learn to predict what filtering will find;
    pulling the posterior toward an untrained prior instead would just empty
    the state. The per-day floor has zero gradient below it, the same
    anti-collapse role `experiment/model.py:kl_free_bits` plays for gru_vae.
    Inputs are `[B, T, stoch]`; returns a scalar.
    """
    kl_prior = _gaussian_kl(post_mu.detach(), post_logvar.detach(), prior_mu, prior_logvar)
    kl_post = _gaussian_kl(post_mu, post_logvar, prior_mu.detach(), prior_logvar.detach())
    kl = alpha * kl_prior + (1.0 - alpha) * kl_post
    return torch.clamp(kl, min=free_bits).mean()


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, n_out))


class RSSM(nn.Module):
    def __init__(
        self, n_action: int, n_obs: int, deter: int = 36, stoch: int = 8,
        hidden: int = 32, anchored: bool = False,
        decoder_layers: int = 1, anchor_skip: bool = False, attn_anchor: bool = False,
        ema_skip: bool = False,
    ) -> None:
        super().__init__()
        if anchor_skip and not anchored:
            raise ValueError("anchor_skip injects the anchor, so it requires anchored=True")
        if attn_anchor and not anchored:
            raise ValueError("attn_anchor picks the anchor, so it requires anchored=True")
        if ema_skip and not anchored:
            raise ValueError("ema_skip is relative to the anchor, so it requires anchored=True")
        if ema_skip and anchor_skip:
            raise ValueError("ema_skip and anchor_skip both drive the decoder skips; pick one")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be at least 1")
        #: Round 8: decoder depth, and residual skips projecting the anchor into
        #: every hidden layer. The defaults are the round 5-7 decoder exactly.
        self.decoder_layers = decoder_layers
        self.anchor_skip = anchor_skip
        #: Round 10: the anchor is attention over the observed window
        #: (experiment/attention.py), not the fixed same-day-type rule.
        self.attn_anchor = attn_anchor
        #: Round 11: the decoder skips carry stacked EMAs of the window
        #: RELATIVE to the anchor (experiment/ema.py), never absolute levels.
        self.ema_skip = ema_skip
        self.n_action = n_action
        self.n_obs = n_obs
        self.deter = deter
        self.stoch = stoch
        self.hidden = hidden
        self.state_dim = deter + stoch
        #: Round 7: decode `anchor + correction` (see experiment/representation.py).
        self.anchored = anchored

        self.cell = nn.GRUCell(stoch + n_action, deter)
        self.prior_head = _mlp(deter, hidden, 2 * stoch)
        self.post_head = _mlp(deter + n_obs, hidden, 2 * stoch)
        #: Width of whatever drives the decoder skips: the anchor itself
        #: (round 8) or the stacked relative EMAs (round 11).
        self.skip_dim = feature_dim(n_obs) if ema_skip else n_obs
        if decoder_layers == 1 and not (anchor_skip or ema_skip):
            # Round 5-7 decoder, same parameter names, so their checkpoints load unchanged.
            self.decoder = _mlp(deter + stoch, hidden, n_obs)
        else:
            self.dec_layers = nn.ModuleList(
                nn.Linear(deter + stoch if l == 0 else hidden, hidden) for l in range(decoder_layers)
            )
            if anchor_skip or ema_skip:
                self.dec_skips = nn.ModuleList(
                    nn.Linear(self.skip_dim, hidden) for _ in range(decoder_layers)
                )
            self.dec_head = nn.Linear(hidden, n_obs)
        if attn_anchor:
            self.attn = AttentionAnchor(n_action=n_action, n_obs=n_obs, deter=deter)
        if anchored:
            # Start exactly at the anchor baseline; training can only add a correction.
            nn.init.zeros_(self.decoder_head.weight)
            nn.init.zeros_(self.decoder_head.bias)

    @property
    def decoder_head(self) -> nn.Linear:
        """The final projection to observables, whichever decoder layout is in use."""
        return self.decoder[-1] if hasattr(self, "decoder") else self.dec_head

    def _decode(
        self, h: torch.Tensor, s: torch.Tensor, anchor: torch.Tensor | None,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One day's observables from `[h, s]`; `anchor + correction` when anchored.

        With `anchor_skip`, every hidden layer also receives a linear projection
        of the anchor -- `x_l = ELU(W_l x_{l-1} + P_l anchor)` -- so the
        correction can depend on the value it corrects.
        """
        x = torch.cat([h, s], dim=-1)
        if hasattr(self, "decoder"):
            out = self.decoder(x)
        else:
            if self.anchor_skip:
                if anchor is None:
                    raise ValueError("an anchor_skip decoder needs anchors")
                skip = anchor
            elif self.ema_skip and skip is None:
                raise ValueError("an ema_skip decoder needs its EMA features")
            for l, layer in enumerate(self.dec_layers):
                z = layer(x)
                if skip is not None:
                    z = z + self.dec_skips[l](skip)
                x = nn.functional.elu(z)
            out = self.dec_head(x)
        return out if anchor is None else anchor + out

    @staticmethod
    def _split(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    @staticmethod
    def _draw(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def zero_state(self, batch: int) -> torch.Tensor:
        """The uninformed belief: h = 0, s = 0. The interventional control."""
        device = next(self.parameters()).device
        return torch.zeros(batch, self.state_dim, device=device)

    def filter(
        self, actions: torch.Tensor, observables: torch.Tensor, sample: bool = True,
        anchors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the posterior over a window. `[B, L, *]` in, per-day tensors out.

        Returns prior/posterior statistics `[B, L, stoch]`, the reconstruction
        of each day from its posterior state `[B, L, n_obs]`, and the final
        belief `state` `[B, state_dim]`.
        """
        batch, days, _ = actions.shape
        if self.attn_anchor and anchors is None:
            anchors = "attend"  # computed per day below, causally
        ema_features = None
        if self.ema_skip and torch.is_tensor(anchors):
            # Day t's EMA covers days strictly before t, like window_anchors.
            ema_features = torch.tensor(
                relative_ema_features_causal(
                    observables.detach().cpu().numpy(), anchors.detach().cpu().numpy(),
                ), device=observables.device,
            )
        state = self.zero_state(batch)
        h, s = state[:, :self.deter], state[:, self.deter:]
        prior_mu, prior_logvar, post_mu, post_logvar, recon = [], [], [], [], []
        for t in range(days):
            h = self.cell(torch.cat([s, actions[:, t, :]], dim=-1), h)
            p_mu, p_logvar = self._split(self.prior_head(h))
            q_mu, q_logvar = self._split(self.post_head(torch.cat([h, observables[:, t, :]], dim=-1)))
            s = self._draw(q_mu, q_logvar, sample)
            prior_mu.append(p_mu)
            prior_logvar.append(p_logvar)
            post_mu.append(q_mu)
            post_logvar.append(q_logvar)
            if isinstance(anchors, str):
                # Attention anchor, causal: day t may only copy days before it.
                upto = max(t, 1)
                anchor_t = self.attn(
                    actions[:, :upto, :], observables[:, :upto, :],
                    h.unsqueeze(1), actions[:, t:t + 1, :],
                )[:, 0, :]
                recon.append(self._decode(h, s, anchor_t))
            elif not ((self.anchor_skip or self.ema_skip) and anchors is None):
                # Abduction filters without anchors; the belief never depends on the
                # reconstruction, so a skip decoder simply has none to produce.
                recon.append(self._decode(
                    h, s, None if anchors is None else anchors[:, t, :],
                    skip=None if ema_features is None else ema_features[:, t, :],
                ))
        return {
            "prior_mu": torch.stack(prior_mu, dim=1),
            "prior_logvar": torch.stack(prior_logvar, dim=1),
            "post_mu": torch.stack(post_mu, dim=1),
            "post_logvar": torch.stack(post_logvar, dim=1),
            "recon": torch.stack(recon, dim=1) if recon else None,
            "state": torch.cat([h, s], dim=-1),
        }

    def posterior(self, actions: torch.Tensor, observables: torch.Tensor):
        """q(s_L | ...) on the window's final day -- what gate2b compares to the prior."""
        out = self.filter(actions, observables, sample=False)
        return out["post_mu"][:, -1, :], out["post_logvar"][:, -1, :]

    def abduct(self, actions: torch.Tensor, observables: torch.Tensor) -> torch.Tensor:
        """Posterior-mean filtering over the window; the belief it ends on."""
        return self.filter(actions, observables, sample=False)["state"]

    def rollout(
        self, state: torch.Tensor, action_seq: torch.Tensor,
        prev_obs0: torch.Tensor | None = None, sample: bool = False,
        anchors: torch.Tensor | None = None,
        window_actions: torch.Tensor | None = None,
        window_observables: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Imagine `K` days from `state` under `action_seq` `[B, K, n_action]`.

        Prior means by default, so a rollout is a single deterministic
        prediction; `sample=True` draws each day's `s_t` instead.
        `prev_obs0` is accepted for protocol parity with WorldModel and unused.
        """
        if self.attn_anchor:
            if window_actions is None or window_observables is None:
                raise ValueError("an attn_anchor rollout needs the observed window")
            anchors = None  # picked per day below
        ema_features = None
        if self.ema_skip:
            if window_observables is None:
                raise ValueError("an ema_skip rollout needs the observed window")
            if anchors is None:
                raise ValueError("an ema_skip rollout needs anchors to be relative to")
            # EMA frozen at the window's end: never fed the model's own predictions.
            ema_features = torch.tensor(
                relative_ema_features(
                    window_observables.detach().cpu().numpy(), anchors.detach().cpu().numpy(),
                ), device=anchors.device,
            )
        h, s = state[:, :self.deter], state[:, self.deter:]
        outputs = []
        for t in range(action_seq.shape[1]):
            h = self.cell(torch.cat([s, action_seq[:, t, :]], dim=-1), h)
            mu, logvar = self._split(self.prior_head(h))
            s = self._draw(mu, logvar, sample)
            if self.attn_anchor:
                anchor_t = self.attn(
                    window_actions, window_observables, h.unsqueeze(1), action_seq[:, t:t + 1, :],
                )[:, 0, :]
            else:
                anchor_t = None if anchors is None else anchors[:, t, :]
            outputs.append(self._decode(
                h, s, anchor_t, skip=None if ema_features is None else ema_features[:, t, :],
            ))
        return torch.stack(outputs, dim=1)

    def sample_rollouts(
        self, actions: torch.Tensor, observables: torch.Tensor,
        future_actions: torch.Tensor, n: int, anchors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`n` stochastic futures per window: sampled filtering, sampled imagination.

        Returns `[n, B, K, n_obs]`.
        """
        batch = actions.shape[0]
        window_a, window_o = actions.repeat(n, 1, 1), observables.repeat(n, 1, 1)
        post = self.filter(window_a, window_o, sample=True)
        needs_window = self.attn_anchor or self.ema_skip
        pred = self.rollout(
            post["state"], future_actions.repeat(n, 1, 1), sample=True,
            anchors=None if anchors is None else anchors.repeat(n, 1, 1),
            window_actions=window_a if needs_window else None,
            window_observables=window_o if needs_window else None,
        )
        return pred.view(n, batch, *pred.shape[1:])
