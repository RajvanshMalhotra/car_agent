"""A small stochastic-latent sequence world model over daily observation windows.

Encoder: a single-layer GRU reads the `WINDOW_LEN`-day window of
`[action_t, observable_t]` pairs and produces a Gaussian posterior
`q(z | window) = N(mu, diag(exp(logvar)))` over an 8-dimensional latent --
the ABDUCTED summary of the window, including whatever it captured about the
hidden crystal state.

Decoder: a `GRUCell` that steps forward one calendar day at a time. At each
step it takes the (fixed, abducted) latent `z`, the day's action
(`ACTION_FEATURES` -- exogenous, supplied by whatever schedule is being asked
about), and the previous day's observable vector, and predicts the next
day's `OBSERVABLE_FEATURES`. Unrolled once, this is the Gate-2 next-step
predictor; unrolled `K` times in closed loop (feeding its own predictions
back in as `prev_obs`), this is the rollout the abduction test in
`experiment/abduction.py` uses to answer "what would the next `K` days have
looked like under a different action sequence".

Deliberately small (~11k parameters): the training set is a few thousand
windows drawn from 64 scenarios, not the kind of corpus that justifies a
large sequence model, and the brief's own instruction is "keep it small".
"""

from __future__ import annotations

import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(
        self, n_action: int, n_obs: int, enc_hidden: int = 32,
        latent_dim: int = 8, dec_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.n_action = n_action
        self.n_obs = n_obs
        self.latent_dim = latent_dim
        self.dec_hidden = dec_hidden

        self.encoder = nn.GRU(
            input_size=n_action + n_obs, hidden_size=enc_hidden, batch_first=True,
        )
        self.to_latent = nn.Linear(enc_hidden, 2 * latent_dim)
        self.z2h = nn.Linear(latent_dim, dec_hidden)
        self.dec_cell = nn.GRUCell(latent_dim + n_action + n_obs, dec_hidden)
        self.out = nn.Linear(dec_hidden, n_obs)

    def encode(self, actions: torch.Tensor, observables: torch.Tensor):
        """(actions, observables): [B, L, n_action], [B, L, n_obs] -> mu, logvar [B, latent_dim]."""
        x = torch.cat([actions, observables], dim=-1)
        _, h_n = self.encoder(x)
        stats = self.to_latent(h_n[-1])
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def init_decoder_state(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.z2h(z))

    def decode_step(
        self, z: torch.Tensor, action_t: torch.Tensor, prev_obs: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decoder step. Returns (predicted observable, new decoder hidden)."""
        inp = torch.cat([z, action_t, prev_obs], dim=-1)
        h = self.dec_cell(inp, h)
        obs_t = self.out(h)
        return obs_t, h

    def rollout(
        self, z: torch.Tensor, action_seq: torch.Tensor, prev_obs0: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-loop multi-day rollout under a GIVEN action sequence.

        `action_seq`: [B, K, n_action] -- the sequence being "acted" under,
        possibly counterfactual. `prev_obs0`: [B, n_obs] -- the last TRUE
        observable before the rollout starts (the window's final day, or a
        held-out continuation's day 0). Returns predicted observables
        [B, K, n_obs], each step's input `prev_obs` being the MODEL's own
        prior prediction, not ground truth -- this is what makes it a
        rollout rather than a teacher-forced one-step evaluation.
        """
        batch, horizon, _ = action_seq.shape
        h = self.init_decoder_state(z)
        prev_obs = prev_obs0
        outputs = []
        for t in range(horizon):
            obs_t, h = self.decode_step(z, action_seq[:, t, :], prev_obs, h)
            outputs.append(obs_t)
            prev_obs = obs_t
        return torch.stack(outputs, dim=1)

    def forward(
        self, actions: torch.Tensor, observables: torch.Tensor,
        next_action: torch.Tensor, sample: bool = True,
    ):
        """One training step: encode the window, predict the SINGLE next day.

        Returns (predicted next observable, mu, logvar).
        """
        mu, logvar = self.encode(actions, observables)
        z = self.reparameterize(mu, logvar) if sample else mu
        h = self.init_decoder_state(z)
        prev_obs = observables[:, -1, :]
        obs_next, _ = self.decode_step(z, next_action, prev_obs, h)
        return obs_next, mu, logvar


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
