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
day's `OBSERVABLE_FEATURES`. Unrolled `K` times in closed loop (feeding its
own predictions back in as `prev_obs`), this is the rollout used both for
TRAINING (`experiment/train_world_model.py`) and for the abduction test in
`experiment/abduction.py`.

**Posterior collapse, and why training is multi-step.** An earlier version
of this file trained only on SINGLE-step prediction, decoder conditioned on
the TRUE previous-day observable. Tomorrow's daily-aggregate state is almost
entirely determined by today's, so the decoder reached a very low
single-step MSE with an EMPTY latent, and the KL penalty then drove
`q(z|window)` to the prior (mu near 0, sigma near 1) -- a posterior collapse.
`experiment/gate2b.py` is the check that catches this. The fix
(`experiment/train_world_model.py`) is to train the SAME closed-loop
`rollout()` this docstring describes: once `prev_obs` is the model's OWN
drifting prediction rather than ground truth, single-step shortcuts stop
being sufficient over a multi-day horizon and the latent has to carry
something for the reconstruction loss to improve. See the experiment report
for the full account of the collapse and the fix.

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
        latent_dim: int = 8, dec_hidden: int = 32, anchored: bool = False,
    ) -> None:
        super().__init__()
        self.n_action = n_action
        self.n_obs = n_obs
        self.latent_dim = latent_dim
        self.dec_hidden = dec_hidden
        #: Round 7: predict `anchor + correction` (see experiment/representation.py).
        self.anchored = anchored

        self.encoder = nn.GRU(
            input_size=n_action + n_obs, hidden_size=enc_hidden, batch_first=True,
        )
        self.to_latent = nn.Linear(enc_hidden, 2 * latent_dim)
        self.z2h = nn.Linear(latent_dim, dec_hidden)
        self.dec_cell = nn.GRUCell(latent_dim + n_action + n_obs, dec_hidden)
        self.out = nn.Linear(dec_hidden, n_obs)
        if anchored:
            # Start exactly at the anchor baseline; training can only add a correction.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def encode(self, actions: torch.Tensor, observables: torch.Tensor):
        """(actions, observables): [B, L, n_action], [B, L, n_obs] -> mu, logvar [B, latent_dim]."""
        x = torch.cat([actions, observables], dim=-1)
        _, h_n = self.encoder(x)
        stats = self.to_latent(h_n[-1])
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def posterior(self, actions: torch.Tensor, observables: torch.Tensor):
        """q(z | window) -- the belief gate2b compares against the prior."""
        return self.encode(actions, observables)

    def abduct(self, actions: torch.Tensor, observables: torch.Tensor) -> torch.Tensor:
        """The posterior mean -- the protocol name `experiment/rssm.py` shares."""
        mu, _logvar = self.encode(actions, observables)
        return mu

    def zero_state(self, batch: int) -> torch.Tensor:
        """The prior mean: the uninformed latent, the interventional control."""
        device = next(self.parameters()).device
        return torch.zeros(batch, self.latent_dim, device=device)

    def sample_rollouts(
        self, actions: torch.Tensor, observables: torch.Tensor,
        future_actions: torch.Tensor, n: int, anchors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`n` futures per window, each from one posterior draw of `z`.

        The rollout itself is deterministic, so all the spread comes from
        that single draw. Returns `[n, B, K, n_obs]`.
        """
        batch = actions.shape[0]
        mu, logvar = self.encode(actions, observables)
        z = self.reparameterize(mu.repeat(n, 1), logvar.repeat(n, 1))
        pred = self.rollout(
            z, future_actions.repeat(n, 1, 1), observables[:, -1, :].repeat(n, 1),
            anchors=None if anchors is None else anchors.repeat(n, 1, 1),
        )
        return pred.view(n, batch, *pred.shape[1:])

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
        anchors: torch.Tensor | None = None,
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
            if anchors is not None:
                # The anchored prediction, not the raw correction, is what feeds back.
                obs_t = anchors[:, t, :] + obs_t
            outputs.append(obs_t)
            prev_obs = obs_t
        return torch.stack(outputs, dim=1)

    def encode_and_rollout(
        self, actions: torch.Tensor, observables: torch.Tensor,
        future_actions: torch.Tensor, sample: bool = True,
        anchors: torch.Tensor | None = None,
    ):
        """Encode the window, then CLOSED-LOOP roll out over `future_actions`.

        This is the training-time call, and it is deliberately the exact
        same `rollout()` path Step 3 evaluates -- see the module docstring
        on why single-step training let the latent go empty. Returns
        (predicted future observables [B, K, n_obs], mu, logvar).
        """
        mu, logvar = self.encode(actions, observables)
        z = self.reparameterize(mu, logvar) if sample else mu
        prev_obs0 = observables[:, -1, :]
        predicted = self.rollout(z, future_actions, prev_obs0, anchors=anchors)
        return predicted, mu, logvar

    def forward(
        self, actions: torch.Tensor, observables: torch.Tensor,
        next_action: torch.Tensor, sample: bool = True,
    ):
        """SINGLE next-day prediction, for the Gate-2 diagnostic only.

        Kept separate from training (see `encode_and_rollout`) because this
        is exactly the objective that collapsed the posterior the first
        time -- it is reported for comparison, never trained against.
        Returns (predicted next observable, mu, logvar).
        """
        mu, logvar = self.encode(actions, observables)
        z = self.reparameterize(mu, logvar) if sample else mu
        h = self.init_decoder_state(z)
        prev_obs = observables[:, -1, :]
        obs_next, _ = self.decode_step(z, next_action, prev_obs, h)
        return obs_next, mu, logvar


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Plain KL(q(z|x) || N(0,I)), summed over dims, averaged over the batch.

    Kept for reference/comparison -- see `kl_free_bits` below for what
    training actually uses. Applying this WITHOUT a floor is exactly what
    let the encoder drive every dimension's KL to ~0 last time: nothing
    stops the optimizer from collapsing a dimension all the way once the
    reconstruction loss no longer needs it.
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def kl_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.15) -> torch.Tensor:
    """Per-dimension KL, floored at `free_bits` nats (Kingma et al. 2016).

    `torch.clamp(..., min=free_bits)` has zero gradient below the floor, so
    once a dimension's raw KL drops to `free_bits` the optimizer has no
    further incentive to push it toward the prior -- collapse to mu=0,
    sigma=1 on that dimension becomes impossible regardless of how long
    training runs or how large `KL_WEIGHT` is. Chosen over (or alongside) KL
    annealing because it is a hard, permanent floor rather than a schedule
    that could still let the model collapse after the anneal completes.
    Summed over the latent dimensions (each contributing at least
    `free_bits`), then averaged over the batch.
    """
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    kl_floored = torch.clamp(kl_per_dim, min=free_bits)
    return kl_floored.sum(dim=-1).mean()
