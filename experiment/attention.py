"""An anchor the model picks by attention, instead of by a fixed rule.

Round 7's anchor is "the latest window day whose drive/layup type matches the
day being predicted". That rule alone scores 83.3% on the unseen 42 C test,
and the network only corrects it. Round 8 then showed that feeding the
anchor's RAW values into hidden layers destroys generalization: on an unseen
climate every injected temperature is outside the training range.

This module keeps the anchor in range BY CONSTRUCTION. Attention over the
observed window produces weights that are non-negative and sum to one, and
the anchor is that convex combination of days the model actually saw:

    score_i  = (W_q [action_t, h_t]) . (W_k action_i) / sqrt(d)
             + beta  * 1[type_i == type_t]        type-match bonus
             - gamma * (L - 1 - i)                recency penalty, per day ago
    anchor_t = sum_i softmax(score)_i * observable_i

so no anchor value can fall outside that battery's own recent history,
whatever the climate.

`W_q` and `W_k` start at zero and `beta`/`gamma` start large, so an
untrained module puts almost all its weight on the latest same-type day --
exactly round 7's anchor. With the RSSM's zero-initialised head, round 10
therefore STARTS at round 7 and can only refine which days it copies.

`causal=True` restricts each query day to strictly earlier window days (day 0
attends to itself), which is what filtering reconstruction needs so that
reconstructing day t never peeks at day t.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from experiment.calendar_sim import ACTION_FEATURES

_LAYUP = ACTION_FEATURES.index("is_layup")

#: Initial type-match bonus and per-day-ago recency penalty (both learnable).
#: The recency term is per DAY, not spread over the window, so the module
#: behaves the same whatever `L` is: a wrong-type day costs `BETA_INIT` and a
#: day one step older costs `GAMMA_INIT`, so with these values an untrained
#: module is the same-day-type rule (the latest matching day takes ~98% of the
#: weight, a wrong-type day ~1e-9).
BETA_INIT = 20.0
GAMMA_INIT = 2.0


class AttentionAnchor(nn.Module):
    def __init__(self, n_action: int, n_obs: int, deter: int, dim: int = 16) -> None:
        super().__init__()
        self.n_action = n_action
        self.n_obs = n_obs
        self.deter = deter
        self.dim = dim
        self.to_query = nn.Linear(n_action + deter, dim)
        self.to_key = nn.Linear(n_action, dim)
        nn.init.zeros_(self.to_query.weight)
        nn.init.zeros_(self.to_query.bias)
        nn.init.zeros_(self.to_key.weight)
        nn.init.zeros_(self.to_key.bias)
        self.beta = nn.Parameter(torch.tensor(BETA_INIT))
        self.gamma = nn.Parameter(torch.tensor(GAMMA_INIT))

    def weights(
        self, window_actions: torch.Tensor, belief: torch.Tensor,
        query_actions: torch.Tensor, causal: bool = False,
    ) -> torch.Tensor:
        """`[B, T, L]` attention over window days for each queried day."""
        length = window_actions.shape[1]
        query = self.to_query(torch.cat([query_actions, belief], dim=-1))
        key = self.to_key(window_actions)
        scores = query @ key.transpose(1, 2) / math.sqrt(self.dim)

        window_layup = (window_actions[..., _LAYUP] >= 0.5).unsqueeze(1)
        query_layup = (query_actions[..., _LAYUP] >= 0.5).unsqueeze(2)
        scores = scores + self.beta * (window_layup == query_layup).to(scores.dtype)

        days = torch.arange(length, device=scores.device, dtype=scores.dtype)
        scores = scores - self.gamma * (length - 1 - days)

        if causal:
            # Day t sees days < t; day 0 has no history and sees itself.
            steps = torch.arange(scores.shape[1], device=scores.device)
            allowed = days.unsqueeze(0) < steps.unsqueeze(1).to(days.dtype)
            allowed[0, 0] = True
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(
        self, window_actions: torch.Tensor, window_observables: torch.Tensor,
        belief: torch.Tensor, query_actions: torch.Tensor, causal: bool = False,
    ) -> torch.Tensor:
        """`[B, T, n_obs]`: each queried day's anchor, a convex mix of observed days."""
        return self.weights(window_actions, belief, query_actions, causal) @ window_observables
