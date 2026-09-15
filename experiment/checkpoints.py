"""Save and load either world model, and keep their output files apart.

gru_vae (`experiment/model.py`) was the only architecture when its artifacts
were written, so it keeps the original filenames (`world_model.pt`,
`abduction_results.json`, ...) and its checkpoints carry no `arch` key.
Everything the RSSM (`experiment/rssm.py`) produces is prefixed `rssm_`,
so running the evaluation scripts on it never overwrites a reported result.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiment.model import WorldModel
from experiment.rssm import RSSM

ARCHS = ("gru_vae", "rssm")


def artifact(arch: str, name: str) -> str:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")
    return name if arch == "gru_vae" else f"{arch}_{name}"


def model_config(model: nn.Module) -> dict:
    """The constructor arguments a checkpoint needs to rebuild `model`."""
    if isinstance(model, RSSM):
        return {
            "arch": "rssm", "n_action": model.n_action, "n_obs": model.n_obs,
            "deter": model.deter, "stoch": model.stoch, "hidden": model.hidden,
            "anchored": model.anchored,
        }
    return {
        "arch": "gru_vae", "n_action": model.n_action, "n_obs": model.n_obs,
        "enc_hidden": model.encoder.hidden_size, "latent_dim": model.latent_dim,
        "dec_hidden": model.dec_hidden, "anchored": model.anchored,
    }


def build_model(checkpoint: dict) -> nn.Module:
    arch = checkpoint.get("arch", "gru_vae")
    if arch == "rssm":
        model = RSSM(
            n_action=checkpoint["n_action"], n_obs=checkpoint["n_obs"],
            deter=checkpoint["deter"], stoch=checkpoint["stoch"], hidden=checkpoint["hidden"],
            anchored=checkpoint.get("anchored", False),
        )
    else:
        model = WorldModel(
            n_action=checkpoint["n_action"], n_obs=checkpoint["n_obs"],
            enc_hidden=checkpoint["enc_hidden"], latent_dim=checkpoint["latent_dim"],
            dec_hidden=checkpoint["dec_hidden"], anchored=checkpoint.get("anchored", False),
        )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


class BestCheckpoint:
    """Keeps the weights from the epoch with the lowest validation loss.

    Round 5's RSSM went unstable late and its LAST epoch was evaluated. Both
    training scripts now select by validation loss instead, under one rule.
    """

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_loss = float("inf")
        self._state: dict | None = None

    def update(self, epoch: int, loss: float, model: nn.Module) -> bool:
        if loss >= self.best_loss:
            return False
        self.best_epoch, self.best_loss = epoch, loss
        self._state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return True

    def restore(self, model: nn.Module) -> None:
        if self._state is not None:
            model.load_state_dict(self._state)


def validation_rollout_mse(model: nn.Module, split: dict, device) -> float:
    """Posterior-mean abduction, deterministic 30-day rollout, factual continuation.

    Same quantity as Gate 2's model error, on the validation batteries.
    Leaves the model in eval mode; both training loops call `model.train()`
    at the start of every epoch.
    """
    model.eval()
    with torch.no_grad():
        observables = split["observables"].to(device)
        state = model.abduct(split["actions"].to(device), observables)
        # Prepared splits always carry anchors; only an anchored model may use them.
        anchors = split.get("future_anchors") if getattr(model, "anchored", False) else None
        pred = model.rollout(
            state, split["future_actions"].to(device), observables[:, -1, :],
            anchors=None if anchors is None else anchors.to(device),
        )
        return float(((pred - split["future_observables"].to(device)) ** 2).mean())


def load_model(base: Path, arch: str) -> tuple[nn.Module, dict]:
    # weights_only=False: the checkpoint carries numpy normalization arrays
    # alongside the state_dict, and it is always our own file written by a
    # training script in this package -- never untrusted.
    checkpoint = torch.load(
        base / artifact(arch, "world_model.pt"), map_location="cpu", weights_only=False,
    )
    return build_model(checkpoint), checkpoint
