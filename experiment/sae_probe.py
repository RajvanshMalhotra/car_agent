#!/usr/bin/env python3
"""Step 4: a sparse autoencoder over the world model's latents, then a probe.

Encodes every training window's posterior mean `mu` (the abducted latent --
`experiment.model.WorldModel.encode`'s output, same quantity `abduction.py`
uses) and trains a small overcomplete sparse autoencoder (L1 penalty on a
ReLU code, a standard Anthropic-style SAE, no tied weights needed at this
scale) on top of it.

The point of this step is a COMPARISON, not the SAE's reconstruction
quality: does any single sparse feature track the TRUE, hidden `crystal`
state better than the best single feature tracks an OBSERVABLE the model
could trivially have encoded instead (state of charge, battery temperature,
corrosion-hours -- corrosion-hours is technically hidden too, but it is a
near-linear function of ambient and elapsed time, i.e. cheap to encode,
unlike the path-dependent `crystal`). If the top crystal-tracking feature is
no better than the top SoC-tracking feature, the most likely reading is that
the model encoded the easy, instantaneously-available observable and not the
genuinely hidden state.

Run: `python3 -m experiment.sae_probe --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.abduction import standardize
from experiment.model import WorldModel

L1_WEIGHT = 1e-3
EXPANSION = 4  # overcomplete factor relative to the latent dimension
EPOCHS = 200
LR = 1e-3
SEED = 0


class SparseAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, code_dim: int) -> None:
        super().__init__()
        self.enc = nn.Linear(latent_dim, code_dim)
        self.dec = nn.Linear(code_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = torch.relu(self.enc(x))
        recon = self.dec(code)
        return recon, code


def load_model(base: Path):
    # weights_only=False -- see experiment/abduction.py's load_model for why.
    checkpoint = torch.load(base / "world_model.pt", map_location="cpu", weights_only=False)
    model = WorldModel(
        n_action=checkpoint["n_action"], n_obs=checkpoint["n_obs"],
        enc_hidden=checkpoint["enc_hidden"], latent_dim=checkpoint["latent_dim"],
        dec_hidden=checkpoint["dec_hidden"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def encode_all(model: WorldModel, checkpoint: dict, split: dict) -> np.ndarray:
    with torch.no_grad():
        actions = torch.tensor(
            standardize(split["actions"], checkpoint["action_mean"], checkpoint["action_std"]),
        )
        observables = torch.tensor(
            standardize(split["observables"], checkpoint["obs_mean"], checkpoint["obs_std"]),
        )
        mu, _logvar = model.encode(actions, observables)
    return mu.numpy()


def best_feature_correlation(codes: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    """The single sparse-code column with the highest |Pearson r| against target."""
    best_idx, best_r = -1, 0.0
    target_c = target - target.mean()
    target_norm = np.sqrt((target_c ** 2).sum())
    if target_norm < 1e-12:
        return best_idx, best_r
    for j in range(codes.shape[1]):
        col = codes[:, j]
        if col.std() < 1e-12:
            continue
        col_c = col - col.mean()
        denom = np.sqrt((col_c ** 2).sum()) * target_norm
        if denom < 1e-12:
            continue
        r = float((col_c * target_c).sum() / denom)
        if abs(r) > abs(best_r):
            best_idx, best_r = j, r
    return best_idx, best_r


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, checkpoint = load_model(base)
    train = dict(np.load(base / "windows_train.npz", allow_pickle=True))
    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))

    train_latents = encode_all(model, checkpoint, train)
    test_latents = encode_all(model, checkpoint, test)

    latent_mean = train_latents.mean(axis=0, keepdims=True)
    latent_std = train_latents.std(axis=0, keepdims=True)
    latent_std = np.where(latent_std < 1e-8, 1.0, latent_std)
    train_norm = (train_latents - latent_mean) / latent_std
    test_norm = (test_latents - latent_mean) / latent_std

    latent_dim = train_latents.shape[1]
    code_dim = latent_dim * EXPANSION

    sae = SparseAutoencoder(latent_dim, code_dim)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    x_train = torch.tensor(train_norm, dtype=torch.float32)

    history = []
    for epoch in range(EPOCHS):
        sae.train()
        recon, code = sae(x_train)
        recon_loss = ((recon - x_train) ** 2).mean()
        l1_loss = code.abs().mean()
        loss = recon_loss + L1_WEIGHT * l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            history.append({
                "epoch": epoch, "recon_mse": recon_loss.item(), "l1": l1_loss.item(),
            })
            print(f"epoch {epoch:3d}  recon_mse={recon_loss.item():.4f}  "
                  f"l1={l1_loss.item():.4f}", flush=True)

    sae.eval()
    with torch.no_grad():
        _recon, code_test = sae(torch.tensor(test_norm, dtype=torch.float32))
    codes = code_test.numpy()

    active_per_example = (codes > 1e-6).sum(axis=1)
    sparsity = {
        "code_dim": code_dim,
        "mean_active_features_per_example": float(active_per_example.mean()),
        "fraction_active": float(active_per_example.mean() / code_dim),
        "dead_features": int((codes.max(axis=0) < 1e-6).sum()),
    }

    targets = {
        "crystal": test["target_crystal"],
        "soc": test["target_soc"],
        "t_bat_mean_window_last_day": test["observables"][:, -1, 5],  # t_bat_mean index
        "corrosion_hours": test["target_corrosion_hours"],
    }
    probe = {}
    for name, target in targets.items():
        idx, r = best_feature_correlation(codes, target)
        probe[name] = {"best_feature_index": idx, "abs_pearson_r": abs(r), "pearson_r": r}

    crystal_r = probe["crystal"]["abs_pearson_r"]
    soc_r = probe["soc"]["abs_pearson_r"]
    verdict = (
        "The top crystal-tracking feature CLEARLY beats the top SoC-tracking "
        "feature: some sparse feature captured something about the hidden "
        "state beyond what SoC alone would give away."
        if crystal_r > soc_r + 0.05 else
        "The top crystal-tracking feature is NO BETTER than the top "
        "SoC-tracking feature: the SAE most likely surfaced an easily-encoded "
        "OBSERVABLE proxy, not evidence that the model represents the "
        "genuinely hidden, path-dependent crystal state."
    )

    result = {
        "latent_dim": latent_dim, "code_dim": code_dim,
        "sparsity": sparsity, "probe": probe, "verdict": verdict,
        "history": history,
    }
    (base / "sae_probe_results.json").write_text(json.dumps(result, indent=2))

    print("\nSTEP 4 -- sparse autoencoder probe (test-set latents):")
    print(f"  code_dim={code_dim}  mean active/example="
          f"{sparsity['mean_active_features_per_example']:.2f} "
          f"({sparsity['fraction_active']:.1%})  dead_features={sparsity['dead_features']}")
    for name, info in probe.items():
        print(f"  best |r| for {name:28s}: {info['abs_pearson_r']:.4f} "
              f"(feature #{info['best_feature_index']})")
    print(f"\n  {verdict}")
    print(f"\nwrote {base / 'sae_probe_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
