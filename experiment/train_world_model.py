#!/usr/bin/env python3
"""Step 2: train the stochastic-latent world model, then run Gate 2.

Trains `experiment.model.WorldModel` to predict the SINGLE next day's
`OBSERVABLE_FEATURES` from a 60-day window plus the next day's action, with a
standard VAE-style KL penalty on the latent. Never sees `crystal`,
`corrosion_hours`, `shedding`, or `soh` -- see `experiment/windows.py` for
what the window tensors actually contain.

**Gate 2.** Reports next-step prediction error (mean squared error over the
standardized observable vector) for:
  (a) the model
  (b) persistence -- predict the window's last observed day, unchanged
  (c) the training-set mean -- predict the same constant vector every time
If the model does not clearly beat (a) < (b) and (a) < (c), the brief says
stop: a model that has not learned the dynamics cannot be meaningfully
probed for abduction. This script reports the comparison and exits either
way; it does not decide to keep going, the caller does after reading Gate 2.

Run: `python3 -m experiment.train_world_model --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.model import WorldModel, kl_to_standard_normal

KL_WEIGHT = 0.01
LATENT_DIM = 8
ENC_HIDDEN = 32
DEC_HIDDEN = 32
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
SEED = 0


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def load_split(base: Path, split: str) -> dict[str, np.ndarray]:
    return dict(np.load(base / f"windows_{split}.npz", allow_pickle=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train = load_split(base, "train")
    test = load_split(base, "test")

    action_mean, action_std = _standardize_fit(train["actions"])
    obs_mean, obs_std = _standardize_fit(train["observables"])

    def prep(d: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        actions = _apply(d["actions"], action_mean, action_std)
        observables = _apply(d["observables"], obs_mean, obs_std)
        next_action = _apply(d["next_action"], action_mean, action_std)
        next_observable = _apply(d["next_observable"], obs_mean, obs_std)
        return {
            "actions": torch.tensor(actions, dtype=torch.float32),
            "observables": torch.tensor(observables, dtype=torch.float32),
            "next_action": torch.tensor(next_action, dtype=torch.float32),
            "next_observable": torch.tensor(next_observable, dtype=torch.float32),
        }

    train_t = prep(train)
    test_t = prep(test)

    n_action = train["actions"].shape[-1]
    n_obs = train["observables"].shape[-1]
    device = torch.device(args.device)

    model = WorldModel(
        n_action=n_action, n_obs=n_obs, enc_hidden=ENC_HIDDEN,
        latent_dim=LATENT_DIM, dec_hidden=DEC_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    n_train = train_t["actions"].shape[0]
    rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n_train)
        epoch_recon, epoch_kl, n_batches = 0.0, 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            actions = train_t["actions"][idx].to(device)
            observables = train_t["observables"][idx].to(device)
            next_action = train_t["next_action"][idx].to(device)
            next_observable = train_t["next_observable"][idx].to(device)

            pred, mu, logvar = model(actions, observables, next_action, sample=True)
            recon = mse(pred, next_observable)
            kl = kl_to_standard_normal(mu, logvar)
            loss = recon + KL_WEIGHT * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1
        history.append({
            "epoch": epoch, "recon_mse": epoch_recon / n_batches,
            "kl": epoch_kl / n_batches,
        })
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  recon_mse={history[-1]['recon_mse']:.4f}  "
                  f"kl={history[-1]['kl']:.4f}", flush=True)

    # ---- Gate 2: next-step prediction error vs persistence and mean -------
    model.eval()
    with torch.no_grad():
        actions = test_t["actions"].to(device)
        observables = test_t["observables"].to(device)
        next_action = test_t["next_action"].to(device)
        next_observable = test_t["next_observable"].to(device)

        pred, mu, logvar = model(actions, observables, next_action, sample=False)
        model_mse = mse(pred, next_observable).item()

        persistence_pred = observables[:, -1, :]
        persistence_mse = mse(persistence_pred, next_observable).item()

        train_mean_obs = train_t["next_observable"].mean(dim=0, keepdim=True)
        mean_pred = train_mean_obs.expand_as(next_observable).to(device)
        mean_mse = mse(mean_pred, next_observable).item()

        # Per-feature breakdown, standardized units.
        per_feature_model = ((pred - next_observable) ** 2).mean(dim=0)
        per_feature_persist = ((persistence_pred - next_observable) ** 2).mean(dim=0)

    from experiment.calendar_sim import OBSERVABLE_FEATURES
    per_feature = {
        name: {
            "model_mse": float(per_feature_model[j]),
            "persistence_mse": float(per_feature_persist[j]),
        }
        for j, name in enumerate(OBSERVABLE_FEATURES)
    }

    gate2 = {
        "model_mse": model_mse,
        "persistence_mse": persistence_mse,
        "mean_mse": mean_mse,
        "model_beats_persistence": model_mse < persistence_mse,
        "model_beats_mean": model_mse < mean_mse,
        "per_feature": per_feature,
    }
    print("\nGATE 2 -- next-step prediction (standardized MSE, test set):")
    print(f"  model:       {model_mse:.5f}")
    print(f"  persistence: {persistence_mse:.5f}")
    print(f"  train mean:  {mean_mse:.5f}")
    verdict = gate2["model_beats_persistence"] and gate2["model_beats_mean"]
    print(f"  model clearly beats both baselines: {verdict}")

    # ---- Save everything downstream steps need -----------------------------
    out = base / "world_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_action": n_action, "n_obs": n_obs,
        "enc_hidden": ENC_HIDDEN, "latent_dim": LATENT_DIM, "dec_hidden": DEC_HIDDEN,
        "action_mean": action_mean, "action_std": action_std,
        "obs_mean": obs_mean, "obs_std": obs_std,
    }, out)
    print(f"\nwrote {out}")

    (base / "gate2_results.json").write_text(json.dumps({
        "gate2": gate2,
        "model_params": n_params,
        "epochs": args.epochs,
        "history": history,
        "hyperparameters": {
            "latent_dim": LATENT_DIM, "enc_hidden": ENC_HIDDEN,
            "dec_hidden": DEC_HIDDEN, "kl_weight": KL_WEIGHT,
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        },
    }, indent=2))
    print(f"wrote {base / 'gate2_results.json'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
