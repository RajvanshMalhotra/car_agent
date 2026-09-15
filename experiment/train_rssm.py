#!/usr/bin/env python3
"""Train the RSSM (`experiment/rssm.py`) under gru_vae's exact regime, then run Gate 2.

Everything that is not architecture is held identical to
`experiment/train_world_model.py`: the same windows, split, standardisation
(fit on the training windows), counterfactual curriculum
(`windows_train_cf.npz`), 30-day closed-loop horizon, KL weight and warmup,
free bits (gru_vae floors 0.15 nats on each of 8 latent dims; here the same
1.2 nats floors each day's 8-dim stochastic state), batch size, learning rate,
epochs and seed. A difference in the results is then the architecture.

    loss = window reconstruction   -- every observed day, from its posterior state
         + KL_WEIGHT * balanced KL -- per day, prior vs posterior
         + factual 30-day imagined rollout MSE
         + CF_LOSS_WEIGHT * counterfactual 30-day imagined rollout MSE

The imagined rollouts start from a SAMPLED final belief (as gru_vae's start
from a sampled `z`) and step the prior mean forward. Both rollouts share that
one belief, so it has to explain two divergent futures -- the same curriculum
argument gru_vae's training makes.

Gradient norm is clipped at `GRAD_CLIP` (Dreamer's value): unlike gru_vae,
this backpropagates through 60 filtering steps plus 30 imagined ones.

Gate 2 is reported exactly as gru_vae's was -- posterior-mean abduction,
prior-mean rollout, against persistence and the training mean -- so the two
`gate2_results.json` files compare line for line. gru_vae's single-step
diagnostic is omitted: it existed to compare against gru_vae's own collapsed
first run and has no RSSM counterpart.

Run: `python3 -m experiment.train_rssm --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.calendar_sim import OBSERVABLE_FEATURES
from experiment.checkpoints import BestCheckpoint, artifact, model_config, validation_rollout_mse
from experiment.data import load_daily_trajectory
from experiment.pipeline import fit_stats, prepare, standardize_prepared
from experiment.rssm import RSSM, balanced_kl
from experiment.train_world_model import (
    BATCH_SIZE,
    CF_LOSS_WEIGHT,
    EPOCHS,
    FREE_BITS,
    KL_WARMUP_EPOCHS,
    KL_WEIGHT,
    LR,
    SEED,
    _apply,
    _standardize_fit,
    build_future_arrays,
    load_split,
)

DETER = 36
STOCH = 8
HIDDEN = 32
KL_BALANCE = 0.8
FREE_BITS_PER_DAY = FREE_BITS * STOCH
GRAD_CLIP = 100.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offsets", action="store_true",
                        help="round 7: temperatures as offset above ambient")
    parser.add_argument("--anchored", action="store_true",
                        help="round 7: decode a correction to the same-day-type anchor")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    horizon = json.loads((base / "windows_meta.json").read_text())["rollout_k"]
    train = load_split(base, "train")
    test = load_split(base, "test")
    train_cf = dict(np.load(base / "windows_train_cf.npz", allow_pickle=True))
    assert np.array_equal(train_cf["scenario"], train["scenario"]) and \
        np.array_equal(train_cf["end_day"], train["end_day"]), (
        "windows_train_cf.npz does not align row-for-row with windows_train.npz"
    )

    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")

    # Round 7: the shared pipeline builds every array (see train_world_model.py).
    prep_train = prepare(train, scenarios, horizon, offsets=args.offsets, cf=train_cf)
    action_mean, action_std, obs_mean, obs_std = fit_stats(prep_train)

    def to_tensors(prepared: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        std = standardize_prepared(prepared, action_mean, action_std, obs_mean, obs_std)
        return {key: torch.tensor(value) for key, value in std.items()}

    tr = to_tensors(prep_train)
    te = to_tensors(prepare(test, scenarios, horizon, offsets=args.offsets))

    # Round 6: with a validation split present, keep the best epoch, not the last.
    va = None
    if (base / "windows_val.npz").exists():
        val = load_split(base, "val")
        va = to_tensors(prepare(val, scenarios, horizon, offsets=args.offsets))
    best = BestCheckpoint()

    device = torch.device(args.device)
    model = RSSM(
        n_action=train["actions"].shape[-1], n_obs=train["observables"].shape[-1],
        deter=DETER, stoch=STOCH, hidden=HIDDEN, anchored=args.anchored,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RSSM parameters: {n_params}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    n_train = tr["actions"].shape[0]
    rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(args.epochs):
        model.train()
        kl_weight_t = KL_WEIGHT * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)
        sums = {"recon_window": 0.0, "recon_factual": 0.0, "recon_cf": 0.0, "kl": 0.0}
        n_batches = 0
        perm = rng.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            b = {k: v[idx].to(device) for k, v in tr.items()}

            anchor = (lambda key: b[key]) if args.anchored else (lambda key: None)
            post = model.filter(b["actions"], b["observables"], sample=True, anchors=anchor("window_anchors"))
            recon_window = mse(post["recon"], b["observables"])
            kl = balanced_kl(
                post["post_mu"], post["post_logvar"], post["prior_mu"], post["prior_logvar"],
                alpha=KL_BALANCE, free_bits=FREE_BITS_PER_DAY,
            )
            pred_factual = model.rollout(post["state"], b["future_actions"], anchors=anchor("future_anchors"))
            pred_cf = model.rollout(post["state"], b["cf_actions"], anchors=anchor("cf_anchors"))
            recon_f = mse(pred_factual, b["future_observables"])
            recon_cf = mse(pred_cf, b["cf_observables"])
            loss = recon_window + kl_weight_t * kl + recon_f + CF_LOSS_WEIGHT * recon_cf

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            sums["recon_window"] += recon_window.item()
            sums["recon_factual"] += recon_f.item()
            sums["recon_cf"] += recon_cf.item()
            sums["kl"] += kl.item()
            n_batches += 1
        row = {"epoch": epoch, "kl_weight": kl_weight_t, **{k: v / n_batches for k, v in sums.items()}}
        if va is not None:
            row["val_rollout_mse"] = validation_rollout_mse(model, va, device)
            best.update(epoch, row["val_rollout_mse"], model)
        history.append(row)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  recon_window={row['recon_window']:.4f}  "
                  f"recon_factual={row['recon_factual']:.4f}  recon_cf={row['recon_cf']:.4f}  "
                  f"kl={row['kl']:.4f}  kl_weight={kl_weight_t:.4f}"
                  + (f"  val_rollout={row['val_rollout_mse']:.4f} (best epoch {best.best_epoch})"
                     if va is not None else ""), flush=True)

    if va is not None:
        best.restore(model)
        print(f"\nrestored best epoch {best.best_epoch} (val rollout MSE {best.best_loss:.5f})")

    # ---- Gate 2: identical comparison to train_world_model.py ------------
    model.eval()
    with torch.no_grad():
        actions = te["actions"].to(device)
        observables = te["observables"].to(device)
        future_actions = te["future_actions"].to(device)
        future_observables = te["future_observables"].to(device)

        pred = model.rollout(
            model.abduct(actions, observables), future_actions,
            anchors=te["future_anchors"].to(device) if args.anchored else None,
        )
        model_mse = mse(pred, future_observables).item()
        persistence_pred = observables[:, -1, :].unsqueeze(1).expand_as(future_observables)
        persistence_mse = mse(persistence_pred, future_observables).item()
        mean_pred = tr["future_observables"].mean(dim=(0, 1), keepdim=True).to(device)
        mean_mse = mse(mean_pred.expand_as(future_observables), future_observables).item()
        per_feature_model = ((pred - future_observables) ** 2).mean(dim=(0, 1))
        per_feature_persist = ((persistence_pred - future_observables) ** 2).mean(dim=(0, 1))

    gate2 = {
        "rollout_horizon_days": horizon,
        "model_rollout_mse": model_mse,
        "persistence_rollout_mse": persistence_mse,
        "mean_rollout_mse": mean_mse,
        "model_beats_persistence": model_mse < persistence_mse,
        "model_beats_mean": model_mse < mean_mse,
        "per_feature": {
            name: {"model_mse": float(per_feature_model[j]), "persistence_mse": float(per_feature_persist[j])}
            for j, name in enumerate(OBSERVABLE_FEATURES)
        },
    }
    verdict = gate2["model_beats_persistence"] and gate2["model_beats_mean"]
    print(f"\nGATE 2 -- {horizon}-day imagined rollout (standardized MSE, test set):")
    print(f"  model:                {model_mse:.5f}")
    print(f"  persistence (repeat): {persistence_mse:.5f}")
    print(f"  train mean:           {mean_mse:.5f}")
    print(f"  model clearly beats both rollout baselines: {verdict}")

    out = base / artifact("rssm", "world_model.pt")
    torch.save({
        "state_dict": model.cpu().state_dict(), **model_config(model), "offsets": args.offsets,
        "action_mean": action_mean, "action_std": action_std,
        "obs_mean": obs_mean, "obs_std": obs_std,
        "rollout_horizon_days": horizon,
        "kl_scheme": {
            "free_bits_per_day": FREE_BITS_PER_DAY, "kl_weight": KL_WEIGHT,
            "warmup_epochs": KL_WARMUP_EPOCHS, "balance": KL_BALANCE,
        },
    }, out)
    print(f"\nwrote {out}")

    results_path = base / artifact("rssm", "gate2_results.json")
    results_path.write_text(json.dumps({
        "gate2": gate2, "model_params": n_params, "epochs": args.epochs,
        "best_epoch": best.best_epoch, "history": history,
        "hyperparameters": {
            "deter": DETER, "stoch": STOCH, "hidden": HIDDEN, "kl_weight": KL_WEIGHT,
            "free_bits_per_day": FREE_BITS_PER_DAY, "kl_balance": KL_BALANCE,
            "kl_warmup_epochs": KL_WARMUP_EPOCHS, "rollout_horizon_days": horizon,
            "cf_loss_weight": CF_LOSS_WEIGHT, "batch_size": BATCH_SIZE, "lr": LR,
            "grad_clip": GRAD_CLIP, "seed": SEED,
            "offsets": args.offsets, "anchored": args.anchored,
        },
    }, indent=2))
    print(f"wrote {results_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
