#!/usr/bin/env python3
"""Step 2: train the stochastic-latent world model, then run Gate 2.

**Multi-step closed-loop training (the fix for the collapse).** A first
version of this script trained SINGLE-step next-day prediction, decoder
conditioned on the TRUE previous-day observable, and that let the posterior
collapse to the prior -- see `experiment/model.py`'s module docstring and
the experiment report for the full diagnosis. This version instead trains
the exact same `WorldModel.rollout()` path Step 3 (`experiment/abduction.py`)
evaluates: the window is encoded once, then the decoder is unrolled
`ROLLOUT_HORIZON` days in CLOSED LOOP -- each day's `prev_obs` input is the
model's OWN previous prediction, not ground truth (`WorldModel.encode_and_rollout`).
Once the day-30 prediction depends on getting days 1-29 right using only
what the encoder put in `z`, a next-day shortcut that ignores the latent no
longer minimizes the loss.

`ROLLOUT_HORIZON` is set to `windows_meta.json`'s `rollout_k` (30 days) --
the same runway every window was built with and the same horizon the
abduction test rolls out over, so training and evaluation always agree on
what "the rollout" means.

**KL scheme actually used: free-bits, PLUS a linear warmup, belt and
braces.** `experiment/model.py:kl_free_bits` floors every latent
dimension's KL at `FREE_BITS` (0.15) nats -- a PERMANENT floor with zero
gradient below it, so collapse to the prior on any one dimension is no
longer reachable no matter how long training runs (see that function's
docstring). On top of that floor this script also linearly ramps the KL
weight from 0 up to `KL_WEIGHT` over the first `KL_WARMUP_EPOCHS` epochs,
so the reconstruction loss gets first pick of the optimizer's attention
before any regularization pressure arrives at all. Free bits is the
structural fix (it is what makes collapse actually impossible); the warmup
is an early-training convenience on top of it, not a replacement for it.

**Gate 2 is now the multi-step rollout diagnostic**, since that is what the
model is actually trained to do: reports rollout MSE (standardized, averaged
over the whole `ROLLOUT_HORIZON`-day horizon and all `OBSERVABLE_FEATURES`)
for (a) the model (posterior mean, no sampling), (b) persistence -- repeat
the window's last observed day for every future day, and (c) the
training-set mean future observable. The single-step `forward()` path is
still reported alongside, purely as a diagnostic matching what the FIRST
(collapsed) run measured -- it is never trained against here.

If the model does not clearly beat both multi-step baselines, the brief
says stop: a model that has not learned the dynamics cannot be meaningfully
probed for abduction. This script reports the comparison and exits either
way; **Gate 2b** (`experiment/gate2b.py`) is the separate, mandatory check
for whether the latent that got there is doing any work.

**The counterfactual-curriculum fix (on top of multi-step training).** The
corrected Step 3 abduction test found the multi-step-trained model was
conditioning on cheap, currently-observable state, not genuinely abducting
history -- see the experiment report's Section 5.2. The suspected cause:
every training example only ever unrolled under the FACTUAL action
continuation, so nothing ever forced the latent to distinguish "what
happened" from "what state resulted" under a schedule that didn't actually
happen. This script now also unrolls the SAME encoded `z` under a
RANDOMIZED counterfactual action sequence per training window
(`experiment/build_counterfactual_targets.py`'s `windows_train_cf.npz` --
run that script first) and trains against its exact ODE ground truth, in
addition to the factual continuation. If a single `z` has to explain two
divergent futures from the same true history, "remember the observable
trend" stops being sufficient on its own.

Run:
```
python3 -m experiment.build_counterfactual_targets --dataset-dir runs/experiment  # once
python3 -m experiment.train_world_model --dataset-dir runs/experiment
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.checkpoints import BestCheckpoint, validation_rollout_mse
from experiment.data import load_daily_trajectory
from experiment.model import WorldModel, kl_free_bits

KL_WEIGHT = 0.01
FREE_BITS = 0.15
KL_WARMUP_EPOCHS = 10
LATENT_DIM = 8
ENC_HIDDEN = 32
DEC_HIDDEN = 32
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
SEED = 0

#: Weight on the counterfactual-rollout term relative to the factual one.
#: 1.0 (equal weighting) so the curriculum can't be satisfied by just
#: getting better at the factual continuation and coasting on the
#: counterfactual one -- both have to improve together, from the same `z`.
CF_LOSS_WEIGHT = 1.0


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def load_split(base: Path, split: str) -> dict[str, np.ndarray]:
    return dict(np.load(base / f"windows_{split}.npz", allow_pickle=True))


def build_future_arrays(
    windows: dict[str, np.ndarray], scenarios: dict, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The `horizon` days of action/observable AFTER each window's end day.

    Mirrors `experiment/abduction.py`'s own lookup (`arr[f][end+1:end+1+k]`)
    exactly, so training and the abduction test read the identical
    continuation for a given (scenario, end_day) -- this is what lets Gate 2
    here and Step 3's FACTUAL rollout be directly comparable.
    """
    n = windows["scenario"].shape[0]
    n_action, n_obs = len(ACTION_FEATURES), len(OBSERVABLE_FEATURES)
    future_actions = np.zeros((n, horizon, n_action), dtype=np.float32)
    future_observables = np.zeros((n, horizon, n_obs), dtype=np.float32)
    for idx in range(n):
        name = str(windows["scenario"][idx])
        end = int(windows["end_day"][idx])
        arr = scenarios[name]
        for j, f in enumerate(ACTION_FEATURES):
            future_actions[idx, :, j] = arr[f][end + 1:end + 1 + horizon]
        for j, f in enumerate(OBSERVABLE_FEATURES):
            future_observables[idx, :, j] = arr[f][end + 1:end + 1 + horizon]
    return future_actions, future_observables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    windows_meta = json.loads((base / "windows_meta.json").read_text())
    horizon = windows_meta["rollout_k"]

    train = load_split(base, "train")
    test = load_split(base, "test")

    cf_path = base / "windows_train_cf.npz"
    if not cf_path.exists():
        raise FileNotFoundError(
            f"{cf_path} not found -- run "
            "`python3 -m experiment.build_counterfactual_targets "
            f"--dataset-dir {base}` first. The counterfactual-curriculum "
            "training loop needs a precomputed randomized counterfactual "
            "continuation (with exact ODE ground truth) for every training "
            "window; see this module's docstring for why."
        )
    train_cf = dict(np.load(cf_path, allow_pickle=True))
    assert np.array_equal(train_cf["scenario"], train["scenario"]) and \
        np.array_equal(train_cf["end_day"], train["end_day"]), (
        f"{cf_path} does not align row-for-row with windows_train.npz -- "
        "rebuild it with experiment.build_counterfactual_targets."
    )

    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")
    train_future_actions, train_future_obs = build_future_arrays(train, scenarios, horizon)
    test_future_actions, test_future_obs = build_future_arrays(test, scenarios, horizon)

    action_mean, action_std = _standardize_fit(train["actions"])
    obs_mean, obs_std = _standardize_fit(train["observables"])

    def prep(
        d: dict[str, np.ndarray], future_actions: np.ndarray, future_obs: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        actions = _apply(d["actions"], action_mean, action_std)
        observables = _apply(d["observables"], obs_mean, obs_std)
        next_action = _apply(d["next_action"], action_mean, action_std)
        next_observable = _apply(d["next_observable"], obs_mean, obs_std)
        fut_actions = _apply(future_actions, action_mean, action_std)
        fut_obs = _apply(future_obs, obs_mean, obs_std)
        return {
            "actions": torch.tensor(actions, dtype=torch.float32),
            "observables": torch.tensor(observables, dtype=torch.float32),
            "next_action": torch.tensor(next_action, dtype=torch.float32),
            "next_observable": torch.tensor(next_observable, dtype=torch.float32),
            "future_actions": torch.tensor(fut_actions, dtype=torch.float32),
            "future_observables": torch.tensor(fut_obs, dtype=torch.float32),
        }

    train_t = prep(train, train_future_actions, train_future_obs)
    test_t = prep(test, test_future_actions, test_future_obs)

    # Round 6: with a validation split present, keep the best epoch, not the last.
    val_t = None
    if (base / "windows_val.npz").exists():
        val = load_split(base, "val")
        val_t = prep(val, *build_future_arrays(val, scenarios, horizon))
    best = BestCheckpoint()
    train_t["cf_actions"] = torch.tensor(
        _apply(train_cf["cf_actions"], action_mean, action_std), dtype=torch.float32,
    )
    train_t["cf_observables"] = torch.tensor(
        _apply(train_cf["cf_observables"], obs_mean, obs_std), dtype=torch.float32,
    )

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
        kl_weight_t = KL_WEIGHT * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)
        perm = rng.permutation(n_train)
        epoch_recon_f, epoch_recon_cf, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            actions = train_t["actions"][idx].to(device)
            observables = train_t["observables"][idx].to(device)
            future_actions = train_t["future_actions"][idx].to(device)
            future_observables = train_t["future_observables"][idx].to(device)
            cf_actions = train_t["cf_actions"][idx].to(device)
            cf_observables = train_t["cf_observables"][idx].to(device)

            # One encode, one sampled z, TWO rollouts from it: the factual
            # continuation and a randomized counterfactual one. Both losses
            # are computed against the same z on purpose -- see the module
            # docstring on why a single z having to explain two divergent,
            # ODE-true futures is the actual curriculum fix being tested.
            mu, logvar = model.encode(actions, observables)
            z = model.reparameterize(mu, logvar)
            prev_obs0 = observables[:, -1, :]
            pred_factual = model.rollout(z, future_actions, prev_obs0)
            pred_cf = model.rollout(z, cf_actions, prev_obs0)

            recon_f = mse(pred_factual, future_observables)
            recon_cf = mse(pred_cf, cf_observables)
            kl = kl_free_bits(mu, logvar, free_bits=FREE_BITS)
            loss = recon_f + CF_LOSS_WEIGHT * recon_cf + kl_weight_t * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon_f += recon_f.item()
            epoch_recon_cf += recon_cf.item()
            epoch_kl += kl.item()
            n_batches += 1
        history.append({
            "epoch": epoch, "recon_mse_factual": epoch_recon_f / n_batches,
            "recon_mse_counterfactual": epoch_recon_cf / n_batches,
            "kl_free_bits": epoch_kl / n_batches, "kl_weight": kl_weight_t,
        })
        if val_t is not None:
            history[-1]["val_rollout_mse"] = validation_rollout_mse(model, val_t, device)
            best.update(epoch, history[-1]["val_rollout_mse"], model)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            if val_t is not None:
                print(f"          val_rollout={history[-1]['val_rollout_mse']:.4f}  "
                      f"best so far: epoch {best.best_epoch}", flush=True)
            print(f"epoch {epoch:3d}  recon_factual={history[-1]['recon_mse_factual']:.4f}  "
                  f"recon_cf={history[-1]['recon_mse_counterfactual']:.4f}  "
                  f"kl_free_bits={history[-1]['kl_free_bits']:.4f}  "
                  f"kl_weight={kl_weight_t:.4f}", flush=True)

    if val_t is not None:
        best.restore(model)
        print(f"\nrestored best epoch {best.best_epoch} (val rollout MSE {best.best_loss:.5f})")

    # ---- Gate 2: multi-step rollout error vs persistence and mean ---------
    model.eval()
    with torch.no_grad():
        actions = test_t["actions"].to(device)
        observables = test_t["observables"].to(device)
        future_actions = test_t["future_actions"].to(device)
        future_observables = test_t["future_observables"].to(device)

        pred, mu, logvar = model.encode_and_rollout(
            actions, observables, future_actions, sample=False,
        )
        model_mse = mse(pred, future_observables).item()

        # Persistence: repeat the window's last observed day for every future day.
        last_obs = observables[:, -1, :].unsqueeze(1)
        persistence_pred = last_obs.expand_as(future_observables)
        persistence_mse = mse(persistence_pred, future_observables).item()

        # Training-set mean future observable, broadcast over the horizon.
        train_mean_obs = train_t["future_observables"].mean(dim=(0, 1), keepdim=True)
        mean_pred = train_mean_obs.expand_as(future_observables).to(device)
        mean_mse = mse(mean_pred, future_observables).item()

        # Per-feature breakdown, standardized units, averaged over the horizon.
        per_feature_model = ((pred - future_observables) ** 2).mean(dim=(0, 1))
        per_feature_persist = ((persistence_pred - future_observables) ** 2).mean(dim=(0, 1))

        # Single-step diagnostic (never trained against) -- for comparison
        # against the FIRST run's collapsed numbers only.
        next_action = test_t["next_action"].to(device)
        next_observable = test_t["next_observable"].to(device)
        single_pred, _mu1, _logvar1 = model(actions, observables, next_action, sample=False)
        single_step_mse = mse(single_pred, next_observable).item()

    per_feature = {
        name: {
            "model_mse": float(per_feature_model[j]),
            "persistence_mse": float(per_feature_persist[j]),
        }
        for j, name in enumerate(OBSERVABLE_FEATURES)
    }

    gate2 = {
        "rollout_horizon_days": horizon,
        "model_rollout_mse": model_mse,
        "persistence_rollout_mse": persistence_mse,
        "mean_rollout_mse": mean_mse,
        "model_beats_persistence": model_mse < persistence_mse,
        "model_beats_mean": model_mse < mean_mse,
        "single_step_diagnostic_mse": single_step_mse,
        "per_feature": per_feature,
    }
    print(f"\nGATE 2 -- {horizon}-day closed-loop rollout prediction (standardized MSE, test set):")
    print(f"  model:                {model_mse:.5f}")
    print(f"  persistence (repeat): {persistence_mse:.5f}")
    print(f"  train mean:           {mean_mse:.5f}")
    print(f"  [diagnostic only, not trained against] single-step MSE: {single_step_mse:.5f}")
    verdict = gate2["model_beats_persistence"] and gate2["model_beats_mean"]
    print(f"  model clearly beats both rollout baselines: {verdict}")

    # ---- Save everything downstream steps need -----------------------------
    out = base / "world_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_action": n_action, "n_obs": n_obs,
        "enc_hidden": ENC_HIDDEN, "latent_dim": LATENT_DIM, "dec_hidden": DEC_HIDDEN,
        "action_mean": action_mean, "action_std": action_std,
        "obs_mean": obs_mean, "obs_std": obs_std,
        "rollout_horizon_days": horizon,
        "kl_scheme": {"free_bits": FREE_BITS, "kl_weight": KL_WEIGHT, "warmup_epochs": KL_WARMUP_EPOCHS},
    }, out)
    print(f"\nwrote {out}")

    (base / "gate2_results.json").write_text(json.dumps({
        "gate2": gate2,
        "model_params": n_params,
        "epochs": args.epochs,
        "best_epoch": best.best_epoch,
        "history": history,
        "hyperparameters": {
            "latent_dim": LATENT_DIM, "enc_hidden": ENC_HIDDEN,
            "dec_hidden": DEC_HIDDEN, "kl_weight": KL_WEIGHT,
            "free_bits": FREE_BITS, "kl_warmup_epochs": KL_WARMUP_EPOCHS,
            "rollout_horizon_days": horizon, "cf_loss_weight": CF_LOSS_WEIGHT,
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        },
    }, indent=2))
    print(f"wrote {base / 'gate2_results.json'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
