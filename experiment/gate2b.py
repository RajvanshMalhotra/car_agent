#!/usr/bin/env python3
"""Gate 2b -- the latent-informativeness gate the first run never had.

The FIRST training run (single-step, plain KL -- see `experiment/model.py`'s
module docstring) produced an apparently clean Step 3 negative result that
turned out to be void: the posterior had collapsed to the prior, so the
"abducted" latent handed to the counterfactual rollout WAS the prior/zero
latent, and its agreement with the interventional baseline was arithmetic,
not evidence. Nothing in the original pipeline checked this before running
Step 3. This script is that check, and it is mandatory: Step 3
(`experiment/abduction.py`) and Step 4 (`experiment/sae_probe.py`) should
not be trusted -- or run at all, per the brief -- if this gate fails.

It tests two INDEPENDENT ways the pipeline could still be meaningless even
though Gate 2's rollout MSE looks good:

**(a) Is the posterior different from the prior?**
Encode every held-out test window and look at the distribution of posterior
means `mu` and stds `sigma = exp(0.5*logvar)` across examples. A collapsed
posterior looks like: every example's `mu` near the zero vector (low
per-dimension std of `mu` across examples -- the encoder outputs nearly the
same point regardless of input) and `sigma` near 1 (the prior's std) on most
dimensions. This is exactly what the void run measured (mean |mu|=0.025,
per-dim std of mu 0.025-0.089, mean sigma 0.9926). Requires, PER DIMENSION,
std of `mu` across examples clearly above 0.1 and mean `sigma` clearly below
0.9, on a majority of the latent's dimensions.

**(b) Does the decoder actually USE the latent it's given?**
A non-collapsed posterior is not sufficient either -- the decoder could
still learn to ignore `z` and drive entirely off `[action_t, prev_obs]` (the
autoregressive/persistence path already does most of the work; the closed-
loop training pressure in Task A only rewards the latent if the DECODER
reads it). So this compares multi-step closed-loop rollout error, on held-
out test windows, using (i) the TRUE abducted latent (`mu` from encoding the
window) against (ii) a ZEROED latent (`z = 0`, prior mean, same as the
INTERVENTIONAL condition in Step 3), everything else (actions, prev_obs0)
identical. If zeroing the latent barely moves the error, the decoder is
ignoring `z` and Step 3's "abducted vs interventional" comparison would be
measuring nothing again.

Run: `python3 -m experiment.gate2b --dataset-dir runs/experiment`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiment.abduction import standardize
from experiment.checkpoints import ARCHS, artifact, load_model, rollout_kwargs
from experiment.data import load_daily_trajectory
from experiment.train_world_model import build_future_arrays

#: Part (a) thresholds -- "clearly" above/below the prior's own values (0
#: mean, 1 std), not merely nonzero/non-one.
MU_STD_THRESHOLD = 0.1
SIGMA_THRESHOLD = 0.9

#: Part (b) threshold -- zeroing the latent must cost at least this much
#: relative rollout error for the decoder to be judged as using it.
MIN_RELATIVE_DEGRADATION = 0.15


def part_a_posterior_vs_prior(model, checkpoint: dict, test: dict) -> dict:
    """Mean |mu|, per-dimension std of mu, and mean sigma over the test set.

    For the RSSM this is the final day's posterior, and the thresholds still
    compare against N(0, 1) -- but an RSSM's prior is learned, not N(0, 1),
    so passing here does not rule out posterior == prior. The raw final-day
    KL(q || p) is reported alongside as the direct check; part (b) and the
    abduction test are the ones that decide.
    """
    with torch.no_grad():
        actions = torch.tensor(
            standardize(test["actions"], checkpoint["action_mean"], checkpoint["action_std"]),
        )
        observables = torch.tensor(
            standardize(test["observables"], checkpoint["obs_mean"], checkpoint["obs_std"]),
        )
        mu, logvar = model.posterior(actions, observables)
        sigma = torch.exp(0.5 * logvar)
        final_day_kl = None
        if hasattr(model, "filter"):
            from experiment.rssm import _gaussian_kl
            out = model.filter(actions, observables, sample=False)
            final_day_kl = float(_gaussian_kl(
                out["post_mu"][:, -1], out["post_logvar"][:, -1],
                out["prior_mu"][:, -1], out["prior_logvar"][:, -1],
            ).mean())

    mu_np = mu.numpy()
    sigma_np = sigma.numpy()

    mean_abs_mu = float(np.abs(mu_np).mean())
    per_dim_std_mu = mu_np.std(axis=0)
    per_dim_mean_sigma = sigma_np.mean(axis=0)
    mean_sigma = float(sigma_np.mean())

    n_dims = mu_np.shape[1]
    dims_std_ok = int((per_dim_std_mu > MU_STD_THRESHOLD).sum())
    dims_sigma_ok = int((per_dim_mean_sigma < SIGMA_THRESHOLD).sum())
    majority = n_dims / 2.0
    passed = dims_std_ok > majority and dims_sigma_ok > majority

    return {
        "n_examples": int(mu_np.shape[0]),
        "latent_dim": n_dims,
        "mean_abs_mu": mean_abs_mu,
        "per_dim_std_mu": [float(v) for v in per_dim_std_mu],
        "per_dim_mean_sigma": [float(v) for v in per_dim_mean_sigma],
        "mean_sigma": mean_sigma,
        "mu_std_threshold": MU_STD_THRESHOLD,
        "sigma_threshold": SIGMA_THRESHOLD,
        "dims_with_std_mu_above_threshold": dims_std_ok,
        "dims_with_mean_sigma_below_threshold": dims_sigma_ok,
        "final_day_kl_posterior_vs_learned_prior": final_day_kl,
        "passed": passed,
    }


def part_b_decoder_uses_latent(
    model, checkpoint: dict, test: dict, scenarios: dict, horizon: int,
) -> dict:
    """Multi-step rollout error: true abducted latent vs zeroed latent."""
    future_actions_raw, future_obs_raw = build_future_arrays(test, scenarios, horizon)

    with torch.no_grad():
        actions = torch.tensor(
            standardize(test["actions"], checkpoint["action_mean"], checkpoint["action_std"]),
        )
        observables = torch.tensor(
            standardize(test["observables"], checkpoint["obs_mean"], checkpoint["obs_std"]),
        )
        future_actions = torch.tensor(
            standardize(future_actions_raw, checkpoint["action_mean"], checkpoint["action_std"]),
        )
        future_observables = torch.tensor(
            standardize(future_obs_raw, checkpoint["obs_mean"], checkpoint["obs_std"]),
        )

        state = model.abduct(actions, observables)
        prev_obs0 = observables[:, -1, :]

        window = rollout_kwargs(model, actions, observables)
        pred_true_latent = model.rollout(state, future_actions, prev_obs0, **window)
        pred_zero_latent = model.rollout(
            model.zero_state(state.shape[0]), future_actions, prev_obs0, **window)

        true_latent_mse = float(((pred_true_latent - future_observables) ** 2).mean())
        zero_latent_mse = float(((pred_zero_latent - future_observables) ** 2).mean())

    relative_degradation = (
        (zero_latent_mse - true_latent_mse) / true_latent_mse
        if true_latent_mse > 1e-12 else float("inf")
    )
    passed = relative_degradation > MIN_RELATIVE_DEGRADATION

    return {
        "n_examples": int(test["actions"].shape[0]),
        "rollout_horizon_days": horizon,
        "true_latent_rollout_mse": true_latent_mse,
        "zero_latent_rollout_mse": zero_latent_mse,
        "relative_degradation_from_zeroing": relative_degradation,
        "min_relative_degradation_threshold": MIN_RELATIVE_DEGRADATION,
        "passed": passed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment")
    parser.add_argument("--arch", choices=ARCHS, default="gru_vae")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    model, checkpoint = load_model(base, args.arch)
    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))
    windows_meta = json.loads((base / "windows_meta.json").read_text())
    horizon = checkpoint.get("rollout_horizon_days", windows_meta["rollout_k"])

    scenarios, _order = load_daily_trajectory(base / "daily_trajectory.csv")

    part_a = part_a_posterior_vs_prior(model, checkpoint, test)
    part_b = part_b_decoder_uses_latent(model, checkpoint, test, scenarios, horizon)

    overall_pass = part_a["passed"] and part_b["passed"]

    result = {
        "part_a_posterior_not_prior": part_a,
        "part_b_decoder_uses_latent": part_b,
        "passed": overall_pass,
        "verdict": (
            "GATE 2b PASSED: the posterior is measurably different from the "
            "prior and zeroing the latent measurably degrades rollout error "
            "-- the latent carries information and the decoder reads it. "
            "Safe to proceed to Step 3."
            if overall_pass else
            "GATE 2b FAILED: " + (
                "the posterior looks collapsed (mu near 0, sigma near 1 on "
                "most dimensions) -- "
                if not part_a["passed"] else ""
            ) + (
                "the decoder does not meaningfully use the latent (zeroing it "
                "barely changes rollout error) -- "
                if not part_b["passed"] else ""
            ) + "STOP. Do not proceed to Step 3 with this checkpoint."
        ),
    }
    results_path = base / artifact(args.arch, "gate2b_results.json")
    results_path.write_text(json.dumps(result, indent=2))

    print("GATE 2b -- latent informativeness")
    print("\n(a) posterior vs prior (test set, n={}):".format(part_a["n_examples"]))
    print(f"    mean |mu|:               {part_a['mean_abs_mu']:.4f}")
    print(f"    per-dim std of mu:       {[round(v, 4) for v in part_a['per_dim_std_mu']]}")
    print(f"    per-dim mean sigma:      {[round(v, 4) for v in part_a['per_dim_mean_sigma']]}")
    print(f"    mean sigma:              {part_a['mean_sigma']:.4f}")
    print(f"    dims with std(mu) > {MU_STD_THRESHOLD}:   {part_a['dims_with_std_mu_above_threshold']}/{part_a['latent_dim']}")
    print(f"    dims with mean sigma < {SIGMA_THRESHOLD}: {part_a['dims_with_mean_sigma_below_threshold']}/{part_a['latent_dim']}")
    print(f"    PART A: {'PASS' if part_a['passed'] else 'FAIL'}")

    print(f"\n(b) decoder uses latent ({part_b['rollout_horizon_days']}-day rollout, n={part_b['n_examples']}):")
    print(f"    true-latent rollout MSE: {part_b['true_latent_rollout_mse']:.5f}")
    print(f"    zero-latent rollout MSE: {part_b['zero_latent_rollout_mse']:.5f}")
    print(f"    relative degradation from zeroing: {part_b['relative_degradation_from_zeroing']:.1%}")
    print(f"    PART B: {'PASS' if part_b['passed'] else 'FAIL'}")

    print(f"\n{result['verdict']}")
    print(f"\nwrote {results_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
