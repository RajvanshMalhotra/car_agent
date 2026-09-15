"""One path from a windows split to model-ready arrays, and back to real units.

Round 7 adds two representation choices -- temperature as offset above ambient
and anchored residual prediction (`experiment/representation.py`) -- that both
training scripts and every evaluation must apply identically, or a model is
scored in a different space than it was trained in. This module is that
single path:

    prepare(split, scenarios, horizon, offsets=..., cf=...)   raw representation
    fit_stats(prepared)                                        standardisation, train only
    standardize_prepared(prepared, *stats)                     model space
    real_units(pred, future_actions, obs_mean, obs_std, offsets=...)

`prepare` reads a window's continuation from `scenarios` exactly as
`experiment/train_world_model.py:build_future_arrays` does (`arr[f][end+1 :
end+1+horizon]`). Anchors are computed AFTER the offset transform, so an
anchored model's residual lives in the representation it is trained on;
counterfactual anchors follow the counterfactual day types. With
`offsets=False` and no anchors in use, the arrays equal rounds 4-6's.
"""

from __future__ import annotations

import numpy as np

from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.representation import from_offsets, same_type_anchors, to_offsets, window_anchors

#: Keys standardised with the action statistics; everything else uses the observable ones.
ACTION_KEYS = frozenset({"actions", "future_actions", "cf_actions"})


def _continuation(split: dict, scenarios: dict, horizon: int, features: tuple[str, ...]) -> np.ndarray:
    n = split["scenario"].shape[0]
    out = np.zeros((n, horizon, len(features)), dtype=np.float32)
    for idx in range(n):
        arr = scenarios[str(split["scenario"][idx])]
        end = int(split["end_day"][idx])
        for j, feature in enumerate(features):
            out[idx, :, j] = arr[feature][end + 1:end + 1 + horizon]
    return out


def prepare(
    split: dict, scenarios: dict, horizon: int, *, offsets: bool, cf: dict | None = None,
) -> dict[str, np.ndarray]:
    actions = split["actions"].astype(np.float32)
    observables = split["observables"].astype(np.float32)
    future_actions = _continuation(split, scenarios, horizon, ACTION_FEATURES)
    future_observables = _continuation(split, scenarios, horizon, OBSERVABLE_FEATURES)
    if offsets:
        observables = to_offsets(observables, actions)
        future_observables = to_offsets(future_observables, future_actions)
    out = {
        "actions": actions, "observables": observables,
        "future_actions": future_actions, "future_observables": future_observables,
        "future_anchors": same_type_anchors(actions, observables, future_actions),
        "window_anchors": window_anchors(actions, observables),
    }
    if cf is not None:
        cf_actions = cf["cf_actions"].astype(np.float32)
        cf_observables = cf["cf_observables"].astype(np.float32)
        if offsets:
            cf_observables = to_offsets(cf_observables, cf_actions)
        out.update({
            "cf_actions": cf_actions, "cf_observables": cf_observables,
            "cf_anchors": same_type_anchors(actions, observables, cf_actions),
        })
    return out


def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    return mean.astype(np.float32), np.where(std < 1e-8, 1.0, std).astype(np.float32)


def fit_stats(prepared: dict[str, np.ndarray]):
    """(action_mean, action_std, obs_mean, obs_std) from a TRAINING split's windows."""
    return (*_stats(prepared["actions"]), *_stats(prepared["observables"]))


def standardize_prepared(
    prepared: dict[str, np.ndarray], action_mean, action_std, obs_mean, obs_std,
) -> dict[str, np.ndarray]:
    return {
        key: ((value - action_mean) / action_std if key in ACTION_KEYS
              else (value - obs_mean) / obs_std).astype(np.float32)
        for key, value in prepared.items()
    }


def real_units(
    standardized: np.ndarray, future_actions: np.ndarray, obs_mean, obs_std, *, offsets: bool,
) -> np.ndarray:
    raw = (standardized * obs_std + obs_mean).astype(np.float32)
    return from_offsets(raw, future_actions) if offsets else raw
