#!/usr/bin/env python3
"""Step 6: ask the world model for the gentlest driving schedule, then check it.

Everything before this predicts. This prescribes: given a battery's observed
window, search the 30-day schedule that does it the least damage, and say how
much gentler that is than what the driver actually did.

    actions -> world model -> predicted observables -> damage (experiment/damage.py)

is differentiable end to end, so the search is gradient descent on the dials a
DRIVER controls:

    trips today        1 to 3 on a driving day
    minutes per trip   1.5 to 20
    accessory load     12 to 55 A

Three things are deliberately NOT optimised. Ambient temperature, because
nobody chooses the weather -- and because `load/thermal.py`'s bay estimate is
pinned to the one recording, so letting the optimiser move temperature would
let it exploit a known modelling artifact. Parked days stay parked, because a
layup is a life event rather than a driving choice. And `is_layup` therefore
never changes, which keeps the anchor selection fixed and the action row
physically consistent (round 8's lesson: optimise the free parameters and
rebuild the row from them, never the row's columns independently).

**The honesty check is the point.** Optimising against a learned model finds
that model's mistakes as readily as real physics. So every schedule it
recommends is replayed through the actual ODE from the battery's true hidden
state, and the report compares what the model PROMISED against what the
simulator DELIVERED. A large promise with a small delivery means the model is
dreaming, and that is a result worth having either way.

Damage is reported as a ratio against the factual schedule. Converting it to
days would need a rate constant fitted to batteries that actually died, which
this project does not have -- see `cell/life.py`, which refuses to report one.

Run: `python3 -m experiment.planner --dataset-dir runs/experiment_planned --windows 100`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cell.aging import AgingRates
from experiment.abduction import SOAK_H, planned_ground_truth
from experiment.calendar_sim import ACTION_FEATURES, OBSERVABLE_FEATURES
from experiment.checkpoints import artifact, load_model
from experiment.damage import (
    accumulate_crystal,
    daily_damage,
    health_loss,
    hours_from_actions,
    soft_daily_damage,
)
from experiment.data import load_daily_trajectory
from experiment.pipeline import prepare, real_units, standardize_prepared
from experiment.schedule import ACCESSORY_RANGE_A, MAX_TRIPS, TRIP_MINUTES_RANGE, DayPlan
from sweep_dataset import load_driving

STEPS = 250
LR = 0.08
OBS_IDX = {name: i for i, name in enumerate(OBSERVABLE_FEATURES)}
ACT_IDX = {name: i for i, name in enumerate(ACTION_FEATURES)}


def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def build_action_row(
    trips: torch.Tensor, minutes: torch.Tensor, accessory: torch.Tensor,
    is_layup: torch.Tensor, ambient: torch.Tensor,
) -> torch.Tensor:
    """`[B, K, n_action]` rebuilt from the free dials, in ACTION_FEATURES order.

    The columns are not independent -- driving minutes is trips x minutes, soak
    follows the trip count, and a parked day zeroes the driving terms -- so the
    row is always rebuilt rather than optimised column by column.
    """
    driving = torch.where(is_layup >= 0.5, torch.zeros_like(trips), trips)
    driving_minutes = driving * minutes
    soak = torch.where(is_layup >= 0.5, torch.full_like(trips, 24.0), driving * SOAK_H)
    columns = {
        "is_layup": is_layup,
        "driving_minutes": driving_minutes,
        "soak_hours": soak,
        "ambient_c": ambient,
        "trips_today": driving,
        "accessory_a": accessory,
    }
    return torch.stack([columns[name] for name in ACTION_FEATURES], dim=-1)


def predicted_damage(
    model, state: torch.Tensor, actions_std: torch.Tensor, anchors: torch.Tensor,
    prev_obs: torch.Tensor, actions_raw: torch.Tensor, obs_mean, obs_std, crystal0,
    soft: bool = True,
) -> torch.Tensor:
    """Health lost over the horizon, as the world model sees it."""
    pred_std = model.rollout(state, actions_std, prev_obs, anchors=anchors)
    pred = pred_std * obs_std + obs_mean
    hours = hours_from_actions(
        actions_raw[..., ACT_IDX["is_layup"]],
        actions_raw[..., ACT_IDX["driving_minutes"]],
        actions_raw[..., ACT_IDX["soak_hours"]],
    )
    terms = (soft_daily_damage if soft else daily_damage)(
        pred[..., OBS_IDX["t_bat_mean"]], pred[..., OBS_IDX["soc"]], hours,
        t_bat_max=pred[..., OBS_IDX["t_bat_max"]],
    )
    crystal = accumulate_crystal(crystal0, terms["low_soc_hours"], terms["full_charge_hours"])
    return health_loss(terms["corrosion_equivalent_h"].sum(dim=1), crystal)


def true_damage(rows: list[dict], crystal0: float) -> float:
    """The same quantity, from observables the ODE actually produced."""
    t = lambda key: torch.tensor([[r[key] for r in rows]], dtype=torch.float64)
    hours = hours_from_actions(t("is_layup"), t("driving_minutes"), t("soak_hours"))
    terms = daily_damage(t("t_bat_mean"), t("soc"), hours, t_bat_max=t("t_bat_max"))
    crystal = accumulate_crystal(
        torch.tensor([crystal0], dtype=torch.float64),
        terms["low_soc_hours"], terms["full_charge_hours"],
    )
    return float(health_loss(terms["corrosion_equivalent_h"].sum(dim=1), crystal))


def plans_from_actions(actions: np.ndarray) -> list[DayPlan]:
    """Turn an optimised action row back into plans the ODE can replay."""
    plans = []
    for day in actions:
        trips = int(round(float(day[ACT_IDX["trips_today"]])))
        minutes = float(day[ACT_IDX["driving_minutes"]]) / trips if trips else 0.0
        plans.append(DayPlan(
            trips=trips,
            trip_minutes=float(np.clip(minutes, *TRIP_MINUTES_RANGE)) if trips else 0.0,
            ambient_c=float(day[ACT_IDX["ambient_c"]]),
            accessory_a=float(day[ACT_IDX["accessory_a"]]),
        ))
    return plans


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/experiment_planned")
    parser.add_argument("--arch", default="rssm")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--trajectory", default="runs/telemetry.csv")
    args = parser.parse_args(argv)
    base = Path(args.dataset_dir)

    torch.manual_seed(0)
    model, ck = load_model(base, args.arch)
    horizon = json.loads((base / "windows_meta.json").read_text())["rollout_k"]
    test = dict(np.load(base / "windows_test.npz", allow_pickle=True))
    scenarios, _ = load_daily_trajectory(base / "daily_trajectory.csv")
    meta = {m["scenario"]: m for m in json.loads((base / "scenarios.json").read_text())}
    driving, _ = load_driving(Path(args.trajectory))
    rates = AgingRates()

    n = min(args.windows, test["scenario"].shape[0])
    prep = prepare(test, scenarios, horizon, offsets=False)
    std = standardize_prepared(prep, ck["action_mean"], ck["action_std"], ck["obs_mean"], ck["obs_std"])
    obs_mean = torch.tensor(ck["obs_mean"])
    obs_std = torch.tensor(ck["obs_std"])
    a_mean = torch.tensor(ck["action_mean"])
    a_std = torch.tensor(ck["action_std"])

    with torch.no_grad():
        window_actions = torch.tensor(std["actions"][:n])
        window_obs = torch.tensor(std["observables"][:n])
        state = model.abduct(window_actions, window_obs)
        prev_obs = window_obs[:, -1, :]
        anchors = torch.tensor(std["future_anchors"][:n])

    factual_raw = torch.tensor(prep["future_actions"][:n])
    is_layup = factual_raw[..., ACT_IDX["is_layup"]]
    ambient = factual_raw[..., ACT_IDX["ambient_c"]]
    crystal0 = torch.zeros(n, dtype=torch.float32)

    # Start the search at the driver's actual schedule.
    factual_trips = factual_raw[..., ACT_IDX["trips_today"]].clamp(1.0, MAX_TRIPS)
    factual_minutes = torch.where(
        factual_trips > 0,
        factual_raw[..., ACT_IDX["driving_minutes"]] / factual_trips.clamp(min=1.0),
        torch.full_like(factual_trips, TRIP_MINUTES_RANGE[0]),
    ).clamp(*TRIP_MINUTES_RANGE)
    factual_accessory = factual_raw[..., ACT_IDX["accessory_a"]].clamp(*ACCESSORY_RANGE_A)

    def inverse_sigmoid(value, low, high):
        frac = ((value - low) / (high - low)).clamp(1e-3, 1 - 1e-3)
        return torch.log(frac / (1 - frac))

    trips_raw = inverse_sigmoid(factual_trips, 1.0, MAX_TRIPS).clone().requires_grad_(True)
    minutes_raw = inverse_sigmoid(factual_minutes, *TRIP_MINUTES_RANGE).clone().requires_grad_(True)
    accessory_raw = inverse_sigmoid(factual_accessory, *ACCESSORY_RANGE_A).clone().requires_grad_(True)

    with torch.no_grad():
        baseline_pred = predicted_damage(
            model, state, torch.tensor(std["future_actions"][:n]), anchors, prev_obs,
            factual_raw, obs_mean, obs_std, crystal0, soft=False,
        )

    optimiser = torch.optim.Adam([trips_raw, minutes_raw, accessory_raw], lr=args.steps and LR)
    for step in range(args.steps):
        actions_raw = build_action_row(
            _bounded(trips_raw, 1.0, MAX_TRIPS),
            _bounded(minutes_raw, *TRIP_MINUTES_RANGE),
            _bounded(accessory_raw, *ACCESSORY_RANGE_A),
            is_layup, ambient,
        )
        actions_std = (actions_raw - a_mean) / a_std
        loss = predicted_damage(
            model, state, actions_std, anchors, prev_obs, actions_raw,
            obs_mean, obs_std, crystal0,
        ).sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  predicted damage {loss.item() / n:.6e}", flush=True)

    with torch.no_grad():
        actions_raw = build_action_row(
            _bounded(trips_raw, 1.0, MAX_TRIPS),
            _bounded(minutes_raw, *TRIP_MINUTES_RANGE),
            _bounded(accessory_raw, *ACCESSORY_RANGE_A),
            is_layup, ambient,
        )
        optimised_pred = predicted_damage(
            model, state, (actions_raw - a_mean) / a_std, anchors, prev_obs,
            actions_raw, obs_mean, obs_std, crystal0, soft=False,
        )

    # ---- the honesty check: replay both schedules through the real ODE -------
    promised, delivered, rows_out = [], [], []
    for idx in range(n):
        name, end = str(test["scenario"][idx]), int(test["end_day"][idx])
        arr = scenarios[name]
        crystal_true = float(arr["crystal"][end])

        factual_rows = [
            {key: float(arr[key][end + 1 + d]) for key in
             ("is_layup", "driving_minutes", "soak_hours", "t_bat_mean", "t_bat_max", "soc")}
            for d in range(horizon)
        ]
        plans = plans_from_actions(actions_raw[idx].numpy())
        planned_rows = planned_ground_truth(
            meta[name], driving, arr, end, plans, rates, return_rows=True,
        )

        factual_true = true_damage(factual_rows, crystal_true)
        planned_true = true_damage(planned_rows, crystal_true)
        promised.append(float(optimised_pred[idx]) / max(float(baseline_pred[idx]), 1e-12))
        delivered.append(planned_true / max(factual_true, 1e-12))
        rows_out.append({
            "scenario": name, "end_day": end,
            "promised_ratio": promised[-1], "delivered_ratio": delivered[-1],
            "factual_trips": float(factual_raw[idx, :, ACT_IDX["trips_today"]].mean()),
            "planned_trips": float(actions_raw[idx, :, ACT_IDX["trips_today"]].mean()),
            "factual_minutes_per_trip": float(factual_minutes[idx].mean()),
            "planned_minutes_per_trip": float(_bounded(minutes_raw, *TRIP_MINUTES_RANGE)[idx].mean()),
            "factual_accessory_a": float(factual_accessory[idx].mean()),
            "planned_accessory_a": float(_bounded(accessory_raw, *ACCESSORY_RANGE_A)[idx].mean()),
        })
        if (idx + 1) % 20 == 0:
            print(f"  verified {idx + 1}/{n}", flush=True)

    promised_arr, delivered_arr = np.array(promised), np.array(delivered)
    result = {
        "n_windows": n, "steps": args.steps,
        "promised_damage_ratio_median": float(np.median(promised_arr)),
        "delivered_damage_ratio_median": float(np.median(delivered_arr)),
        "delivered_better_than_factual": float((delivered_arr < 1.0).mean()),
        "promise_kept_fraction": float((delivered_arr <= promised_arr + 0.05).mean()),
        "per_window": rows_out,
    }
    path = base / artifact(args.arch, "planner_results.json")
    path.write_text(json.dumps(result, indent=2))

    print(f"\nPLANNED SCHEDULES, {n} batteries")
    print(f"  model promised damage x {np.median(promised_arr):.3f} of the driver's own schedule")
    print(f"  simulator delivered   x {np.median(delivered_arr):.3f}")
    print(f"  schedules that really were gentler: {(delivered_arr < 1.0).mean():.0%}")
    print(f"  median change in habits: trips {np.median([r['factual_trips'] for r in rows_out]):.2f}"
          f" -> {np.median([r['planned_trips'] for r in rows_out]):.2f},"
          f" minutes/trip {np.median([r['factual_minutes_per_trip'] for r in rows_out]):.1f}"
          f" -> {np.median([r['planned_minutes_per_trip'] for r in rows_out]):.1f},"
          f" accessory {np.median([r['factual_accessory_a'] for r in rows_out]):.0f}"
          f" -> {np.median([r['planned_accessory_a'] for r in rows_out]):.0f} A")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
