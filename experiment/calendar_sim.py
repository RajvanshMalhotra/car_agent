"""Literal calendar-day simulation: real daily observables for the windowed dataset.

`cell/life.py`'s `project()` deliberately SMOOTHS a periodic layup pattern
into a single per-day-average `Damage` (see its own docstring and
`sweep_dataset.py`'s `day_damage` docstring) -- the right move for projecting
the scalar health/hidden-state curve cheaply over a multi-year horizon, and
the wrong move here: this experiment needs the literal day-to-day OBSERVABLE
pattern (trip count, driving minutes, soak hours, terminal voltage, current,
battery and bay temperature) that a real external observer would actually
see, not an average smeared across a whole gap-plus-layup cycle. So this
module reimplements the day-stepping loop literally: it walks real calendar
days, alternates `layup_gap_days`-long stretches of ordinary two-trips-a-day
driving with `layup_days`-long continuous parks, and calls straight into
`cell.integrate.run_trip` / `run_soak` for each one -- which already call
`cell.aging.accumulate()` internally, so the hidden ageing state advances on
the real per-day damage, not an averaged one.

**Cost.** Benchmarked directly (see the experiment report): one ordinary day
(two trips of the recorded ~10-minute average plus an 8 h soak at 60 s
resolution) costs on the order of 5-10 ms; a layup day (a 24 h park at 300 s
resolution) costs under 2 ms. A 2000-day, 80-scenario dataset therefore costs
on the order of ten minutes, not hours -- cheap enough to simulate literally
rather than approximate.

**Action / observable split.** Each calendar day produces two kinds of
field: `ACTION_FEATURES` (`is_layup`, `driving_minutes`, `soak_hours`,
`ambient_c`) are exogenous -- set by the schedule the experiment is asking
"what if" about, not produced by the physics -- and `OBSERVABLE_FEATURES`
(terminal voltage and current summary stats, battery and bay temperature
summary stats, state of charge) are what the physics produces in response.
The world model is trained to predict the observable half from the action
half plus its own latent state; the hidden `HIDDEN_FIELDS` (crystal size,
corrosion-hours, shedding, health) are recorded for scoring only and must
never be fed to the model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from cell.aging import AgingRates, soh
from cell.integrate import CellState, Step, run_soak, run_trip
from load.spec import BatteryScenario

#: Soak resolution for an ordinary night's park -- fine enough to resolve the
#: bay's ~60 s time constant (see `load/thermal.py`), coarse enough that
#: 8 h / 60 s = 480 steps/day is cheap.
NORMAL_SOAK_DT_S = 60.0

#: Soak resolution during a multi-day layup -- the bay has long since settled
#: by day 2 of a park, so a coarser step is fine and keeps a 24 h layup day at
#: 288 steps instead of 2880.
LAYUP_SOAK_DT_S = 300.0

#: Exogenous, schedule-controlled daily fields. Fixed column order shared by
#: dataset build, model, and abduction code so they cannot drift apart.
#: Trip length used only to seed `last_coolant_c` before the first driving day.
TRIP_MINUTES_SEED = 20.0

#: Round 12 appended `trips_today` and `accessory_a` rather than replacing
#: anything, so every name-based lookup and the round 7 anchor keep working.
#: `trips_today` is deliberately separate from `driving_minutes`: together they
#: distinguish one 20-minute trip from four 5-minute ones, which is the whole
#: point of the richer action space.
ACTION_FEATURES: tuple[str, ...] = (
    "is_layup", "driving_minutes", "soak_hours", "ambient_c",
    "trips_today", "accessory_a",
)

#: What the physics produces in response to a day's actions. What a real
#: observer could plausibly read off the vehicle: terminal voltage and
#: current summary stats, battery and bay temperature summary stats, and
#: state of charge. This experiment treats bay temperature and SoC as
#: observable, which is a deliberate departure from `cell/sensors.py`'s
#: stricter measurable/latent split (used for a different purpose, the main
#: project's sensor-noise layer) -- see the experiment report for why.
OBSERVABLE_FEATURES: tuple[str, ...] = (
    "v_mean", "v_min", "v_max", "i_mean", "i_mean_abs",
    "t_bat_mean", "t_bat_max", "t_bay_mean", "t_bay_max", "soc",
)

#: The hidden, path-dependent hAgeing state. Never fed to the model; carried
#: only so a probe can score what the model recovered.
HIDDEN_FIELDS: tuple[str, ...] = ("crystal", "corrosion_hours", "shedding", "soh")

#: Extra per-day fields needed to EXACTLY resume the ODE from a given day --
#: not part of the observable or hidden vocabulary the model ever sees, only
#: used internally to reconstruct a `CellState` for the abduction test's
#: ground-truth counterfactual rerun.
RESUME_FIELDS: tuple[str, ...] = ("resume_temp_c", "resume_bay_temp_c")


def truncate_driving(driving: Sequence, minutes: float) -> list:
    """The first `minutes` of the recorded trip. Mirrors `sweep_dataset.py`.

    Duplicated rather than imported: `sweep_dataset.py` is a standalone
    script, and importing it purely for this one helper would tie this
    package's import graph to a script's `if __name__` block for no benefit.
    """
    if not driving:
        return list(driving)
    start_t = float(driving[0]["t_s"])
    cutoff = minutes * 60.0
    return [s for s in driving if float(s["t_s"]) - start_t < cutoff]


def _summarize(steps: Sequence[Step]) -> dict:
    n = len(steps)
    v = [s.v_bat_v for s in steps]
    i = [s.i_bat_a for s in steps]
    tb = [s.t_bat_c for s in steps]
    tbay = [s.t_bay_c for s in steps]
    return {
        "v_mean": sum(v) / n, "v_min": min(v), "v_max": max(v),
        "i_mean": sum(i) / n, "i_mean_abs": sum(abs(x) for x in i) / n,
        "t_bat_mean": sum(tb) / n, "t_bat_max": max(tb),
        "t_bay_mean": sum(tbay) / n, "t_bay_max": max(tbay),
        "soc": steps[-1].soc,
    }


def step_one_day(
    state: CellState,
    scenario: BatteryScenario,
    trip_samples: Sequence,
    rates: AgingRates,
    is_layup: bool,
    trips_per_day: float,
    soak_h: float,
    last_coolant_c: float,
    accessory_a: float | None = None,
) -> tuple[dict, dict]:
    """Advance `state` by exactly one real calendar day, in place.

    Returns `(observation, hidden)`: `observation` carries `ACTION_FEATURES`
    plus `OBSERVABLE_FEATURES`; `hidden` carries `HIDDEN_FIELDS` plus
    `RESUME_FIELDS`, all read after this day's real damage has been
    accumulated into `state.aging` (which `run_trip`/`run_soak` do
    internally -- see `cell/integrate.py`).
    """
    if is_layup:
        result = run_soak(
            86400.0, scenario, state, rates,
            initial_coolant_c=last_coolant_c, dt_s=LAYUP_SOAK_DT_S,
        )
        obs = _summarize(result.steps)
        obs.update(
            is_layup=1.0, driving_minutes=0.0, soak_hours=24.0,
            ambient_c=scenario.ambient_c,
            trips_today=0.0,
            accessory_a=scenario.accessory_base_a if accessory_a is None else accessory_a,
        )
    else:
        n_trips = int(round(trips_per_day))
        steps: list[Step] = []
        for _ in range(n_trips):
            trip = run_trip(iter(trip_samples), scenario, state, rates, cranked=True)
            steps.extend(trip.steps)
            soak = run_soak(
                soak_h * 3600.0, scenario, state, rates,
                initial_coolant_c=last_coolant_c, dt_s=NORMAL_SOAK_DT_S,
            )
            steps.extend(soak.steps)
        obs = _summarize(steps)
        obs.update(
            is_layup=0.0,
            driving_minutes=n_trips * (len(trip_samples) / 60.0),
            soak_hours=n_trips * soak_h,
            ambient_c=scenario.ambient_c,
            trips_today=float(n_trips),
            accessory_a=scenario.accessory_base_a if accessory_a is None else accessory_a,
        )
    health = soh(state.aging, rates)
    hidden = {
        "crystal": state.aging.crystal,
        "corrosion_hours": state.aging.corrosion_hours,
        "shedding": state.aging.shedding,
        "soh": health,
        "resume_temp_c": state.temp_c,
        "resume_bay_temp_c": (
            state.bay_temp_c if state.bay_temp_c is not None else state.temp_c
        ),
    }
    return obs, hidden


def build_schedule(layup_gap_days: float, layup_days: float, horizon_days: int) -> list[bool]:
    """One `True`/`False` (is-layup) flag per calendar day of the horizon.

    Continuous `layup_gap_days`/`layup_days` are rounded to whole days here --
    the day-stepping loop needs an integer calendar, and rounding rather than
    truncating keeps a scenario drawn near, say, 0.4 days of layup landing on
    zero layup days (no special case needed) rather than always flooring
    away a fractional park.
    """
    gap = max(1, round(layup_gap_days))
    layup = max(0, round(layup_days))
    pattern = [False] * gap + [True] * layup
    return [pattern[d % len(pattern)] for d in range(horizon_days)]


def run_calendar_scenario(
    driving_samples: Sequence,
    ambient_c: float,
    trip_minutes: float,
    layup_days: float,
    layup_gap_days: float,
    parasitic_a: float,
    trips_per_day: float,
    soak_h: float,
    horizon_days: int,
    rates: AgingRates,
    seed: int,
) -> list[dict]:
    """Run one scenario day by day. Returns one dict per calendar day.

    Each dict holds `day` plus every field in `ACTION_FEATURES`,
    `OBSERVABLE_FEATURES`, `HIDDEN_FIELDS`, and `RESUME_FIELDS`.
    """
    trip_samples = truncate_driving(driving_samples, trip_minutes)
    last_coolant_c = float(trip_samples[-1]["coolant_c"])
    scenario = BatteryScenario(
        name=f"cal_a{ambient_c:g}_s{seed}", ambient_c=ambient_c,
        parasitic_a=parasitic_a, seed=seed,
    )
    state = CellState(temp_c=ambient_c)
    pattern = build_schedule(layup_gap_days, layup_days, horizon_days)
    rows = []
    for day, is_layup in enumerate(pattern):
        obs, hidden = step_one_day(
            state, scenario, trip_samples, rates, is_layup, trips_per_day,
            soak_h, last_coolant_c,
        )
        row = {"day": day}
        row.update(obs)
        row.update(hidden)
        rows.append(row)
    return rows


def run_planned_scenario(
    driving_samples: Sequence,
    plans: Sequence,
    rates: AgingRates,
    soak_h: float,
    seed: int,
    parasitic_a: float,
) -> list[dict]:
    """Round 12: one row per day, every day drawn from its own `DayPlan`.

    The round 4-11 path above holds trip length, ambient and accessory load
    fixed for a battery's whole life, so within one battery `ambient_c` had a
    single value and trip length at most two. Here each day carries its own
    dials, which is what makes trip-pattern counterfactuals expressible.

    Two details that are easy to get wrong and would quietly shrink the very
    effects this round exists to measure:

    - the drive is truncated **per day**, so a short day really is a short
      drive rather than the battery's habitual one;
    - `last_coolant_c` is carried from the last day that actually drove.
      Computing it once per battery (as the fixed path does) would hand every
      day the heat soak of a trip length it never took.
    """
    first = plans[0]
    scenario = BatteryScenario(
        name=f"cal_s{seed}", ambient_c=first.ambient_c,
        parasitic_a=parasitic_a, seed=seed,
    )
    state = CellState(temp_c=first.ambient_c)
    # Seeded from a full-length drive so day 0 has a coolant reading even if
    # the battery happens to start parked.
    last_coolant_c = float(truncate_driving(driving_samples, TRIP_MINUTES_SEED)[-1]["coolant_c"])

    rows = []
    for day, plan in enumerate(plans):
        day_scenario = replace(
            scenario, ambient_c=plan.ambient_c, accessory_base_a=plan.accessory_a,
        )
        if plan.trips:
            trip_samples = truncate_driving(driving_samples, plan.trip_minutes)
            last_coolant_c = float(trip_samples[-1]["coolant_c"])
        else:
            trip_samples = ()
        obs, hidden = step_one_day(
            state, day_scenario, trip_samples, rates, plan.is_layup, plan.trips,
            soak_h, last_coolant_c, accessory_a=plan.accessory_a,
        )
        row = {"day": day}
        row.update(obs)
        row.update(hidden)
        rows.append(row)
    return rows


def roll_planned_days(
    state: CellState,
    scenario: BatteryScenario,
    driving_samples: Sequence,
    rates: AgingRates,
    plans: Sequence,
    soak_h: float,
    last_coolant_c: float,
) -> list[dict]:
    """`roll_days` for the round 12 action space: a sequence of `DayPlan`s.

    The counterfactual replay path. `state` is rebuilt from a factual
    scenario's true hidden state at some day, and `plans` is a DIFFERENT
    action sequence than what happened -- for round 12 that is
    `experiment.schedule.split_trips`, which holds total driving minutes fixed
    and doubles the cold starts. Same ODE, same starting hidden state,
    different actions: Pearl's action step, with `state` supplying abduction.
    """
    rows = []
    for day, plan in enumerate(plans):
        day_scenario = replace(
            scenario, ambient_c=plan.ambient_c, accessory_base_a=plan.accessory_a,
        )
        if plan.trips:
            trip_samples = truncate_driving(driving_samples, plan.trip_minutes)
            last_coolant_c = float(trip_samples[-1]["coolant_c"])
        else:
            trip_samples = ()
        obs, hidden = step_one_day(
            state, day_scenario, trip_samples, rates, plan.is_layup, plan.trips,
            soak_h, last_coolant_c, accessory_a=plan.accessory_a,
        )
        row = {"day": day}
        row.update(obs)
        row.update(hidden)
        rows.append(row)
    return rows


def roll_days(
    state: CellState,
    scenario: BatteryScenario,
    trip_samples: Sequence,
    rates: AgingRates,
    day_types: Sequence[bool],
    trips_per_day: float,
    soak_h: float,
    last_coolant_c: float,
) -> list[dict]:
    """Roll `state` forward under an arbitrary day-type sequence.

    Used by the abduction test to compute the exact ground-truth
    counterfactual: `state` is reconstructed from a factual scenario's true
    hidden state at some day, and `day_types` is a DIFFERENT action sequence
    (e.g. the factual pattern with layup/driving flipped) than what actually
    happened. Same underlying ODE, same starting hidden state, different
    action -- exactly Pearl's action step after `state` supplies abduction.
    """
    rows = []
    for day, is_layup in enumerate(day_types):
        obs, hidden = step_one_day(
            state, scenario, trip_samples, rates, is_layup, trips_per_day,
            soak_h, last_coolant_c,
        )
        row = {"day": day}
        row.update(obs)
        row.update(hidden)
        rows.append(row)
    return rows


def rebuild_scenario(ambient_c: float, parasitic_a: float, seed: int) -> BatteryScenario:
    """Reconstruct the exact same `BatteryScenario` a dataset row came from.

    `BatteryScenario` is frozen and fully determined by its fields, so this
    is just re-applying `run_calendar_scenario`'s own construction -- kept as
    a named function so the abduction code does not have to restate the
    naming convention (`cal_a{ambient}_s{seed}`) inline.
    """
    return BatteryScenario(
        name=f"cal_a{ambient_c:g}_s{seed}", ambient_c=ambient_c,
        parasitic_a=parasitic_a, seed=seed,
    )
