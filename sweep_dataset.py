#!/usr/bin/env python3
"""Sweep duty cycles over a recorded trajectory and emit one combined dataset.

`battery_run.py` answers "what does THIS scenario do to the battery". This
answers "what does the space of scenarios do", which is what a model needs if
it is to generalise rather than memorise one operating point.

Three tables come out, for the same reason `cell/dataset.py` emits two: the
1 Hz rows, the per-scenario life, and the per-scenario ageing trajectory are
three different granularities, and broadcasting a per-scenario constant
across thousands of near-identical rows is recoverable from the row index.

    within_trip.csv     1 Hz, every (ambient, trip_minutes) combination,
                        tagged with `scenario`. The driving channels are
                        identical across scenarios sharing an (ambient,
                        trip_minutes) pair by construction -- the same
                        recording is replayed, truncated the same way -- so
                        what varies is the battery response.
    scenario_life.csv   one row per full scenario: the declared axes, the
                        damage it accumulated per day, the terminal ageing
                        state, and where end of life landed.
    aging_trajectory.csv one row per scenario per sampled day: how the hidden
                        ageing state (crystal, corrosion_hours, shedding) and
                        soc/soh moved over the projection, not just where
                        they ended up. This is what a counterfactual rollout
                        gets compared against.

**Why sulfation needed new axes, not just more of the old ones.** A 20-minute
trip fully recovers its own crank and charge acceptance self-stabilises near
0.99, so `low_soc_hours` -- the only thing that makes the sulfation crystal
state move at all -- was identically zero across the original ambient x
schedule x accessories grid. None of those three axes ever depresses SoC
below `LOW_SOC` on the driving side. Four things do, and they are the new
axes below: very short trips that do not recover their own crank
(`TRIP_MINUTES`), an occasional long park (`LAYUP_DAYS`), a heavier key-off
draw (`PARASITIC_A`), and cold ambient collapsing charge acceptance (folded
into the trimmed `AMBIENTS_C`). `ACCESSORIES` is dropped entirely -- it moved
life by about two days out of a thousand in the prior sweep, i.e. noise, and
every day of it spent looping is a day not spent on axes that move sulfation.
`SCHEDULES` is fixed to one representative pattern rather than swept, for the
same budget reason -- see the axis constants below for what stays and why.

Absolute days remain unfitted. Ratios between rows are the usable output.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from cell.aging import AgingRates, AgingState, soh
from cell.dataset import FEATURES, TARGETS
from cell.integrate import CellState, run_soak, run_trip
from cell.life import TripSchedule, ensemble
from cell.sensors import SensorNoise
from load.legacy import LEGACY_PROVENANCE, read_legacy
from load.spec import BatteryScenario
from load.trips import segment

#: The dominant ageing axis, per the project's own measurements. Trimmed from
#: five points to four, but widened downward: -10 C is cold enough to
#: collapse charge acceptance (`load/electrical.py`'s `acceptance_temperature_
#: factor` reaches its floor by -10 C), which is what lets a short, cold trip
#: fail to recover its own crank -- ROUTE 3 in the driving brief. 50 C was
#: dropped as redundant with 42 C for corrosion purposes; both are well past
#: the Arrhenius knee and the ratio between them is not where the interesting
#: behaviour is.
AMBIENTS_C = (-10.0, 10.0, 25.0, 42.0)

#: How much of the recorded ~1192 s (20 min) trip is actually driven before
#: the engine is shut off again. Short trips do not run long enough for
#: charge acceptance to put back what the crank took -- this is the ROUTE 2
#: pathway, and it was entirely absent from the original sweep because every
#: scenario replayed the full recording.
TRIP_MINUTES = (2.0, 5.0, 10.0, 20.0)

#: An occasional long park on top of the normal daily soak, e.g. a business
#: trip, holiday, or seasonal layup -- ROUTE 1. 0 means the car is never left
#: standing beyond its normal daily soak.
LAYUP_DAYS = (0.0, 7.0, 21.0)

#: How often a layup happens, in days. `run_scenario` folds "one `layup_days`
#: -long park every `LAYUP_PERIOD_DAYS`" into a single per-day-average damage
#: figure -- see its docstring for why an average, not a literal calendar
#: simulation, is what `cell/life.py.project()`'s per-day abstraction can
#: consume. 60 days (about two months) keeps the parked fraction of time
#: plausible: 7/60 = 12% for the light case, 21/60 = 35% for the heavy one --
#: a lot, but this axis exists specifically to reach the continuous-park
#: regime ROUTE 1 was measured in, not to be subtle about it.
LAYUP_PERIOD_DAYS = 60.0

#: Coarse enough to keep a 21-day layup's soak integration cheap (6048 steps
#: instead of 181440 at the usual 10 s), fine enough that the bay's ~60 s time
#: constant is still resolved for the first hour, which is all that matters
#: on a soak lasting weeks.
LAYUP_SOAK_DT_S = 300.0

#: Key-off draw: alarm, clock, ECU keep-alive -- ROUTE 4. 0.03 A is
#: `BatteryScenario`'s own default (an old, well-behaved car); 0.08 A is a
#: modern one with more always-on electronics. The brief's own numbers show
#: this axis is one of the strongest levers on parked SoC (0.03 A/14 d -> 0.832,
#: 0.08 A/14 d -> 0.552), stronger than trip length or layup duration, so it
#: gets two points despite the tight scenario budget.
PARASITIC_A = (0.03, 0.08)

#: Fixed rather than swept: a plain daily commute, two trips with an 8 h soak
#: between them. `SCHEDULES` varied trip frequency and soak length in the
#: original sweep, but neither axis ever moved `low_soc_hours` off zero --
#: the actual sulfation-relevant lever is TRIP_MINUTES (whether a trip
#: recovers its own crank at all), not how many such trips happen per day.
#: Keeping one representative schedule buys the budget for the three new
#: axes instead of a fourth cross-product dimension that does not move the
#: pathway this sweep exists to unlock.
SCHEDULE_TRIPS_PER_DAY = 2.0
SCHEDULE_SOAK_H = 8.0

WITHIN_COLUMNS = (
    ("scenario", "ambient_c", "trip_minutes", "trip", "t_s") + FEATURES + TARGETS
)

LIFE_COLUMNS = (
    "scenario", "ambient_c", "trip_minutes", "layup_days", "parasitic_a",
    "trips_per_day", "soak_h",
    "corrosion_equivalent_h_per_day", "ah_throughput_per_day",
    "low_soc_hours_per_day", "full_charge_hours_per_day", "vibration_dose_per_day",
    "soc_after_repeats", "soh_after_repeats",
    "corrosion_hours", "crystal", "shedding",
    "eol_days", "eol_years", "interval_low_days", "interval_high_days",
    "scale_unfitted",
)

#: One row per scenario per sampled day of the life projection -- the
#: trajectory a counterfactual rollout gets compared against, not just the
#: terminal state `scenario_life.csv` carries. Sampled every
#: `cell.life.CURVE_INTERVAL_DAYS` (30) by `cell.life.project`.
AGING_TRAJECTORY_COLUMNS = (
    "scenario", "day", "soc", "crystal", "corrosion_hours", "shedding", "soh",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_driving(path: Path):
    """The driving-phase samples at 1 Hz, dropouts already excluded."""
    trajectory = read_legacy(path)
    samples = list(trajectory.at_hz(1.0))
    segments = segment(iter(samples))
    driving = [
        s for s in samples
        if any(g.kind == "trip" and g.start_s <= s["t_s"] <= g.end_s for g in segments)
    ]
    return driving, trajectory.dropouts.spans


def truncate_driving(driving, minutes):
    """The first `minutes` of a trip. 20 covers the whole ~1192 s recording.

    A short trip is not a different recording, it is the same one cut off
    early -- the whole point of the ROUTE 2 pathway is that the driver did
    the same commute but for less time, not a different commute.
    """
    if not driving:
        return driving
    start_t = float(driving[0]["t_s"])
    cutoff = minutes * 60.0
    return [s for s in driving if float(s["t_s"]) - start_t < cutoff]


def run_scenario(driving, ambient_c, trips_per_day, soak_h, trip_minutes,
                 layup_days, parasitic_a, repeats, rates, draws, seed):
    """One scenario: integrate `repeats` trips, then project a life.

    `layup_days` folds an occasional long park into the per-day damage
    average -- see the module docstring and `LAYUP_PERIOD_DAYS` above for
    why an average rather than a literal calendar simulation.
    """
    trip_samples = truncate_driving(driving, trip_minutes)
    scenario = BatteryScenario(
        name=f"a{ambient_c:g}", ambient_c=ambient_c, parasitic_a=parasitic_a,
        seed=seed,
    )
    schedule = TripSchedule(trips_per_day=trips_per_day, soak_s=soak_h * 3600.0)
    last_coolant = float(trip_samples[-1]["coolant_c"])

    state = CellState(temp_c=ambient_c)
    trips = []
    for index in range(repeats):
        result = run_trip(iter(trip_samples), scenario, state, rates, cranked=index > 0)
        trips.append(result)
        if index < repeats - 1:
            run_soak(schedule.soak_s, scenario, state, rates,
                     initial_coolant_c=last_coolant)

    def normal_day(day_start):
        """One ordinary day: `trips_per_day` trips plus the scheduled soak."""
        probe = CellState(
            soc=day_start.soc, temp_c=ambient_c, aging=state.aging.copy()
        )
        trip = run_trip(iter(trip_samples), scenario, probe, rates, cranked=True,
                        health=day_start.health).damage
        soak = run_soak(schedule.soak_s, scenario, probe, rates,
                        initial_coolant_c=last_coolant,
                        health=day_start.health).damage
        damage = trip.scaled(trips_per_day) + soak.scaled(trips_per_day)
        cycle_delta = probe.soc - day_start.soc
        soc_end = max(0.0, min(1.0, day_start.soc + cycle_delta * trips_per_day))
        return damage, soc_end

    if layup_days <= 0.0:
        day_damage = normal_day
    else:
        def day_damage(day_start):
            """A period average: ordinary days, plus one long park.

            `cell.life.project` advances health one calendar day per call and
            holds whatever `Damage` a probe returns fixed until the next
            resim -- it has no notion of "this specific day is the layup
            day". So a periodic occasional-long-park pattern is folded into a
            single per-day-average `Damage` here instead: simulate the
            `LAYUP_PERIOD_DAYS - layup_days` ordinary days at the steady
            daily SoC (one call stands in for all of them, since consecutive
            ordinary days converge to nearly the same SoC), then the
            continuous `layup_days`-long park itself, sum the damage over the
            whole period and divide by its length.

            This does NOT flatten the hidden state itself: `accumulate()`
            still runs once per calendar day inside `project`'s own loop, so
            `crystal` still grows and saturates day by day against whatever
            average rate this returns. Only the INPUT rate is smoothed across
            the period, not the state evolution -- the thing the downstream
            counterfactual experiment needs to stay path-dependent.

            The returned SoC is the value right after the park: the most
            depleted point in the period, and the point the layup's own
            `low_soc_hours` was actually earned against.
            """
            normal_damage, soc_after_normal = normal_day(day_start)
            ordinary_days = max(0.0, LAYUP_PERIOD_DAYS - layup_days)
            total = normal_damage.scaled(ordinary_days)

            park = CellState(
                soc=soc_after_normal, temp_c=ambient_c, aging=state.aging.copy()
            )
            parked = run_soak(
                layup_days * 86400.0, scenario, park, rates,
                initial_coolant_c=last_coolant, dt_s=LAYUP_SOAK_DT_S,
                health=day_start.health,
            )
            total = total + parked.damage
            avg_damage = total.scaled(1.0 / LAYUP_PERIOD_DAYS)
            return avg_damage, park.soc

    life = ensemble(day_damage, rates, schedule, draws=draws, seed=seed,
                    soc0=state.soc)
    # `life.avg_damage_per_day` -- what was actually integrated, averaged over
    # the whole projected life -- not a single day-0 snapshot at health=1.0.
    # Health degrades over the life and capacity shrinks with it, so a
    # pathway invisible on day 0 can still show up later; a snapshot would
    # silently miss it. See cell/life.py's docstring on the field.
    per_day = life.avg_damage_per_day
    return scenario, schedule, trips, life, per_day, state, trip_samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--out", default="runs/sweep")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--one", action="store_true",
                        help="run a single scenario and report timing only")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="RNG seed for the sensor-noise layer")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="sensor-noise sigma multiplier; 0.0 (default) is "
                             "off, 1.0 is the declared sigmas, 2.0 is 2x, etc.")
    args = parser.parse_args(argv)

    driving, dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()
    print(f"driving samples at 1 Hz: {len(driving)}", flush=True)

    if args.one:
        start = time.time()
        run_scenario(driving, 25.0, SCHEDULE_TRIPS_PER_DAY, SCHEDULE_SOAK_H,
                     20.0, 21.0, 0.08, args.repeats, rates, args.draws, 0)
        print(f"one scenario (layup) took {time.time() - start:.1f} s", flush=True)
        total = len(AMBIENTS_C) * len(TRIP_MINUTES) * len(LAYUP_DAYS) * len(PARASITIC_A)
        print(f"full sweep is {total} scenarios "
              f"~= {(time.time() - start) * total / 60:.1f} min", flush=True)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    within_rows, life_rows, aging_rows = [], [], []
    seen_within = set()
    index = 0
    started = time.time()

    # Noise applies ONLY to within_trip.csv, the emitted 1 Hz "sensor" rows.
    # scenario_life.csv and aging_trajectory.csv are model outputs (damage,
    # projected end of life, hidden ageing state), not measurements, so they
    # stay exact regardless of --noise-scale.
    noise = SensorNoise(seed=args.noise_seed, scale=args.noise_scale)

    total_scenarios = len(AMBIENTS_C) * len(TRIP_MINUTES) * len(LAYUP_DAYS) * len(PARASITIC_A)

    for ambient_c in AMBIENTS_C:
        for trip_minutes in TRIP_MINUTES:
            for layup_days in LAYUP_DAYS:
                for parasitic_a in PARASITIC_A:
                    name = (
                        f"amb{ambient_c:g}_trip{trip_minutes:g}m_"
                        f"layup{layup_days:g}d_par{parasitic_a:g}A"
                    )
                    scenario, schedule, trips, life, per_day, state, trip_samples = (
                        run_scenario(
                            driving, ambient_c, SCHEDULE_TRIPS_PER_DAY,
                            SCHEDULE_SOAK_H, trip_minutes, layup_days,
                            parasitic_a, args.repeats, rates, args.draws, index,
                        )
                    )
                    health = soh(state.aging, rates)
                    low, high = (life.interval_days or (None, None))
                    # `state.aging` is only the warm-up state after `repeats`
                    # trips -- a few days old, not a life. The TERMINAL ageing
                    # state (what the acceptance criterion actually asks for)
                    # is where the point-estimate projection landed, at end
                    # of life or the horizon: the last sample of its own
                    # state_curve, which cell.life.project() populates for
                    # exactly this reason.
                    _, _, terminal_crystal, terminal_corrosion_h, terminal_shedding, _ = (
                        life.state_curve[-1]
                    )
                    life_rows.append({
                        "scenario": name, "ambient_c": ambient_c,
                        "trip_minutes": trip_minutes, "layup_days": layup_days,
                        "parasitic_a": parasitic_a,
                        "trips_per_day": SCHEDULE_TRIPS_PER_DAY,
                        "soak_h": SCHEDULE_SOAK_H,
                        "corrosion_equivalent_h_per_day": per_day.corrosion_equivalent_h,
                        "ah_throughput_per_day": per_day.ah_throughput,
                        "low_soc_hours_per_day": per_day.low_soc_hours,
                        "full_charge_hours_per_day": per_day.full_charge_hours,
                        "vibration_dose_per_day": per_day.vibration_dose,
                        "soc_after_repeats": state.soc,
                        "soh_after_repeats": health,
                        "corrosion_hours": terminal_corrosion_h,
                        "crystal": terminal_crystal,
                        "shedding": terminal_shedding,
                        "eol_days": "" if life.eol_days is None else life.eol_days,
                        "eol_years": (
                            "" if life.eol_days is None
                            else round(life.eol_days / 365.0, 3)
                        ),
                        "interval_low_days": "" if low is None else low,
                        "interval_high_days": "" if high is None else high,
                        "scale_unfitted": int(life.scale_unfitted),
                    })

                    for day, soc, crystal, corrosion_hours, shedding, soh_val in (
                        life.state_curve
                    ):
                        aging_rows.append({
                            "scenario": name, "day": day, "soc": soc,
                            "crystal": crystal, "corrosion_hours": corrosion_hours,
                            "shedding": shedding, "soh": soh_val,
                        })

                    # The 1 Hz rows depend only on ambient and trip length --
                    # layup and parasitic draw affect only the soak, which
                    # never appears in a "trip" row -- so emit them once per
                    # (ambient, trip_minutes) pair rather than once per
                    # scenario.
                    key = (ambient_c, trip_minutes)
                    if key not in seen_within:
                        seen_within.add(key)
                        wname = f"amb{ambient_c:g}_trip{trip_minutes:g}m"
                        for trip_index, result in enumerate(trips):
                            for sample, step in zip(trip_samples, result.steps):
                                row = {
                                    "scenario": wname, "ambient_c": ambient_c,
                                    "trip_minutes": trip_minutes,
                                    "trip": trip_index, "t_s": step.t_s,
                                }
                                for column in FEATURES:
                                    row[column] = (
                                        step.t_bay_c if column == "t_bay_c"
                                        else float(sample[column])
                                    )
                                row.update({
                                    "i_bat_a": step.i_bat_a, "v_bat_v": step.v_bat_v,
                                    "t_bat_c": step.t_bat_c, "soc": step.soc,
                                    "r_int_ohm": step.r_int_ohm,
                                })
                                within_rows.append(noise.apply(row))
                    index += 1
                    print(f"  [{index:3d}/{total_scenarios}] {name:38s} "
                          f"eol={life.eol_days} days low_soc_h/d="
                          f"{per_day.low_soc_hours:.3f} terminal_crystal="
                          f"{terminal_crystal:.5f}",
                          flush=True)

    for path, columns, rows in (
        (out / "within_trip.csv", WITHIN_COLUMNS, within_rows),
        (out / "scenario_life.csv", LIFE_COLUMNS, life_rows),
        (out / "aging_trajectory.csv", AGING_TRAJECTORY_COLUMNS, aging_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[c] for c in columns])

    reached = [r for r in life_rows if r["eol_days"] != ""]
    low_soc_scenarios = [r for r in life_rows if r["low_soc_hours_per_day"] > 0.0]
    crystals = [r["crystal"] for r in life_rows]
    sidecar = {
        "git_commit": _git_commit(),
        "trajectory": str(args.trajectory),
        "repeats_per_scenario": args.repeats,
        "ensemble_draws": args.draws,
        "scenarios": len(life_rows),
        "within_trip_rows": len(within_rows),
        "aging_trajectory_rows": len(aging_rows),
        "dropouts_excluded": [
            {"start_s": s.start_s, "end_s": s.end_s, "rows": s.rows} for s in dropouts
        ],
        "axes": {
            "ambient_c": list(AMBIENTS_C),
            "trip_minutes": list(TRIP_MINUTES),
            "layup_days": list(LAYUP_DAYS),
            "parasitic_a": list(PARASITIC_A),
            "fixed_schedule_trips_per_day_soak_h": [
                SCHEDULE_TRIPS_PER_DAY, SCHEDULE_SOAK_H
            ],
            "layup_period_days": LAYUP_PERIOD_DAYS,
        },
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "sulfation": {
            "scenarios_with_nonzero_low_soc_hours_per_day": len(low_soc_scenarios),
            "fraction_with_nonzero_low_soc_hours_per_day": (
                round(len(low_soc_scenarios) / len(life_rows), 3) if life_rows else None
            ),
            "low_soc_hours_per_day_range": (
                [min(r["low_soc_hours_per_day"] for r in low_soc_scenarios),
                 max(r["low_soc_hours_per_day"] for r in low_soc_scenarios)]
                if low_soc_scenarios else None
            ),
            "terminal_crystal_range": (
                [min(crystals), max(crystals)] if crystals else None
            ),
        },
        "life_spread": (
            {"min_days": min(r["eol_days"] for r in reached),
             "max_days": max(r["eol_days"] for r in reached),
             "ratio": round(max(r["eol_days"] for r in reached)
                            / min(r["eol_days"] for r in reached), 3)}
            if reached else None
        ),
        "ageing_rates": {
            "corrosion_eol_h": rates.corrosion_eol_h,
            "corrosion_exponent": rates.corrosion_exponent,
            "sulfation_weight": rates.sulfation_weight,
            "sulfation_per_low_soc_h": rates.sulfation_per_low_soc_h,
            "sulfation_recovery_per_full_h": rates.sulfation_recovery_per_full_h,
            "fitted": rates.fitted,
        },
        "channels": dict(LEGACY_PROVENANCE),
        "sensor_noise": noise.manifest(),
        "caveats": [
            "Absolute days are NOT validated. No battery in this project reached "
            "end of life and no full-life dataset has been fitted against, so "
            "nothing sets the absolute rate. Ratios between scenarios cancel the "
            "unfitted constant and are the usable output.",
            "The ageing rate constant was derived from engine-BAY temperatures "
            "while corrosion is evaluated at BATTERY temperature, which lags by "
            "C_th/h ~ 7500 s. Measured exposure is 0.32x what the constant "
            "assumed, so absolute lives read roughly 3x long.",
            "Driving channels are FEATURES and battery channels are TARGETS. "
            "Within a trip soc = soc_0 - cumsum(i_bat_a*dt)/Q by construction, so "
            "offering current alongside soc hands a model the identity.",
            "t_bat_c tracks t_bay_c with no other forcing of consequence; "
            "predicting one from the other is close to trivial and the "
            "exact-match leakage detector will not flag it.",
            "Within an (ambient, trip_minutes) pair the driving channels are "
            "IDENTICAL across scenarios -- one recording, truncated the same "
            "way, replayed under different layup and parasitic conditions. "
            "This is an ambient/trip-length/layup/parasitic sweep, not a "
            "driving-style sweep.",
            "ACCESSORIES (hvac/lights) was DROPPED as a swept axis. In the "
            "prior 40-scenario sweep it moved life by about two days out of a "
            "thousand -- noise, not signal -- and every scenario spent on it "
            "was a scenario not spent on an axis that moves sulfation. It is "
            "held fixed off (hvac=0, lights=False) in every scenario here.",
            "SCHEDULES (trip frequency x soak length) was likewise fixed to a "
            "single representative pattern (2 trips/day, 8 h soak) rather than "
            "swept. Neither axis of the original 4-point sweep ever moved "
            "low_soc_hours off zero; TRIP_MINUTES is the lever that actually "
            "controls whether a trip recovers its own crank, so the scenario "
            "budget went there instead of a fourth cross-product dimension.",
            f"LAYUP_DAYS is folded into day_damage as a PERIOD AVERAGE, not a "
            f"literal calendar simulation: a {LAYUP_PERIOD_DAYS:g}-day period "
            f"of ordinary driving days plus one continuous layup_days-long park "
            f"is integrated once, and the total damage is divided by the "
            f"period length before being handed to cell.life.project(), which "
            f"has no notion of 'this specific day is the layup day'. The "
            f"hidden crystal state still accumulates and saturates day by day "
            f"inside project()'s own loop -- only the INPUT rate is smoothed, "
            f"not the state evolution. A true day-by-day calendar simulation "
            f"of the periodic pattern would show sharper, spikier low_soc_hours "
            f"concentrated in the actual layup days rather than a smooth "
            f"average; this sweep's low_soc_hours_per_day should be read as "
            f"the long-run average rate, not a literal single day's value.",
            "Every *_per_day column in scenario_life.csv is life.avg_damage_"
            "per_day: what cell.life.project() actually integrated into the "
            "ageing state, averaged over every calendar day of the projection "
            "(to end of life, or the horizon) -- NOT a single day-0 snapshot "
            "at health=1.0. This matters because capacity shrinks with health "
            "(cell/integrate.py), so a pathway that is invisible at the start "
            "of life (say, low_soc_hours == 0 while the battery is new) can "
            "still engage later as the battery ages and its shrunken capacity "
            "no longer absorbs the same crank; averaging over the whole life "
            "reports that, a day-0 snapshot would have missed it entirely.",
            "Sensor noise (see sensor_noise above) simulates measurement error "
            "on MEASURABLE channels only (speed_mps, rpm, throttle, coolant_c, "
            "i_bat_a, v_bat_v, t_bat_c) in within_trip.csv; LATENT channels "
            "(soc, r_int_ohm, t_bay_c, soh) and every scenario_life.csv and "
            "aging_trajectory.csv figure stay exact. This does not make the "
            "underlying data any less deterministic -- it is still a physics "
            "model, and a large enough network can still learn it from enough "
            "noisy rows.",
        ],
    }
    (out / "dataset.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    print(f"\nwrote {out}/within_trip.csv       ({len(within_rows)} rows)")
    print(f"wrote {out}/scenario_life.csv    ({len(life_rows)} scenarios)")
    print(f"wrote {out}/aging_trajectory.csv ({len(aging_rows)} rows)")
    print(f"wrote {out}/dataset.json")
    if reached:
        print(f"life spread across scenarios: {sidecar['life_spread']['ratio']}x "
              f"({sidecar['life_spread']['min_days']:.0f} to "
              f"{sidecar['life_spread']['max_days']:.0f} days)")
    print(f"scenarios with non-zero low_soc_hours_per_day: "
          f"{len(low_soc_scenarios)}/{len(life_rows)} "
          f"({sidecar['sulfation']['fraction_with_nonzero_low_soc_hours_per_day']})")
    if crystals:
        print(f"terminal crystal range: {min(crystals):.5f} to {max(crystals):.5f}")
    print(f"elapsed {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
