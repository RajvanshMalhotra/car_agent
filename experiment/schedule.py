"""Per-day schedules, so a battery's own life varies instead of repeating.

Rounds 4-11 gave the world model one binary dial per day: within a single
battery, `ambient_c` had exactly one distinct value and trip length at most
two. Copying the latest day of the same type therefore scored 83%, and
"fewer short trips would have extended it by X%" -- a deliverable in
CLAUDE.md -- could not even be expressed.

This module draws each day from the battery's *habit*, so between-battery
identity survives (one owner does short hops, another commutes) while
within-battery variation finally exists. Nothing here needs new driving data:
trip length truncates the same recording, trips-per-day is a loop count, and
ambient and accessory load are scenario parameters.

`split_trips` is the new evaluation counterfactual. It holds total driving
minutes fixed and doubles the number of trips -- same distance, same duration,
twice the cold starts -- which isolates trip *pattern* from trip *amount*.
That is the claim; flipping a drive/park bit never could be.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

#: Trip length, in minutes of the recorded drive. The upper bound is the whole
#: recording (~20 min); below the lower bound a "trip" is barely a crank.
TRIP_MINUTES_RANGE = (1.5, 20.0)

#: Total accessory draw while running: ECU and fuel pump at the low end,
#: lights plus blower plus AC at the high end (`load/spec.py`).
ACCESSORY_RANGE_A = (12.0, 55.0)

#: Seasonal swing about a battery's annual mean. A mild maritime climate sits
#: near the bottom, a continental one near the top.
AMPLITUDE_RANGE_C = (5.0, 15.0)

#: Trips on a driving day. 0 is reserved for layup, which the schedule forces.
MAX_TRIPS = 3

#: Every drawn day is clamped inside this, comfortably within `load/spec.py`'s
#: AMBIENT_RANGE_C of (-40, 60). Without it the tails compound -- a 42 C annual
#: mean plus a 15 C amplitude plus daily noise reached 64.5 C over 1600 draws,
#: which `BatteryScenario.__post_init__` rejects, and a rebuild would have died
#: partway through a 20-minute simulation rather than here.
AMBIENT_CLAMP_C = (-35.0, 55.0)

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Habit:
    """What makes one owner different from another, held for life."""

    trip_minutes_mean: float
    trip_minutes_spread: float
    trips_mean: float
    accessory_mean_a: float
    accessory_spread_a: float
    annual_mean_c: float
    amplitude_c: float
    phase_day: float
    daily_noise_c: float


@dataclass(frozen=True)
class DayPlan:
    """One calendar day's dials -- the row the world model sees as actions."""

    trips: int
    trip_minutes: float
    ambient_c: float
    accessory_a: float

    @property
    def is_layup(self) -> bool:
        return self.trips == 0

    @property
    def driving_minutes(self) -> float:
        return 0.0 if self.trips == 0 else self.trips * self.trip_minutes


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def draw_habit(rng: random.Random, annual_mean_c: float) -> Habit:
    """One battery's lifelong habit, around a given annual mean temperature."""
    return Habit(
        trip_minutes_mean=rng.uniform(*TRIP_MINUTES_RANGE),
        trip_minutes_spread=rng.uniform(0.15, 0.55),
        trips_mean=rng.uniform(0.8, 2.6),
        accessory_mean_a=rng.uniform(*ACCESSORY_RANGE_A),
        accessory_spread_a=rng.uniform(2.0, 9.0),
        annual_mean_c=annual_mean_c,
        amplitude_c=rng.uniform(*AMPLITUDE_RANGE_C),
        phase_day=rng.uniform(0.0, DAYS_PER_YEAR),
        daily_noise_c=rng.uniform(0.5, 3.0),
    )


def seasonal_ambient(day: int, annual_mean_c: float, amplitude_c: float, phase_day: float) -> float:
    """A year-long sinusoid: the same battery now sees summer and winter."""
    return annual_mean_c + amplitude_c * math.sin(2.0 * math.pi * (day - phase_day) / DAYS_PER_YEAR)


def plan_days(habit: Habit, layup_schedule: list[bool], rng: random.Random) -> list[DayPlan]:
    """One `DayPlan` per calendar day, drawn from `habit`.

    `layup_schedule` comes from `experiment.calendar_sim.build_schedule` and
    still forces runs of parked days; every other day draws its own trip count,
    trip length, temperature and accessory load.
    """
    plans: list[DayPlan] = []
    for day, parked in enumerate(layup_schedule):
        ambient = seasonal_ambient(day, habit.annual_mean_c, habit.amplitude_c, habit.phase_day)
        ambient = _clip(ambient + rng.gauss(0.0, habit.daily_noise_c), AMBIENT_CLAMP_C)
        accessory = _clip(rng.gauss(habit.accessory_mean_a, habit.accessory_spread_a), ACCESSORY_RANGE_A)
        if parked:
            plans.append(DayPlan(trips=0, trip_minutes=0.0, ambient_c=ambient, accessory_a=accessory))
            continue
        trips = min(MAX_TRIPS, max(1, round(rng.gauss(habit.trips_mean, 0.7))))
        minutes = _clip(
            rng.gauss(habit.trip_minutes_mean, habit.trip_minutes_mean * habit.trip_minutes_spread),
            TRIP_MINUTES_RANGE,
        )
        plans.append(DayPlan(trips=trips, trip_minutes=minutes, ambient_c=ambient, accessory_a=accessory))
    return plans


def split_trips(plans: list[DayPlan]) -> list[DayPlan]:
    """The evaluation counterfactual: same driving minutes, twice the trips.

    A parked day has nothing to split, and a day already at the minimum trip
    length cannot be halved without inventing a trip shorter than the recording
    supports -- both are returned unchanged, so the perturbation never leaves
    the physical envelope.
    """
    out: list[DayPlan] = []
    for plan in plans:
        halved = plan.trip_minutes / 2.0
        if plan.trips == 0 or halved < TRIP_MINUTES_RANGE[0]:
            out.append(plan)
            continue
        out.append(replace(plan, trips=plan.trips * 2, trip_minutes=halved))
    return out
