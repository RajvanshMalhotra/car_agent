"""Round 12: per-day schedules drawn from a battery's habit."""

import random

import numpy as np
import pytest

from experiment.schedule import (
    ACCESSORY_RANGE_A,
    AMBIENT_CLAMP_C,
    AMPLITUDE_RANGE_C,
    TRIP_MINUTES_RANGE,
    DayPlan,
    Habit,
    draw_habit,
    plan_days,
    seasonal_ambient,
    split_trips,
)

HORIZON = 400


def _habit(seed=0):
    return draw_habit(random.Random(seed), annual_mean_c=25.0)


def test_seasonal_ambient_swings_around_the_annual_mean():
    days = np.arange(365)
    temps = np.array([seasonal_ambient(d, annual_mean_c=25.0, amplitude_c=10.0, phase_day=0) for d in days])
    assert temps.mean() == pytest.approx(25.0, abs=0.2)
    assert temps.max() == pytest.approx(35.0, abs=0.01)
    assert temps.min() == pytest.approx(15.0, abs=0.01)


def test_seasonal_ambient_repeats_every_year():
    for day in (0, 37, 200):
        a = seasonal_ambient(day, 25.0, 10.0, phase_day=60)
        b = seasonal_ambient(day + 365, 25.0, 10.0, phase_day=60)
        assert a == pytest.approx(b, abs=1e-9)


def test_a_battery_now_varies_within_its_own_life():
    """The whole point of round 12: rounds 4-11 had at most 2 distinct values."""
    plans = plan_days(_habit(), [False] * HORIZON, random.Random(1))
    assert len({p.trip_minutes for p in plans}) > 50
    assert len({round(p.ambient_c, 3) for p in plans}) > 50
    assert len({p.accessory_a for p in plans}) > 50
    assert len({p.trips for p in plans}) > 1


def test_layup_days_force_zero_trips():
    layup = [False] * 10 + [True] * 5 + [False] * 10
    plans = plan_days(_habit(), layup, random.Random(2))
    assert [p.trips for p in plans[10:15]] == [0, 0, 0, 0, 0]
    assert all(p.trips > 0 for p in plans[:10])


def test_every_drawn_day_stays_inside_the_physical_ranges():
    plans = plan_days(_habit(3), [False] * HORIZON, random.Random(3))
    assert all(TRIP_MINUTES_RANGE[0] <= p.trip_minutes <= TRIP_MINUTES_RANGE[1] for p in plans)
    assert all(ACCESSORY_RANGE_A[0] <= p.accessory_a <= ACCESSORY_RANGE_A[1] for p in plans)
    assert all(0 <= p.trips <= 3 for p in plans)


def test_no_drawn_day_can_leave_the_simulator_ambient_bounds():
    """The hottest habit plus its noise tail must stay inside load/spec.py's range.

    Measured before the clamp: 1600 draws reached 64.5 C, which
    BatteryScenario rejects -- mid-rebuild, minutes into a simulation.
    """
    from load.spec import AMBIENT_RANGE_C

    hottest = []
    for seed in range(120):
        habit = draw_habit(random.Random(seed), annual_mean_c=42.0)
        hottest += plan_days(habit, [False] * 400, random.Random(seed + 7))
    coldest = []
    for seed in range(120):
        habit = draw_habit(random.Random(seed), annual_mean_c=-10.0)
        coldest += plan_days(habit, [False] * 400, random.Random(seed + 7))

    temps = [p.ambient_c for p in hottest + coldest]
    assert max(temps) <= AMBIENT_CLAMP_C[1] and min(temps) >= AMBIENT_CLAMP_C[0]
    assert AMBIENT_RANGE_C[0] < min(temps) and max(temps) < AMBIENT_RANGE_C[1]


def test_habit_amplitude_is_drawn_in_range():
    habits = [draw_habit(random.Random(s), annual_mean_c=10.0) for s in range(20)]
    assert all(AMPLITUDE_RANGE_C[0] <= h.amplitude_c <= AMPLITUDE_RANGE_C[1] for h in habits)
    assert len({h.trip_minutes_mean for h in habits}) > 1, "batteries must still differ from each other"


def test_planning_is_deterministic_for_a_seed():
    a = plan_days(_habit(), [False] * 50, random.Random(7))
    b = plan_days(_habit(), [False] * 50, random.Random(7))
    assert a == b


def test_plans_round_trip_through_recorded_action_columns():
    """What the simulator emitted must rebuild into the plans it ran."""
    from experiment.schedule import actions_from_plans, plans_from_rows

    original = plan_days(_habit(5), [False, False, True, False], random.Random(5))
    rows = actions_from_plans(original, soak_h=8.0)
    arr = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    rebuilt = plans_from_rows(arr, 0, len(original))

    for before, after in zip(original, rebuilt):
        assert after.trips == before.trips
        assert after.trip_minutes == pytest.approx(before.trip_minutes, abs=1e-6)
        assert after.ambient_c == pytest.approx(before.ambient_c, abs=1e-6)
        assert after.accessory_a == pytest.approx(before.accessory_a, abs=1e-6)


def test_action_rows_match_what_the_simulator_emits():
    from experiment.schedule import actions_from_plans

    rows = actions_from_plans(
        [DayPlan(trips=0, trip_minutes=0.0, ambient_c=5.0, accessory_a=20.0),
         DayPlan(trips=2, trip_minutes=6.0, ambient_c=5.0, accessory_a=20.0)],
        soak_h=8.0,
    )
    assert rows[0]["is_layup"] == 1.0 and rows[0]["soak_hours"] == 24.0
    assert rows[0]["driving_minutes"] == 0.0 and rows[0]["trips_today"] == 0.0
    assert rows[1]["is_layup"] == 0.0 and rows[1]["soak_hours"] == 16.0
    assert rows[1]["driving_minutes"] == pytest.approx(12.0)
    assert rows[1]["trips_today"] == 2.0


def test_training_perturbation_differs_in_shape_from_the_eval_one():
    """Training must not be handed the literal test transformation."""
    from experiment.schedule import random_perturbation_plans

    plans = plan_days(_habit(6), [False] * 60, random.Random(6))
    perturbed = random_perturbation_plans(plans, random.Random(7))
    split = split_trips(plans)

    assert perturbed != split
    assert all(p.ambient_c == q.ambient_c for p, q in zip(plans, perturbed)), "weather is not a driver's choice"
    assert any(p.trips != q.trips for p, q in zip(plans, perturbed))
    assert all(TRIP_MINUTES_RANGE[0] <= p.trip_minutes <= TRIP_MINUTES_RANGE[1] for p in perturbed)
    assert all(ACCESSORY_RANGE_A[0] <= p.accessory_a <= ACCESSORY_RANGE_A[1] for p in perturbed)


def test_training_perturbation_leaves_parked_days_parked():
    from experiment.schedule import random_perturbation_plans

    plans = [DayPlan(trips=0, trip_minutes=0.0, ambient_c=5.0, accessory_a=20.0)]
    assert random_perturbation_plans(plans, random.Random(1)) == plans


def test_split_trips_keeps_total_driving_and_doubles_the_cold_starts():
    """The new eval counterfactual: same minutes, twice the trips."""
    plans = [DayPlan(trips=2, trip_minutes=10.0, ambient_c=25.0, accessory_a=22.0),
             DayPlan(trips=0, trip_minutes=0.0, ambient_c=25.0, accessory_a=22.0)]
    split = split_trips(plans)
    assert split[0].trips == 4
    assert split[0].trip_minutes == pytest.approx(5.0)
    assert split[0].trips * split[0].trip_minutes == pytest.approx(plans[0].trips * plans[0].trip_minutes)
    assert split[0].ambient_c == plans[0].ambient_c
    assert split[1] == plans[1], "a parked day has no trips to split"


def test_split_trips_respects_the_minimum_trip_length():
    plans = [DayPlan(trips=2, trip_minutes=TRIP_MINUTES_RANGE[0], ambient_c=25.0, accessory_a=22.0)]
    split = split_trips(plans)
    assert split[0].trip_minutes >= TRIP_MINUTES_RANGE[0]
    assert split[0] == plans[0], "a day already at the floor cannot be split further"
