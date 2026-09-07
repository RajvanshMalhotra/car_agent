"""Learning controller gains per vehicle.

The controller's gains were hand-tuned for a passenger car. On a 6x6 truck they
are wrong, and wrong gains are what wandering actually is. Calibration measures
the vehicle's limits; this learns the gains that suit them.

Policy search (cross-entropy method) over five continuous parameters, scored by
how well a whole run tracks its route and holds its speed. It trains against the
fake backend configured with the *measured* limits of the real vehicle, so a few
hundred episodes take seconds rather than a week of BeamNG time -- which is the
only reason a learned controller is affordable here at all.
"""

import pytest

from control.calibration import VehicleLimits
from control.tuner import ControllerGains, evaluate, tune
from tests.helpers import a_behaviour_spec

CAR = VehicleLimits("car", max_accel_mps2=3.0, max_decel_mps2=8.0, wheelbase_m=2.7)
TRUCK = VehicleLimits("truck", max_accel_mps2=1.2, max_decel_mps2=4.0, wheelbase_m=6.5)


def test_default_gains_are_within_their_bounds():
    assert ControllerGains.default().is_valid()


def test_gains_outside_the_bounds_are_invalid():
    assert not ControllerGains(kp=-5.0, ki=0.0, kd=0.0,
                               lookahead_gain_s=0.6, min_lookahead_m=5.0).is_valid()


def test_a_score_is_produced_for_the_default_gains():
    score = evaluate(ControllerGains.default(), a_behaviour_spec(), CAR, seconds=40.0)
    assert score < 0.0  # scores are penalties, so higher is better


def test_wild_gains_score_worse_than_sensible_ones():
    sensible = evaluate(ControllerGains.default(), a_behaviour_spec(), CAR, seconds=40.0)
    wild = evaluate(
        ControllerGains(kp=9.0, ki=4.0, kd=2.5, lookahead_gain_s=0.05,
                        min_lookahead_m=1.0),
        a_behaviour_spec(), CAR, seconds=40.0,
    )
    assert wild < sensible


def test_scoring_is_deterministic():
    gains = ControllerGains.default()
    first = evaluate(gains, a_behaviour_spec(), CAR, seconds=30.0, seed=4)
    assert evaluate(gains, a_behaviour_spec(), CAR, seconds=30.0, seed=4) == first


def test_tuning_beats_the_hand_tuned_defaults_on_an_unfamiliar_vehicle():
    # The point of the exercise: gains chosen for a car are not right for a
    # 6.5 m wheelbase that accelerates at 1.2 m/s^2.
    spec = a_behaviour_spec()
    before = evaluate(ControllerGains.default(), spec, TRUCK, seconds=40.0)
    learned = tune(spec, TRUCK, iterations=4, population=12, seconds=40.0, seed=1)
    assert evaluate(learned, spec, TRUCK, seconds=40.0) > before


def test_the_learned_gains_are_valid():
    learned = tune(a_behaviour_spec(), TRUCK, iterations=3, population=10,
                   seconds=30.0, seed=2)
    assert learned.is_valid()


def test_the_same_seed_learns_the_same_gains():
    kwargs = dict(iterations=3, population=10, seconds=30.0)
    first = tune(a_behaviour_spec(), TRUCK, seed=7, **kwargs)
    assert tune(a_behaviour_spec(), TRUCK, seed=7, **kwargs) == first


def test_different_seeds_explore_differently():
    kwargs = dict(iterations=3, population=10, seconds=30.0)
    assert tune(a_behaviour_spec(), TRUCK, seed=1, **kwargs) != tune(
        a_behaviour_spec(), TRUCK, seed=2, **kwargs
    )


def test_gains_round_trip_through_a_dict():
    gains = ControllerGains.default()
    assert ControllerGains.from_dict(gains.to_dict()) == gains
