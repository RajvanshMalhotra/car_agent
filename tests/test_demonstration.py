"""Learning a route from a human drive.

You drive A to B once. The recording keeps where you went *and how fast you
went there*, and that speed profile becomes the baseline the behaviours modulate:
the same drive, done aggressively or economically.

This removes the part of the project most prone to being wrong. There is no
reward function to design, no randomised training, no restarts that fail three
times in five. The demonstration is the answer, and a behaviour scales it.
"""

import math

import pytest

from behaviour.spec import BehaviourSpec
from control.demonstration import Demonstration, load_demonstration, save_demonstration
from control.path import Path


def a_drive(n=200, spacing=2.0, speed=12.0):
    return [(i * spacing, 0.0, speed) for i in range(n)]


def a_spec(**overrides):
    fields = dict(
        name="test", target_speed_factor=1.0, accel_limit_mps2=2.0,
        decel_limit_mps2=3.0, jerk_limit_mps3=4.0, following_distance_s=1.8,
        corner_speed_factor=0.9, reaction_lag_s=0.0, erraticness=0.0,
        trip_duration_s=1200.0, idle_fraction=0.0, hvac_setting=0.5,
        ambient_temp_c=22.0, cold_start=True, start_stop_enabled=False,
    )
    fields.update(overrides)
    return BehaviourSpec(**fields)


# -- what a demonstration is ----------------------------------------------


def test_a_demonstration_keeps_the_route_that_was_driven():
    demo = Demonstration.from_samples(a_drive(n=100, spacing=2.0), name="lap")
    assert demo.to_path().length_m == pytest.approx(198.0, rel=0.02)


def test_a_demonstration_keeps_the_speed_that_was_driven():
    demo = Demonstration.from_samples(a_drive(speed=14.0), name="lap")
    assert demo.speed_at(100.0) == pytest.approx(14.0, rel=0.05)


def test_the_speed_profile_follows_the_drive():
    # Slow at the start, quick in the middle, slow again: a real trip shape.
    samples = [
        (i * 2.0, 0.0, 4.0 + 12.0 * math.sin(math.pi * i / 200)) for i in range(200)
    ]
    demo = Demonstration.from_samples(samples, name="trip")
    assert demo.speed_at(10.0) < demo.speed_at(200.0)
    assert demo.speed_at(380.0) < demo.speed_at(200.0)


def test_speed_between_samples_is_interpolated():
    samples = [(0.0, 0.0, 0.0), (10.0, 0.0, 10.0), (20.0, 0.0, 20.0)]
    demo = Demonstration.from_samples(samples, name="ramp")
    assert demo.speed_at(5.0) == pytest.approx(5.0, abs=1.0)


def test_speed_past_the_end_clamps_rather_than_extrapolating():
    demo = Demonstration.from_samples(a_drive(speed=9.0), name="lap")
    assert demo.speed_at(99_999.0) == pytest.approx(9.0, rel=0.05)


def test_a_stationary_stretch_is_kept_as_a_stop():
    # Standing at a junction is data, not noise: idling with a hot engine is
    # the mechanism this whole project measures.
    samples = (
        [(i * 2.0, 0.0, 10.0) for i in range(50)]
        + [(100.0, 0.0, 0.0)] * 300
        + [(100.0 + i * 2.0, 0.0, 10.0) for i in range(1, 50)]
    )
    demo = Demonstration.from_samples(samples, name="with a stop", sample_hz=10.0)
    stops = demo.stops()
    assert len(stops) == 1
    assert stops[0].arc_length_m == pytest.approx(100.0, abs=5.0)
    assert stops[0].duration_s == pytest.approx(30.0, rel=0.2)


def test_a_brief_hesitation_is_not_a_stop():
    samples = (
        [(i * 2.0, 0.0, 10.0) for i in range(50)]
        + [(100.0, 0.0, 0.0)] * 5
        + [(100.0 + i * 2.0, 0.0, 10.0) for i in range(1, 50)]
    )
    demo = Demonstration.from_samples(samples, name="hesitation", sample_hz=10.0)
    assert demo.stops() == []


def test_a_demonstration_that_never_moved_is_refused():
    with pytest.raises(ValueError, match="did not move"):
        Demonstration.from_samples([(0.0, 0.0, 0.0)] * 100, name="parked")


# -- replaying it in a style ----------------------------------------------


def test_replaying_it_plainly_reproduces_the_speed_driven():
    demo = Demonstration.from_samples(a_drive(speed=12.0), name="lap")
    assert demo.target_speed_at(100.0, a_spec(target_speed_factor=1.0)) == pytest.approx(
        12.0, rel=0.05
    )


def test_an_aggressive_behaviour_drives_it_faster():
    demo = Demonstration.from_samples(a_drive(speed=12.0), name="lap")
    brisk = demo.target_speed_at(100.0, a_spec(target_speed_factor=1.3))
    assert brisk > 12.0


def test_an_economical_behaviour_drives_it_slower():
    demo = Demonstration.from_samples(a_drive(speed=12.0), name="lap")
    gentle = demo.target_speed_at(100.0, a_spec(target_speed_factor=0.6))
    assert gentle < 12.0


def test_the_shape_of_the_drive_survives_the_styling():
    # Aggressive should not flatten a trip into one constant speed: where the
    # human slowed, the agent still slows.
    samples = [
        (i * 2.0, 0.0, 4.0 + 10.0 * math.sin(math.pi * i / 200)) for i in range(200)
    ]
    demo = Demonstration.from_samples(samples, name="trip")
    spec = a_spec(target_speed_factor=1.3)
    assert demo.target_speed_at(20.0, spec) < demo.target_speed_at(200.0, spec)


def test_a_stop_stays_a_stop_however_aggressive_the_style():
    samples = (
        [(i * 2.0, 0.0, 10.0) for i in range(50)]
        + [(100.0, 0.0, 0.0)] * 300
        + [(100.0 + i * 2.0, 0.0, 10.0) for i in range(1, 50)]
    )
    demo = Demonstration.from_samples(samples, name="with a stop", sample_hz=10.0)
    assert demo.target_speed_at(100.0, a_spec(target_speed_factor=1.3)) < 1.0


# -- on disk --------------------------------------------------------------


def test_a_demonstration_survives_being_saved_and_loaded(tmp_path):
    demo = Demonstration.from_samples(a_drive(), name="lap")
    save_demonstration(tmp_path / "lap.json", demo)
    loaded = load_demonstration(tmp_path / "lap.json")
    assert loaded.name == "lap"
    assert loaded.to_path().length_m == pytest.approx(demo.to_path().length_m)
    assert loaded.speed_at(50.0) == pytest.approx(demo.speed_at(50.0))


def test_the_saved_file_says_what_was_driven(tmp_path):
    import json

    demo = Demonstration.from_samples(a_drive(), name="west coast run")
    save_demonstration(tmp_path / "d.json", demo, metadata={"vehicle": "etk800"})
    stored = json.loads((tmp_path / "d.json").read_text())
    assert stored["name"] == "west coast run"
    assert stored["metadata"]["vehicle"] == "etk800"
    assert stored["mean_speed_mps"] > 0.0
