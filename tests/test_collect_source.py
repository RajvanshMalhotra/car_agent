import math

from collect.schema import COLUMNS
from collect.source import FakeTrajectorySource


def a_source(**kwargs):
    kwargs.setdefault("interval_s", 0.01)
    # Tests step time by hand. A test that waits on the wall clock is a test
    # that eventually gets deleted.
    kwargs.setdefault("auto", False)
    return FakeTrajectorySource(**kwargs)


def test_a_drain_before_start_is_empty():
    assert a_source().drain() == []


def test_every_sample_has_every_column():
    source = a_source()
    source.start()
    source.advance(0.1)
    for sample in source.drain():
        assert set(sample) == set(COLUMNS)


def test_it_produces_one_sample_per_interval():
    source = a_source()
    source.start()
    source.advance(1.0)
    assert len(source.drain()) == 100


def test_draining_empties_the_buffer():
    source = a_source()
    source.start()
    source.advance(0.1)
    assert source.drain()
    assert source.drain() == []


def test_the_sequence_number_never_repeats_across_drains():
    source = a_source()
    source.start()
    source.advance(0.1)
    first = [s["seq"] for s in source.drain()]
    source.advance(0.1)
    second = [s["seq"] for s in source.drain()]
    assert first + second == list(range(1, 21))


def test_time_advances_monotonically():
    source = a_source()
    source.start()
    source.advance(0.5)
    times = [s["t_s"] for s in source.drain()]
    assert times == sorted(times)


def test_the_car_moves_at_the_speed_it_was_given():
    source = a_source(interval_s=0.1, speed_mps=20.0)
    source.start()
    source.advance(1.0)
    samples = source.drain()
    assert math.isclose(samples[-1]["x_m"] - samples[0]["x_m"], 18.0, rel_tol=1e-9)


def test_a_grade_shows_up_as_grade_and_not_as_acceleration():
    # The fake must strip gravity too, or it would let through a bug the real
    # source would hit.
    source = a_source(interval_s=0.1, grade_rad=math.radians(10))
    source.start()
    source.advance(0.5)
    sample = source.drain()[0]
    assert math.isclose(sample["grade_rad"], math.radians(10), abs_tol=1e-9)
    assert math.isclose(sample["a_long_mps2"], 0.0, abs_tol=1e-9)


def test_stop_is_safe_to_call_twice():
    source = a_source()
    source.start()
    source.stop()
    source.stop()
    assert source.running is False


def test_an_auto_source_catches_up_to_its_clock_on_drain():
    ticks = iter([0.0, 1.0])
    source = FakeTrajectorySource(interval_s=0.01, auto=True,
                                  clock=lambda: next(ticks))
    source.start()
    assert len(source.drain()) == 100
