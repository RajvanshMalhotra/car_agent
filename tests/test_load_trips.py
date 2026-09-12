import pytest

from load.trips import Segment, crank_times, segment


def _run(spans):
    """spans: list of (running, seconds). One sample per second."""
    t = 0.0
    for running, seconds in spans:
        for _ in range(seconds):
            yield {"t_s": t, "engine_running": 1.0 if running else 0.0}
            t += 1.0


def test_a_drive_then_a_park_is_one_trip_and_one_soak():
    segments = segment(_run([(True, 60), (False, 60)]))
    assert [s.kind for s in segments] == ["trip", "soak"]
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(59.0)


def test_the_first_trip_is_not_a_crank_when_the_engine_is_already_running():
    # The recording starts mid-drive. Counting that as a crank would invent a
    # 350 A event that never happened.
    segments = segment(_run([(True, 60), (False, 60)]))
    assert segments[0].cranked is False
    assert crank_times(segments) == ()


def test_a_trip_that_follows_a_soak_is_a_crank():
    segments = segment(_run([(True, 30), (False, 30), (True, 30)]))
    assert [s.kind for s in segments] == ["trip", "soak", "trip"]
    assert segments[2].cranked is True
    assert crank_times(segments) == (60.0,)


def test_a_brief_stall_does_not_split_a_trip():
    # A momentary rpm dip is not a trip boundary, and treating it as one would
    # manufacture cranks out of noise.
    segments = segment(_run([(True, 30), (False, 2), (True, 30)]))
    assert [s.kind for s in segments] == ["trip"]
    assert crank_times(segments) == ()


def test_a_recording_that_starts_parked_gives_a_soak_first():
    segments = segment(_run([(False, 30), (True, 30)]))
    assert [s.kind for s in segments] == ["soak", "trip"]
    assert segments[1].cranked is True


def test_an_empty_trajectory_has_no_segments():
    assert segment(iter(())) == []


def test_segments_cover_the_whole_trajectory_without_gaps():
    segments = segment(_run([(True, 30), (False, 30), (True, 30)]))
    for earlier, later in zip(segments, segments[1:]):
        assert later.start_s > earlier.end_s
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[-1].end_s == pytest.approx(89.0)


def test_a_leading_dip_shorter_than_min_soak_does_not_split_or_crank():
    # The absorb-backward check requires a preceding trip, and at position 0
    # there is none. A leading sub-MIN_SOAK_S dip used to survive as a
    # standalone soak and wrongly mark the following trip as cranked --
    # inventing a 350 A crank event out of a sub-threshold glitch.
    segments = segment(_run([(False, 2), (True, 30)]))
    assert [s.kind for s in segments] == ["trip"]
    assert segments[0].cranked is False
    assert crank_times(segments) == ()


def test_a_leading_dip_at_least_min_soak_is_still_a_genuine_soak():
    # The fix for the position-0 bug must not swing the other way and start
    # absorbing real leading parks just because they come first.
    segments = segment(_run([(False, 10), (True, 20)]))
    assert [s.kind for s in segments] == ["soak", "trip"]
    assert segments[1].cranked is True
    assert crank_times(segments) == (10.0,)


def test_a_trajectory_of_only_a_short_dip_does_not_crash():
    # No following trip to absorb into. Must not raise (a naive forward-only
    # fix can index past the end of the run list here) and must not
    # fabricate a crank out of nothing.
    segments = segment(_run([(False, 2)]))
    assert [s.kind for s in segments] == ["soak"]
    assert crank_times(segments) == ()
