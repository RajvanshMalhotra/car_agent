import pytest

from load.dropout import OUTGAUGE_CHANNELS, Dropout, DropoutFilter


def _sample(t, coolant=90.0, oil=95.0, fuel=40.0, speed=10.0):
    return {"t_s": t, "coolant_c": coolant, "oil_c": oil,
            "fuel_volume_l": fuel, "speed_mps": speed}


def test_live_samples_pass_through_untouched():
    samples = [_sample(0.0), _sample(1.0), _sample(2.0)]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [0.0, 1.0, 2.0]


def test_a_block_where_every_outgauge_channel_is_zero_is_excluded():
    samples = [
        _sample(0.0), _sample(1.0),
        _sample(2.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(3.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [0.0, 1.0]


def test_the_excluded_span_is_reported_for_the_sidecar():
    samples = [
        _sample(0.0),
        _sample(1.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(2.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    detector = DropoutFilter()
    list(detector(iter(samples)))
    assert detector.spans == (Dropout(start_s=1.0, end_s=2.0, rows=2),)
    assert detector.rows_excluded == 2


def test_one_zero_channel_alone_is_not_a_dropout():
    # An empty tank is a real measurement. Only the whole packet going dark is
    # a dropout.
    samples = [_sample(0.0), _sample(1.0, fuel=0.0)]
    kept = list(DropoutFilter()(iter(samples)))
    assert len(kept) == 2


def test_leading_zeros_before_any_live_sample_are_excluded_too():
    # Coolant is a real temperature even with the engine cold, so a zero here
    # still means no packet arrived.
    samples = [
        _sample(0.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(1.0),
    ]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [1.0]


def test_two_separate_dropouts_are_reported_separately():
    samples = [
        _sample(0.0),
        _sample(1.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(2.0),
        _sample(3.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    detector = DropoutFilter()
    list(detector(iter(samples)))
    assert len(detector.spans) == 2
    assert detector.spans[0].start_s == pytest.approx(1.0)
    assert detector.spans[1].start_s == pytest.approx(3.0)


def test_the_watched_channels_are_the_outgauge_ones():
    assert set(OUTGAUGE_CHANNELS) == {"coolant_c", "oil_c", "fuel_volume_l"}
