import statistics

from cell.sensors import LATENT, MEASURABLE, NOISE, SensorNoise


def _row():
    return {
        "speed_mps": 20.0,
        "rpm": 2500.0,
        "throttle": 0.4,
        "coolant_c": 90.0,
        "engine_load": 0.5,
        "grade_rad": 0.0,
        "a_long_mps2": 0.1,
        "engine_running": 1.0,
        "t_bay_c": 55.0,
        "i_bat_a": 10.0,
        "v_bat_v": 12.6,
        "t_bat_c": 30.0,
        "soc": 0.9,
        "r_int_ohm": 0.02,
        "soh": 0.95,
    }


def test_latent_channels_stay_byte_identical():
    noise = SensorNoise(seed=0)
    row = _row()
    out = noise.apply(row)
    for channel in ("soc", "r_int_ohm", "t_bay_c"):
        assert out[channel] == row[channel]


def test_measurable_channels_actually_change():
    noise = SensorNoise(seed=0)
    row = _row()
    out = noise.apply(row)
    for channel in MEASURABLE:
        assert out[channel] != row[channel], f"{channel} did not change"


def test_disabled_is_exact_passthrough():
    noise = SensorNoise(seed=0, enabled=False)
    row = _row()
    out = noise.apply(row)
    assert out == row
    # And every value is the same object identity's equal float, not just close.
    for key in row:
        assert out[key] == row[key]


def test_same_seed_is_identical_different_seed_is_not():
    row = _row()
    a = SensorNoise(seed=1).apply(row)
    b = SensorNoise(seed=1).apply(row)
    c = SensorNoise(seed=2).apply(row)
    assert a == b
    assert a != c


def test_scale_zero_matches_disabled():
    row = _row()
    zero_scale = SensorNoise(seed=3, scale=0.0).apply(row)
    disabled = SensorNoise(seed=3, enabled=False).apply(row)
    assert zero_scale == disabled == row


def test_scale_two_roughly_doubles_spread():
    channel = "v_bat_v"
    n = 4000
    row = _row()
    base = [SensorNoise(seed=i, scale=1.0).apply(row)[channel] for i in range(n)]
    doubled = [SensorNoise(seed=i, scale=2.0).apply(row)[channel] for i in range(n)]
    base_sd = statistics.pstdev(base)
    doubled_sd = statistics.pstdev(doubled)
    ratio = doubled_sd / base_sd
    assert 1.8 < ratio < 2.2


def test_throttle_and_rpm_stay_in_bounds_even_when_pushed_out():
    noise = SensorNoise(seed=0)
    edge_low = {**_row(), "throttle": 0.0, "rpm": 0.0, "speed_mps": 0.0}
    edge_high = {**_row(), "throttle": 1.0}
    for seed in range(200):
        out = SensorNoise(seed=seed).apply(edge_low)
        assert 0.0 <= out["throttle"] <= 1.0
        assert out["rpm"] >= 0.0
        assert out["speed_mps"] >= 0.0
        out_high = SensorNoise(seed=seed).apply(edge_high)
        assert 0.0 <= out_high["throttle"] <= 1.0


def test_empirical_sigma_matches_declared_sigma():
    row = _row()
    n = 6000
    for channel in MEASURABLE:
        spec = NOISE[channel]
        sigma = spec["sigma"](row[channel]) if callable(spec["sigma"]) else spec["sigma"]
        if sigma == 0.0:
            continue
        samples = [
            SensorNoise(seed=i).apply(row)[channel] - row[channel] for i in range(n)
        ]
        empirical = statistics.pstdev(samples)
        # Loose tolerance: clamping (throttle, rpm, speed) truncates the tail
        # and pulls the empirical sd below sigma, so allow generous slack.
        assert abs(empirical - sigma) < 0.25 * sigma + 0.02


def test_apply_does_not_mutate_input():
    row = _row()
    original = dict(row)
    SensorNoise(seed=0).apply(row)
    assert row == original


def test_manifest_reports_what_was_applied():
    noise = SensorNoise(seed=7, enabled=True, scale=1.5)
    manifest = noise.manifest()
    assert manifest["enabled"] is True
    assert manifest["seed"] == 7
    assert manifest["scale"] == 1.5
    assert "sigmas" in manifest
    for channel in MEASURABLE:
        assert channel in manifest["sigmas"]
    for channel in LATENT:
        assert channel not in manifest["sigmas"]


def test_measurable_and_latent_partition_declared_channels():
    assert set(MEASURABLE) & set(LATENT) == set()
    assert "soc" in LATENT
    assert "t_bay_c" in LATENT
    assert "r_int_ohm" in LATENT
    assert "v_bat_v" in MEASURABLE
    assert "i_bat_a" in MEASURABLE
