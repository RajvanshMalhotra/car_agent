"""Round 6 pieces: daily sensor noise, the CSV writer, the held-out-climate split, best checkpoints."""

import numpy as np
import pytest

from experiment.calendar_sim import OBSERVABLE_FEATURES
from experiment.data import load_daily_trajectory, write_daily_trajectory
from experiment.noise import DAILY_NOISE_CHANNEL, add_daily_noise
from experiment.windows import holdout_split

DAYS = 400


def _scenarios():
    rng = np.random.default_rng(0)
    out = {}
    for name in ("amb25_s0000", "amb42_s0001"):
        arr = {f: rng.normal(10.0, 1.0, DAYS) for f in OBSERVABLE_FEATURES}
        arr["i_mean_abs"] = np.abs(arr["i_mean_abs"]) * 0.01  # near zero, so the clamp matters
        arr["day"] = np.arange(DAYS, dtype=np.float64)
        arr["crystal"] = rng.random(DAYS)
        out[name] = arr
    return out, ["amb25_s0000", "amb42_s0001"]


def test_noise_touches_only_the_measurable_daily_channels():
    clean, order = _scenarios()
    noisy = add_daily_noise(clean, order, seed=3)
    for name in order:
        for column in clean[name]:
            changed = not np.array_equal(clean[name][column], noisy[name][column])
            assert changed == (column in DAILY_NOISE_CHANNEL), column


def test_latent_channels_are_never_noised():
    assert {"soc", "t_bay_mean", "t_bay_max"}.isdisjoint(DAILY_NOISE_CHANNEL)
    assert set(DAILY_NOISE_CHANNEL) == {
        "v_mean", "v_min", "v_max", "i_mean", "i_mean_abs", "t_bat_mean", "t_bat_max",
    }


def test_noise_size_follows_the_sensor_spec():
    clean, order = _scenarios()
    noisy = add_daily_noise(clean, order, seed=3)
    diff = lambda col: np.concatenate([noisy[n][col] - clean[n][col] for n in order])
    assert diff("v_mean").std() == pytest.approx(0.010, rel=0.15)
    assert diff("t_bat_max").std() == pytest.approx(0.5, rel=0.15)
    # i_mean around 10 A: 0.5% of reading is 0.05 A, so the 0.1 A floor applies.
    assert diff("i_mean").std() == pytest.approx(0.1, rel=0.15)


def test_absolute_current_stays_non_negative_after_noise():
    clean, order = _scenarios()
    noisy = add_daily_noise(clean, order, seed=3)
    assert all((noisy[n]["i_mean_abs"] >= 0.0).all() for n in order)


def test_noise_is_deterministic_and_does_not_mutate_its_input():
    clean, order = _scenarios()
    before = {n: {c: v.copy() for c, v in arr.items()} for n, arr in clean.items()}
    a = add_daily_noise(clean, order, seed=3)
    b = add_daily_noise(clean, order, seed=3)
    c = add_daily_noise(clean, order, seed=4)
    for n in order:
        assert np.array_equal(a[n]["v_mean"], b[n]["v_mean"])
        assert not np.array_equal(a[n]["v_mean"], c[n]["v_mean"])
        for col in clean[n]:
            assert np.array_equal(clean[n][col], before[n][col])


def test_write_then_load_round_trips(tmp_path):
    scenarios, order = _scenarios()
    path = tmp_path / "daily_trajectory.csv"
    write_daily_trajectory(path, scenarios, order)
    loaded, loaded_order = load_daily_trajectory(path)
    assert loaded_order == order
    for n in order:
        assert set(loaded[n]) == set(scenarios[n])
        for col in scenarios[n]:
            np.testing.assert_allclose(loaded[n][col], scenarios[n][col], rtol=0, atol=1e-12)


def _names():
    names = [f"amb{a}_s{i:04d}" for a in (-10, 10, 25, 42) for i in range(20)]
    ambient = {n: float(n[3:n.index("_")]) for n in names}
    return names, ambient


def test_holdout_split_puts_every_held_out_climate_battery_in_test_only():
    names, ambient = _names()
    train, val, test = holdout_split(names, ambient, holdout_ambient=42.0)
    assert test == {n for n in names if ambient[n] == 42.0}
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(names)
    assert {ambient[n] for n in train | val} == {-10.0, 10.0, 25.0}


def test_holdout_split_validation_fraction_and_determinism():
    names, ambient = _names()
    train, val, _ = holdout_split(names, ambient, holdout_ambient=42.0, val_fraction=0.15)
    assert len(val) == 9  # round(60 * 0.15)
    assert holdout_split(names, ambient, 42.0) == holdout_split(list(reversed(names)), ambient, 42.0)


def test_climate_comes_from_the_annual_mean_when_days_vary(tmp_path):
    """Round 12 varies ambient seasonally, so day 0 is no longer the climate."""
    import json as _json

    from experiment.windows import climate_of

    (tmp_path / "scenarios.json").write_text(_json.dumps([
        {"scenario": "amb42_s0000", "annual_mean_c": 42.0},
        {"scenario": "amb-10_s0001", "annual_mean_c": -10.0},
    ]))
    scenarios = {
        "amb42_s0000": {"ambient_c": np.array([30.2, 44.0])},   # starts its year cool
        "amb-10_s0001": {"ambient_c": np.array([-12.0, 1.0])},
    }
    order = ["amb42_s0000", "amb-10_s0001"]
    assert climate_of(tmp_path, scenarios, order) == {"amb42_s0000": 42.0, "amb-10_s0001": -10.0}


def test_climate_falls_back_to_the_column_for_older_datasets(tmp_path):
    """Rounds 4-11 have no annual_mean_c and must split exactly as before."""
    from experiment.windows import climate_of

    scenarios = {"a": {"ambient_c": np.array([25.0, 25.0])}}
    assert climate_of(tmp_path, scenarios, ["a"]) == {"a": 25.0}


def test_holdout_split_rejects_an_ambient_nobody_has():
    names, ambient = _names()
    with pytest.raises(ValueError):
        holdout_split(names, ambient, holdout_ambient=30.0)


def test_best_checkpoint_keeps_the_lowest_validation_epoch():
    torch = pytest.importorskip("torch")
    from experiment.checkpoints import BestCheckpoint

    model = torch.nn.Linear(2, 1)
    tracker = BestCheckpoint()
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert tracker.update(0, 0.5, model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert tracker.update(1, 0.2, model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    assert not tracker.update(2, 0.9, model)
    tracker.restore(model)
    assert tracker.best_epoch == 1 and tracker.best_loss == 0.2
    assert torch.all(model.weight == 2.0)
