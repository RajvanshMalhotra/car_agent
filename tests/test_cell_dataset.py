import csv
import json

import pytest

import dataclasses

from cell.aging import AgingRates, Damage
from cell.dataset import FEATURES, TARGETS, write_dataset
from cell.integrate import CellState, run_trip
from cell.life import TripSchedule, project
from load.spec import BatteryScenario

RATES = AgingRates()
SCENARIO = BatteryScenario(name="fixture", ambient_c=30.0)


def _drive(seconds, t0=0.0):
    for t in range(seconds):
        yield {
            "t_s": t0 + t, "speed_mps": 18.0, "rpm": 2400.0, "coolant_c": 92.0,
            "engine_load": 0.3, "engine_running": 1.0, "throttle": 0.3,
            "grade_rad": 0.0, "a_long_mps2": 0.0, "mass_kg": 1510.62,
            "ax_mps2": 0.1, "ay_mps2": 0.0, "az_mps2": 9.81,
        }


def _fixture(tmp_path):
    trips = []
    state = CellState()
    for index in range(3):
        samples = list(_drive(120, t0=index * 1000.0))
        result = run_trip(iter(samples), SCENARIO, state, RATES, cranked=index > 0)
        trips.append((samples, result))
    life = project(
        lambda health: dataclasses.replace(
            Damage.zero(), duration_s=86400.0, corrosion_equivalent_h=78.7
        ),
        RATES, TripSchedule(),
    )
    return write_dataset(
        out_dir=tmp_path, scenario=SCENARIO, trips=trips, life=life,
        provenance={"speed_mps": "measured", "mass_kg": "assumed"},
        dropouts=(), rates=RATES,
    )


def test_it_writes_two_tables_and_a_sidecar(tmp_path):
    _fixture(tmp_path)
    assert (tmp_path / "within_trip.csv").exists()
    assert (tmp_path / "trip_damage.csv").exists()
    assert (tmp_path / "dataset.json").exists()


def test_the_within_trip_table_has_one_row_per_step(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / "within_trip.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 120


def test_the_trip_damage_table_has_one_row_per_trip(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / "trip_damage.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3


def test_remaining_life_appears_only_on_the_trip_table(tmp_path):
    # Broadcasting it across 1 Hz rows is recoverable from the row index.
    _fixture(tmp_path)
    with (tmp_path / "within_trip.csv").open() as handle:
        within = csv.DictReader(handle).fieldnames
    with (tmp_path / "trip_damage.csv").open() as handle:
        per_trip = csv.DictReader(handle).fieldnames
    assert "rul_days" not in within
    assert "rul_days" in per_trip


def test_battery_current_is_a_target_not_a_feature(tmp_path):
    # soc is cumsum(i_bat_a * dt) / Q by construction, so offering both would
    # make the target analytically recoverable.
    assert "i_bat_a" in TARGETS
    assert "i_bat_a" not in FEATURES
    assert "soc" in TARGETS


def test_the_features_are_driving_channels(tmp_path):
    assert set(FEATURES) <= {
        "t_s", "speed_mps", "rpm", "throttle", "coolant_c", "engine_load",
        "grade_rad", "a_long_mps2", "engine_running", "t_bay_c",
    }


def test_the_sidecar_carries_the_leakage_report(tmp_path):
    sidecar = _fixture(tmp_path)
    assert "leakage" in sidecar
    assert sidecar["leakage"]["findings"] == []
    assert sidecar["leakage"]["features"] == list(FEATURES)


def test_the_sidecar_says_the_scale_is_unfitted(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["life"]["scale_unfitted"] is True


def test_the_sidecar_carries_provenance_and_dropouts(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["channels"]["mass_kg"] == "assumed"
    assert sidecar["dropouts"] == []


def test_the_sidecar_is_addressed_by_the_scenario_hash(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["scenario_hash"] == SCENARIO.scenario_hash
