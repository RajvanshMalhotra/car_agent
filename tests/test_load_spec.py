import json

import pytest

from load.spec import BatteryScenario


def test_defaults_are_a_mid_size_petrol_car():
    scenario = BatteryScenario(name="baseline")
    assert scenario.mass_kg == pytest.approx(1510.62)
    assert scenario.capacity_ah == pytest.approx(60.0)
    assert scenario.ambient_c == pytest.approx(25.0)


def test_out_of_range_values_are_refused():
    with pytest.raises(ValueError, match="capacity_ah"):
        BatteryScenario(name="x", capacity_ah=0.0)
    with pytest.raises(ValueError, match="ambient_c"):
        BatteryScenario(name="x", ambient_c=200.0)
    with pytest.raises(ValueError, match="hvac"):
        BatteryScenario(name="x", hvac=1.5)


def test_blank_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        BatteryScenario(name="   ")


def test_hash_ignores_the_name_but_not_the_physics():
    a = BatteryScenario(name="one")
    b = BatteryScenario(name="two")
    c = BatteryScenario(name="one", capacity_ah=70.0)
    assert a.scenario_hash == b.scenario_hash
    assert a.scenario_hash != c.scenario_hash


def test_round_trips_through_a_dict():
    scenario = BatteryScenario(name="hot", ambient_c=42.0, lights=True)
    assert BatteryScenario.from_dict(scenario.to_dict()) == scenario


def test_from_sidecar_takes_ambient_and_accessories_from_the_trajectory(tmp_path):
    # The trajectory already recorded what the drive assumed. Restating it here
    # would let the two drift apart silently.
    sidecar = tmp_path / "run.json"
    sidecar.write_text(json.dumps({
        "scenario": {"ambient_temp_c": 38.0, "accessory_load_a": 47.0},
    }))
    scenario = BatteryScenario.from_sidecar(sidecar, name="from-run")
    assert scenario.ambient_c == pytest.approx(38.0)
    assert scenario.accessory_base_a == pytest.approx(47.0)
    assert scenario.name == "from-run"


def test_from_sidecar_falls_back_to_defaults_when_fields_are_missing(tmp_path):
    sidecar = tmp_path / "run.json"
    sidecar.write_text(json.dumps({"scenario": {}}))
    scenario = BatteryScenario.from_sidecar(sidecar, name="bare")
    assert scenario.ambient_c == pytest.approx(25.0)
