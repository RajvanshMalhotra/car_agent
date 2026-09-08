"""Run logging.

A log has to be interpretable on its own, months later, without the script that
produced it. That means every row carries the channels the battery layer
consumes, and every run carries the provenance needed to reproduce it:
(spec_hash, scenario, seed).
"""

import csv
import json

import pytest

from behaviour.spec import BehaviourSpec
from datalog.writer import RunLog
from sim.backend import ControlInput, VehicleState


def a_spec(**overrides):
    fields = dict(
        name="courier",
        target_speed_factor=1.0,
        accel_limit_mps2=2.0,
        decel_limit_mps2=3.0,
        jerk_limit_mps3=4.0,
        following_distance_s=1.8,
        corner_speed_factor=0.9,
        reaction_lag_s=0.4,
        erraticness=0.1,
        trip_duration_s=1200.0,
        idle_fraction=0.15,
        hvac_setting=1.0,
        ambient_temp_c=42.0,
        cold_start=True,
        start_stop_enabled=False,
    )
    fields.update(overrides)
    return BehaviourSpec(**fields)


def a_state(**overrides):
    fields = dict(
        sim_time_s=1.0,
        x_m=10.0,
        y_m=-2.0,
        heading_rad=0.25,
        speed_mps=15.0,
        rpm=2400.0,
        coolant_temp_c=88.0,
        underbonnet_temp_c=61.0,
        oil_temp_c=95.0,
        gear=4,
        fuel_fraction=0.6,
        throttle=0.42,
        brake=0.0,
        engine_on=True,
        crank_count=1,
    )
    fields.update(overrides)
    return VehicleState(**fields)


def open_log(tmp_path, spec=None, **kwargs):
    return RunLog(
        tmp_path / "run.csv",
        spec=spec or a_spec(),
        scenario="test-route",
        seed=7,
        log_hz=1.0,
        **kwargs,
    )


def rows(tmp_path):
    with open(tmp_path / "run.csv") as handle:
        return list(csv.DictReader(handle))


def sidecar(tmp_path):
    return json.loads((tmp_path / "run.json").read_text())


# -- provenance -----------------------------------------------------------


def test_the_sidecar_records_what_is_needed_to_reproduce_the_run(tmp_path):
    spec = a_spec()
    with open_log(tmp_path, spec):
        pass
    meta = sidecar(tmp_path)
    assert meta["spec_hash"] == spec.spec_hash
    assert meta["scenario"] == "test-route"
    assert meta["seed"] == 7


def test_the_sidecar_stores_the_whole_behaviour_spec(tmp_path):
    with open_log(tmp_path, a_spec(idle_fraction=0.45)):
        pass
    assert sidecar(tmp_path)["spec"]["idle_fraction"] == 0.45


def test_the_sidecar_records_the_ambient_temperature(tmp_path):
    # Without it a temperature trace cannot be interpreted at all.
    with open_log(tmp_path, a_spec(ambient_temp_c=42.0)):
        pass
    assert sidecar(tmp_path)["ambient_temp_c"] == 42.0


def test_the_sidecar_records_the_accessory_state(tmp_path):
    # HVAC is the electrical load that drives the recharge-deficit pathway.
    with open_log(tmp_path, a_spec(hvac_setting=1.0)):
        pass
    assert sidecar(tmp_path)["hvac_setting"] == 1.0


def test_the_sidecar_records_the_log_rate(tmp_path):
    with open_log(tmp_path):
        pass
    assert sidecar(tmp_path)["log_hz"] == 1.0


def test_the_sidecar_names_its_csv(tmp_path):
    with open_log(tmp_path):
        pass
    assert sidecar(tmp_path)["csv"] == "run.csv"


# -- rows -----------------------------------------------------------------


def test_a_recorded_row_carries_the_engine_channels(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(), ControlInput(0.5, 0.0, 0.1))
    row = rows(tmp_path)[0]
    assert float(row["rpm"]) == 2400.0
    assert float(row["coolant_c"]) == 88.0
    assert float(row["underbonnet_c"]) == 61.0
    assert float(row["oil_c"]) == 95.0


def test_a_recorded_row_carries_the_pose(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(x_m=12.5, y_m=-3.5), ControlInput(0.0, 0.0, 0.0))
    row = rows(tmp_path)[0]
    assert float(row["x_m"]) == 12.5
    assert float(row["y_m"]) == -3.5


def test_actual_and_commanded_throttle_are_both_recorded(tmp_path):
    # They differ: the commanded value is what the controller asked for, the
    # actual is what the engine did. Engine load follows the actual one.
    with open_log(tmp_path) as log:
        log.record(a_state(throttle=0.42), ControlInput(0.9, 0.0, 0.0))
    row = rows(tmp_path)[0]
    assert float(row["throttle"]) == 0.42
    assert float(row["throttle_cmd"]) == 0.9


def test_actual_and_commanded_brake_are_both_recorded(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(brake=0.1), ControlInput(0.0, 0.7, 0.0))
    row = rows(tmp_path)[0]
    assert float(row["brake"]) == 0.1
    assert float(row["brake_cmd"]) == 0.7


def test_the_gear_and_fuel_level_are_recorded(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(gear=3, fuel_fraction=0.55), ControlInput(0, 0, 0))
    row = rows(tmp_path)[0]
    assert int(row["gear"]) == 3
    assert float(row["fuel"]) == 0.55


def test_crank_events_are_recorded(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(crank_count=3), ControlInput(0, 0, 0))
    assert int(rows(tmp_path)[0]["crank_count"]) == 3


# -- derived channels -----------------------------------------------------


def test_a_stationary_running_engine_is_marked_idling(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(speed_mps=0.1, engine_on=True), ControlInput(0, 0, 0))
    assert int(rows(tmp_path)[0]["idling"]) == 1


def test_a_moving_vehicle_is_not_idling(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(speed_mps=12.0), ControlInput(0, 0, 0))
    assert int(rows(tmp_path)[0]["idling"]) == 0


def test_a_stopped_engine_is_not_idling(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(speed_mps=0.0, engine_on=False), ControlInput(0, 0, 0))
    assert int(rows(tmp_path)[0]["idling"]) == 0


def test_rows_are_numbered_by_trip_segment(tmp_path):
    # Trip segmentation is what the sulfation pathway is defined over.
    with open_log(tmp_path) as log:
        log.record(a_state(), ControlInput(0, 0, 0))
        log.end_trip()
        log.record(a_state(), ControlInput(0, 0, 0))
    assert [int(r["trip"]) for r in rows(tmp_path)] == [0, 1]


# -- summary --------------------------------------------------------------


def test_the_summary_reports_the_idle_fraction_actually_achieved(tmp_path):
    with open_log(tmp_path) as log:
        for _ in range(3):
            log.record(a_state(speed_mps=0.0), ControlInput(0, 0, 0))
        for _ in range(1):
            log.record(a_state(speed_mps=20.0), ControlInput(0, 0, 0))
    assert sidecar(tmp_path)["summary"]["idle_fraction"] == pytest.approx(0.75)


def test_the_summary_reports_corrosion_in_equivalent_hours(tmp_path):
    with open_log(tmp_path) as log:
        for _ in range(3600):
            log.record(a_state(underbonnet_temp_c=25.0), ControlInput(0, 0, 0))
    assert sidecar(tmp_path)["summary"]["equivalent_hours"] == pytest.approx(1.0)


def test_the_summary_counts_the_rows(tmp_path):
    with open_log(tmp_path) as log:
        for _ in range(5):
            log.record(a_state(), ControlInput(0, 0, 0))
    assert sidecar(tmp_path)["summary"]["rows"] == 5


def test_a_run_with_no_rows_still_writes_a_readable_sidecar(tmp_path):
    with open_log(tmp_path):
        pass
    assert sidecar(tmp_path)["summary"]["rows"] == 0


# -- durability -----------------------------------------------------------


def test_rows_are_on_disk_before_the_run_ends(tmp_path):
    # A run that crashes at minute 40 must not lose 40 minutes of data.
    with open_log(tmp_path) as log:
        log.record(a_state(), ControlInput(0, 0, 0))
        assert len(rows(tmp_path)) == 1
        log.record(a_state(), ControlInput(0, 0, 0))
        assert len(rows(tmp_path)) == 2


def test_the_sidecar_is_written_even_if_the_run_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with open_log(tmp_path) as log:
            log.record(a_state(), ControlInput(0, 0, 0))
            raise RuntimeError("simulated crash")
    assert sidecar(tmp_path)["summary"]["rows"] == 1


def test_damage_is_recorded(tmp_path):
    # Without it a crashed run cannot be told from a run that clipped a kerb.
    with open_log(tmp_path) as log:
        log.record(a_state(damage=420.0), ControlInput(0, 0, 0))
    assert float(rows(tmp_path)[0]["damage"]) == 420.0


def test_the_summary_reports_the_damage_taken(tmp_path):
    with open_log(tmp_path) as log:
        log.record(a_state(damage=0.0), ControlInput(0, 0, 0))
        log.record(a_state(damage=750.0), ControlInput(0, 0, 0))
    assert sidecar(tmp_path)["summary"]["max_damage"] == 750.0


# -- the log surviving its own failures -----------------------------------


def test_the_sidecar_is_written_even_if_its_directory_vanished(tmp_path):
    # The whole point of this class is that a run survives a crash. It must not
    # be the thing that crashes.
    import shutil

    log = open_log(tmp_path)
    log.record(a_state(), ControlInput(0, 0, 0))
    shutil.rmtree(tmp_path)
    log.close()
    assert sidecar(tmp_path)["summary"]["rows"] == 1


def test_a_sidecar_that_cannot_be_written_does_not_raise(tmp_path, monkeypatch):
    log = open_log(tmp_path)
    log.record(a_state(), ControlInput(0, 0, 0))

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(log.sidecar_path), "write_text", refuse)
    log.close()  # the CSV is the data; losing the summary must not lose the run


def test_closing_twice_is_safe(tmp_path):
    log = open_log(tmp_path)
    log.record(a_state(), ControlInput(0, 0, 0))
    log.close()
    log.close()
    assert sidecar(tmp_path)["summary"]["rows"] == 1


def test_a_row_can_be_recorded_with_no_control_of_our_own(tmp_path):
    # When BeamNG's AI drives there are no commanded values from us. The actual
    # pedals still arrive from the game, and those are what engine load follows.
    with open_log(tmp_path) as log:
        log.record(a_state(throttle=0.7, brake=0.1), None)
    row = rows(tmp_path)[0]
    assert float(row["throttle"]) == 0.7
    assert float(row["brake"]) == 0.1
    assert row["throttle_cmd"] == ""
    assert row["steering_cmd"] == ""


def test_a_run_driven_by_the_game_still_summarises(tmp_path):
    with open_log(tmp_path) as log:
        for _ in range(10):
            log.record(a_state(speed_mps=12.0), None)
    assert sidecar(tmp_path)["summary"]["rows"] == 10
