import math

import pytest

from collect.schema import COLUMNS
from load.legacy import (
    LEGACY_PROVENANCE,
    PROVENANCES,
    AbsentChannel,
    Sample,
    assumed_inputs,
    manifest,
    read_legacy,
)

HEADER = (
    "timestamp,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,"
    "up_x,up_y,up_z,roll,pitch,yaw,roll_vel,pitch_vel,yaw_vel,"
    "roll_acc,pitch_acc,yaw_acc,game_time,speed,rpm,turbo,engine_temp,fuel,"
    "oil_pressure,oil_temp,throttle,brake,clutch,gear"
)


def _row(t, speed=10.0, rpm=2000.0, pitch=0.0, engine_temp=90.0, throttle=0.3):
    return (
        f"{t},283.5,-713.4,148.5,-19.2,-0.1,-2.1,0.5,0.7,-0.1,"
        f"-0.11,0.03,0.99,0.03,{pitch},1.56,0.01,-0.02,-0.01,"
        f"0.16,-0.04,-0.09,0,{speed},{rpm},1.17,{engine_temp},0.99,"
        f"0.0,81.5,{throttle},b'\\x00\\x00',0,4"
    )


def _write(tmp_path, rows):
    path = tmp_path / "telemetry.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_every_canonical_column_has_a_provenance():
    assert set(LEGACY_PROVENANCE) == set(COLUMNS)
    assert set(LEGACY_PROVENANCE.values()) <= set(PROVENANCES)


def test_the_columns_this_file_cannot_supply_are_marked_absent():
    # brake is a blob, clutch and oil_pressure are constant zero, and the rest
    # were never in the OutGauge shape at all.
    for column in (
        "brake", "clutch_ratio", "engine_torque_nm", "ignition_level",
        "wheel_av_fl", "brake_temp_fl", "downforce_fl",
    ):
        assert LEGACY_PROVENANCE[column] == "absent", column


def test_mass_and_engine_load_are_assumed_not_measured():
    assert LEGACY_PROVENANCE["mass_kg"] == "assumed"
    assert LEGACY_PROVENANCE["engine_load"] == "assumed"


def test_speed_and_coolant_are_measured():
    assert LEGACY_PROVENANCE["speed_mps"] == "measured"
    assert LEGACY_PROVENANCE["coolant_c"] == "measured"


def test_reading_an_absent_column_raises_rather_than_returning_zero():
    sample = Sample({"brake": 0.0, "speed_mps": 10.0}, LEGACY_PROVENANCE)
    assert sample["speed_mps"] == pytest.approx(10.0)
    with pytest.raises(AbsentChannel, match="brake"):
        sample["brake"]


def test_samples_carry_every_canonical_column(tmp_path):
    path = _write(tmp_path, [_row(100.0), _row(100.1)])
    trajectory = read_legacy(path)
    first = next(trajectory.samples())
    assert set(first.keys()) == set(COLUMNS)


def test_time_is_rebased_to_zero_at_the_first_row(tmp_path):
    path = _write(tmp_path, [_row(1789128527.0), _row(1789128528.0)])
    times = [s["t_s"] for s in read_legacy(path).samples()]
    assert times[0] == pytest.approx(0.0)
    assert times[1] == pytest.approx(1.0)


def test_grade_comes_from_pitch(tmp_path):
    path = _write(tmp_path, [_row(0.0, pitch=0.1)])
    sample = next(read_legacy(path).samples())
    # dir_z is sin(pitch); grade_rad is its arcsine, so it returns pitch.
    assert sample["grade_rad"] == pytest.approx(0.1, abs=1e-6)


def test_engine_running_is_derived_from_rpm(tmp_path):
    path = _write(tmp_path, [_row(0.0, rpm=2000.0), _row(0.1, rpm=3.0)])
    running = [s["engine_running"] for s in read_legacy(path).samples()]
    assert running == [1.0, 0.0]


def test_at_hz_downsamples_without_loading_everything(tmp_path):
    rows = [_row(i * 0.01) for i in range(500)]  # 5 s at 100 Hz
    path = _write(tmp_path, rows)
    sampled = list(read_legacy(path).at_hz(1.0))
    assert len(sampled) == 5
    assert [round(s["t_s"]) for s in sampled] == [0, 1, 2, 3, 4]


def test_assumed_inputs_names_only_the_assumed_ones():
    assert assumed_inputs(
        LEGACY_PROVENANCE, ("speed_mps", "mass_kg", "coolant_c")
    ) == ("mass_kg",)


def test_manifest_records_why_each_column_is_what_it_is():
    rows = {entry["name"]: entry for entry in manifest()}
    assert rows["brake"]["provenance"] == "absent"
    assert "display1" in rows["brake"]["source"]
    assert rows["mass_kg"]["provenance"] == "assumed"
