import json

import pytest

from battery_run import build, main
from cell.aging import AgingRates
from cell.life import TripSchedule
from load.spec import BatteryScenario

HEADER = (
    "timestamp,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,"
    "up_x,up_y,up_z,roll,pitch,yaw,roll_vel,pitch_vel,yaw_vel,"
    "roll_acc,pitch_acc,yaw_acc,game_time,speed,rpm,turbo,engine_temp,fuel,"
    "oil_pressure,oil_temp,throttle,brake,clutch,gear"
)


def _row(t, speed=18.0, rpm=2400.0, engine_temp=92.0):
    return (
        f"{t},283.5,-713.4,148.5,-19.2,-0.1,-2.1,0.5,0.7,9.8,"
        f"-0.11,0.03,0.99,0.03,0.0,1.56,0.01,-0.02,-0.01,"
        f"0.16,-0.04,-0.09,0,{speed},{rpm},1.17,{engine_temp},0.99,"
        f"0.0,81.5,0.3,b'\\x00',0,4"
    )


@pytest.fixture
def trajectory(tmp_path):
    path = tmp_path / "telemetry.csv"
    rows = [_row(i * 0.1) for i in range(1200)]  # 120 s at 10 Hz
    rows += [_row(120.0 + i * 0.1, speed=0.0, rpm=0.0) for i in range(600)]
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_it_produces_a_dataset(trajectory, tmp_path):
    out = tmp_path / "dataset"
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=2, out_dir=out,
    )
    assert (out / "within_trip.csv").exists()
    assert sidecar["tables"]["within_trip"]["rows"] > 0


def test_repeating_the_trip_produces_cranks(trajectory, tmp_path):
    # One recorded trip has no crank in it. Repeating it under a schedule is
    # what gives the sulfation pathway anything to work with.
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=3, out_dir=tmp_path / "d",
    )
    assert sidecar["tables"]["trip_damage"]["rows"] == 3


def test_the_dataset_has_no_leakage_findings(trajectory, tmp_path):
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=2, out_dir=tmp_path / "d",
    )
    assert sidecar["leakage"]["findings"] == []


def test_physical_bounds_hold(trajectory, tmp_path):
    import csv
    out = tmp_path / "d"
    build(trajectory, BatteryScenario(name="t", ambient_c=25.0), TripSchedule(),
          AgingRates(), repeats=2, out_dir=out)
    with (out / "within_trip.csv").open() as handle:
        for row in csv.DictReader(handle):
            assert 0.0 <= float(row["soc"]) <= 1.0
            assert 9.0 < float(row["v_bat_v"]) < 15.0
            assert float(row["t_bay_c"]) >= 25.0 - 1e-6


def test_a_hot_ambient_ages_it_faster_than_a_cool_one(trajectory, tmp_path):
    cool = build(trajectory, BatteryScenario(name="c", ambient_c=25.0),
                 TripSchedule(), AgingRates(), 2, tmp_path / "cool")
    hot = build(trajectory, BatteryScenario(name="h", ambient_c=42.0),
                TripSchedule(), AgingRates(), 2, tmp_path / "hot")
    assert hot["life"]["eol_days"] < cool["life"]["eol_days"]


def test_the_cli_runs(trajectory, tmp_path, capsys):
    code = main([str(trajectory), "--name", "cli", "--out", str(tmp_path / "d"),
                 "--repeats", "2"])
    assert code == 0
    assert (tmp_path / "d" / "dataset.json").exists()


def test_the_cli_reports_that_the_scale_is_unfitted(trajectory, tmp_path, capsys):
    main([str(trajectory), "--name", "cli", "--out", str(tmp_path / "d")])
    assert "unfitted" in capsys.readouterr().out.lower()
