import csv
import json

from collect.scenario import ScenarioSpec
from collect.schema import COLUMNS
from collect.writer import TrajectoryLog


def header(path):
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def a_spec():
    return ScenarioSpec(name="Test", minutes=1.0)


def some_samples(count=3):
    return [
        {**{c: 0.0 for c in COLUMNS}, "t_s": i * 0.01, "seq": i + 1,
         "speed_mps": 20.0, "coolant_c": 90.0}
        for i in range(count)
    ]


def test_the_header_is_the_schema_column_order(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples())
    assert header(path) == list(COLUMNS)


def test_every_sample_becomes_a_row(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples(5))
    assert len(rows(path)) == 5


def test_rows_are_on_disk_before_close(tmp_path):
    # A run is fifteen minutes on someone else's laptop. A crash at minute
    # fourteen must cost the last drain, not the run.
    log = TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01)
    log.write(some_samples(2))
    try:
        assert len(rows(tmp_path / "run.csv")) == 2
    finally:
        log.close()


def sidecar(tmp_path):
    return json.loads((tmp_path / "run.json").read_text())


def test_the_sidecar_carries_the_channel_manifest(tmp_path):
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
        log.write(some_samples())
    assert [c["name"] for c in sidecar(tmp_path)["channels"]] == list(COLUMNS)


def test_the_sidecar_says_which_columns_are_derived(tmp_path):
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
        log.write(some_samples())
    provenance = {c["name"]: c["provenance"] for c in sidecar(tmp_path)["channels"]}
    assert provenance["speed_mps"] == "measured"
    assert provenance["grade_rad"] == "derived"


def test_the_sidecar_carries_the_scenario_and_its_hash(tmp_path):
    spec = a_spec()
    with TrajectoryLog(tmp_path / "run.csv", spec, 0.01) as log:
        log.write(some_samples())
    written = sidecar(tmp_path)
    assert written["scenario_hash"] == spec.scenario_hash
    assert written["scenario"]["accessory_load_a"] == 35.0


def test_the_sidecar_records_what_the_game_was(tmp_path):
    beamng = {"level": "west_coast_usa", "vehicle": "etki"}
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01, beamng=beamng) as log:
        log.write(some_samples())
    assert sidecar(tmp_path)["beamng"]["level"] == "west_coast_usa"


def test_the_summary_counts_rows_and_duration(tmp_path):
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
        log.write(some_samples(100))
        assert log.summary()["rows"] == 100
        assert log.summary()["duration_s"] == 1.0


def test_the_sidecar_is_written_even_when_the_run_raises(tmp_path):
    try:
        with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
            log.write(some_samples())
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert (tmp_path / "run.json").exists()


def test_a_sample_missing_a_channel_is_refused_not_silently_blanked(tmp_path):
    broken = some_samples(1)
    del broken[0]["speed_mps"]
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
        try:
            log.write(broken)
        except KeyError as error:
            assert "speed_mps" in str(error)
        else:
            raise AssertionError("a missing channel must not pass silently")


def test_no_battery_column_is_written(tmp_path):
    with TrajectoryLog(tmp_path / "run.csv", a_spec(), 0.01) as log:
        log.write(some_samples())
    written = header(tmp_path / "run.csv")
    assert "current_a" not in written and "voltage_v" not in written
