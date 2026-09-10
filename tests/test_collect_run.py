import csv
import json

from collect.run import collect_run
from collect.scenario import ScenarioSpec
from collect.source import FakeTrajectorySource


def a_spec():
    return ScenarioSpec(name="Test", minutes=1.0)


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class Clock:
    """A clock the test drives, so no test ever waits on real time."""

    def __init__(self, source, step=1.0):
        self.now = 0.0
        self.source = source
        self.step = step

    def time(self):
        return self.now

    def sleep(self, _seconds):
        self.now += self.step
        self.source.advance(self.step)


def a_run(tmp_path, source, seconds):
    clock = Clock(source)
    return collect_run(source, a_spec(), tmp_path / "run.csv", seconds=seconds,
                       clock=clock.time, sleep=clock.sleep)


def a_source(cls=FakeTrajectorySource, **kwargs):
    kwargs.setdefault("interval_s", 0.01)
    kwargs.setdefault("auto", False)
    return cls(**kwargs)


def test_it_writes_a_csv_with_a_row_per_sample(tmp_path):
    a_run(tmp_path, a_source(), seconds=3.0)
    assert len(rows(tmp_path / "run.csv")) == 300


def test_it_writes_the_sidecar(tmp_path):
    a_run(tmp_path, a_source(), seconds=2.0)
    assert json.loads((tmp_path / "run.json").read_text())["scenario_hash"]


def test_it_stops_the_source_when_the_time_is_up(tmp_path):
    source = a_source()
    a_run(tmp_path, source, seconds=2.0)
    assert source.running is False


def test_it_stops_the_source_even_when_the_run_raises(tmp_path):
    class Exploding(FakeTrajectorySource):
        def drain(self):
            raise RuntimeError("the VM went away")

    source = a_source(Exploding)
    try:
        a_run(tmp_path, source, seconds=2.0)
    except RuntimeError:
        pass
    assert source.running is False


def test_a_final_drain_happens_after_the_clock_runs_out(tmp_path):
    # Whatever is buffered when time expires is still data.
    assert a_run(tmp_path, a_source(), seconds=1.0)["rows"] == 100


def test_the_summary_reports_rows_and_duration(tmp_path):
    summary = a_run(tmp_path, a_source(), seconds=2.0)
    assert summary["rows"] == 200
    assert summary["duration_s"] == 2.0


def test_dropped_samples_reach_the_summary(tmp_path):
    source = a_source()
    source.dropped = 7
    assert a_run(tmp_path, source, seconds=1.0)["dropped"] == 7


def test_a_ctrl_c_keeps_what_was_already_collected(tmp_path):
    class Interrupted(FakeTrajectorySource):
        drains = 0

        def drain(self):
            type(self).drains += 1
            if type(self).drains > 2:
                raise KeyboardInterrupt
            return super().drain()

    a_run(tmp_path, a_source(Interrupted), seconds=60.0)
    assert len(rows(tmp_path / "run.csv")) > 0
    assert (tmp_path / "run.json").exists()
