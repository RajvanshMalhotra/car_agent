"""Running many drives without losing the ones already done.

A campaign is hours of real time on someone else's laptop. It will be
interrupted -- a crash, a closed lid, a game update -- and restarting from the
beginning each time means it never finishes. So what has been done is written
down as it happens, and a restart picks up where it stopped.
"""

import json

import pytest

from campaign.plan import Cell, plan_runs
from campaign.ledger import Ledger


class Spec:
    """Just enough of a BehaviourSpec to plan against."""

    def __init__(self, name, spec_hash):
        self.name = name
        self.spec_hash = spec_hash


ECON = Spec("Economical", "aaa")
BRISK = Spec("Brisk", "bbb")


# -- planning -------------------------------------------------------------


def test_every_behaviour_is_paired_with_every_ambient():
    runs = plan_runs([ECON, BRISK], ambients=[25.0, 42.0])
    assert len(runs) == 4
    assert {(r.spec_hash, r.ambient_c) for r in runs} == {
        ("aaa", 25.0), ("aaa", 42.0), ("bbb", 25.0), ("bbb", 42.0),
    }


def test_repeats_produce_distinct_cells():
    runs = plan_runs([ECON], ambients=[25.0], repeats=3)
    assert len(runs) == 3
    assert len({r.key for r in runs}) == 3


def test_the_hot_end_is_driven_first():
    # If only some of a campaign gets run, it should be the half that carries
    # the most information. Ambient moves corrosion about 2.9x against driving
    # style's 1.2-1.5x, so the hot cells matter most.
    runs = plan_runs([ECON], ambients=[25.0, 42.0, 33.0])
    assert [r.ambient_c for r in runs] == [42.0, 33.0, 25.0]


def test_a_cell_key_is_stable_across_runs():
    first = plan_runs([ECON], ambients=[25.0])[0]
    again = plan_runs([ECON], ambients=[25.0])[0]
    assert first.key == again.key


def test_a_cell_key_distinguishes_ambient():
    cool, hot = plan_runs([ECON], ambients=[25.0, 42.0])[1], \
        plan_runs([ECON], ambients=[25.0, 42.0])[0]
    assert cool.key != hot.key


def test_planning_without_behaviours_is_refused():
    with pytest.raises(ValueError):
        plan_runs([], ambients=[25.0])


def test_planning_without_ambients_is_refused():
    with pytest.raises(ValueError):
        plan_runs([ECON], ambients=[])


# -- the ledger -----------------------------------------------------------


def test_a_fresh_ledger_has_done_nothing(tmp_path):
    assert Ledger(tmp_path / "ledger.json").done_keys() == set()


def test_a_finished_run_is_remembered(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).finished("aaa@25", csv="runs/a.csv", rows=900)
    assert "aaa@25" in Ledger(path).done_keys()


def test_a_failed_run_is_not_counted_as_done(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).failed("aaa@25", "the car never set off")
    assert Ledger(path).done_keys() == set()


def test_a_failure_is_still_written_down(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).failed("aaa@25", "the car never set off")
    assert "never set off" in Ledger(path).why_failed("aaa@25")


def test_a_retry_that_succeeds_clears_the_failure(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path)
    ledger.failed("aaa@25", "the car never set off")
    ledger.finished("aaa@25", csv="runs/a.csv", rows=900)
    assert "aaa@25" in Ledger(path).done_keys()
    assert Ledger(path).why_failed("aaa@25") == ""


def test_what_is_left_skips_what_is_already_done(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).finished("aaa@42", csv="runs/a.csv", rows=900)
    runs = plan_runs([ECON], ambients=[25.0, 42.0])
    left = Ledger(path).remaining(runs)
    assert [r.ambient_c for r in left] == [25.0]


def test_the_ledger_survives_being_reopened(tmp_path):
    path = tmp_path / "ledger.json"
    for key in ("a", "b", "c"):
        Ledger(path).finished(key, csv=f"runs/{key}.csv", rows=10)
    assert Ledger(path).done_keys() == {"a", "b", "c"}


def test_the_ledger_is_readable_by_a_person(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).finished("aaa@25", csv="runs/a.csv", rows=900)
    stored = json.loads(path.read_text())
    assert stored["aaa@25"]["rows"] == 900
    assert stored["aaa@25"]["csv"] == "runs/a.csv"


def test_a_run_that_collected_nothing_is_not_treated_as_done(tmp_path):
    # An empty CSV means the run technically completed and produced no data.
    # Counting it as done would silently leave a hole in the campaign.
    path = tmp_path / "ledger.json"
    Ledger(path).finished("aaa@25", csv="runs/a.csv", rows=0)
    assert Ledger(path).done_keys() == set()
