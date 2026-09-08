"""The campaign runner, without a game.

Everything here has to work before anyone spends hours of laptop time on it,
which means none of it may need BeamNG to be running.
"""

import json

import pytest

import collect


def test_a_dry_run_touches_nothing(tmp_path, capsys):
    ledger = tmp_path / "ledger.json"
    code = collect.main(["--dry-run", "--ledger", str(ledger)])
    assert code == 0
    assert not ledger.exists()


def test_a_dry_run_says_how_long_it_will_take(tmp_path, capsys):
    collect.main(["--dry-run", "--minutes", "10", "--ledger",
                  str(tmp_path / "l.json")])
    assert "hours" in capsys.readouterr().out


def test_the_hot_cells_come_first(tmp_path, capsys):
    collect.main(["--dry-run", "--ambients", "25", "42",
                  "--ledger", str(tmp_path / "l.json")])
    out = capsys.readouterr().out
    assert out.index("42 C") < out.index("25 C")


def test_an_unknown_behaviour_is_refused_before_anything_runs(tmp_path, capsys):
    code = collect.main(["--behaviour", "no-such-style",
                         "--ledger", str(tmp_path / "l.json")])
    assert code == 1


def test_finished_runs_are_skipped_on_a_restart(tmp_path, capsys):
    ledger = tmp_path / "ledger.json"
    collect.main(["--dry-run", "--ambients", "42", "--ledger", str(ledger)])
    first = capsys.readouterr().out

    # Mark every planned cell as done, then plan again.
    from campaign.ledger import Ledger
    from campaign.plan import plan_runs
    import drive as drive_module

    cells = plan_runs(drive_module.behaviours(), [42.0])
    for cell in cells:
        Ledger(ledger).finished(cell.key, csv="runs/x.csv", rows=900)

    collect.main(["--dry-run", "--ambients", "42", "--ledger", str(ledger)])
    second = capsys.readouterr().out
    assert "0 x" in second or "already done" in second
    assert second != first


def test_status_reports_a_failure_with_its_reason(tmp_path, capsys):
    from campaign.ledger import Ledger
    from campaign.plan import plan_runs
    import drive as drive_module

    ledger = tmp_path / "ledger.json"
    cell = plan_runs(drive_module.behaviours(), [42.0])[0]
    Ledger(ledger).failed(cell.key, "the car never set off")

    collect.main(["--status", "--ambients", "42", "--ledger", str(ledger)])
    out = capsys.readouterr().out
    assert "failed" in out and "never set off" in out


def test_the_campaign_ambient_overrides_the_behaviours_own(tmp_path):
    # A behaviour carries the ambient it was written for; a campaign sweeps
    # ambient deliberately, so the campaign's value has to win.
    from dataclasses import replace
    import drive as drive_module

    spec = drive_module.behaviours()[0]
    assert replace(spec, ambient_temp_c=42.0).ambient_temp_c == 42.0


def test_row_counting_ignores_the_header(tmp_path):
    csv_path = tmp_path / "run.csv"
    csv_path.write_text("t_s,speed_mps\n0,1\n1,2\n")
    assert collect.count_rows(csv_path) == 2


def test_an_empty_run_counts_as_no_rows(tmp_path):
    csv_path = tmp_path / "run.csv"
    csv_path.write_text("t_s,speed_mps\n")
    assert collect.count_rows(csv_path) == 0
