#!/usr/bin/env python3
"""Drive a whole campaign and write down what came out.

    py collect.py --minutes 15                    every behaviour, hot and cool
    py collect.py --minutes 15 --to depot         every behaviour, same journey
    py collect.py --dry-run                       what it would do, and how long
    py collect.py --status                        what is done and what is left

This is `drive.py` in a loop, with the loop written down. A campaign is hours
of real time and it *will* be interrupted -- a crash, a closed lid, an update.
Every finished run goes into `runs/ledger.json` immediately, so starting the
same command again picks up where it stopped instead of driving it all twice.

Hot cells are driven first. Ambient temperature moves corrosion about 2.9x
where driving style moves it 1.2-1.5x, so a campaign cut short after a third
still answers the main question.

One caveat worth knowing before you leave it running: ambient temperature is
currently a parameter of *our* thermal model, not the game's weather. Unless
the MCP server turns out to expose weather control -- run windows_mcp_probe.py
to find out -- the ambient axis varies how we interpret the drive rather than
the drive itself. The behaviour axis is real either way.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import drive  # noqa: E402
from campaign.ledger import Ledger  # noqa: E402
from campaign.plan import plan_runs  # noqa: E402

HERE = Path(__file__).parent
RUNS = HERE / "runs"
LEDGER = RUNS / "ledger.json"

#: The two ends of the range that matters, plus the middle. Ambient is the
#: dominant driver of grid corrosion, so it gets the coverage.
DEFAULT_AMBIENTS = (42.0, 33.0, 25.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=15.0,
                        help="per run. Under 10 the warmup dominates and the "
                             "game-value comparison cannot be trusted")
    parser.add_argument("--ambients", type=float, nargs="+",
                        default=list(DEFAULT_AMBIENTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--behaviour", action="append", dest="behaviours",
                        help="restrict to these styles; repeatable")
    parser.add_argument("--to", help="drive to a saved place instead of roaming")
    parser.add_argument("--from", dest="start", help="teleport here first")
    parser.add_argument("--mode", default="span", choices=("span", "random"))
    parser.add_argument("--refuge", help="somewhere to move to if it keeps crashing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args(argv)

    specs = drive.behaviours()
    if args.behaviours:
        wanted = []
        for needle in args.behaviours:
            spec = drive.find_behaviour(needle)
            if spec is None:
                print(f"No style matching {needle!r}", file=sys.stderr)
                return 1
            wanted.append(spec)
        specs = wanted
    if not specs:
        print(f"No behaviours in {drive.BEHAVIOURS}.", file=sys.stderr)
        print('Make one:  ./agent.py generate "a delivery courier in Delhi"',
              file=sys.stderr)
        return 1

    cells = plan_runs(specs, args.ambients, repeats=args.repeats)
    ledger = Ledger(args.ledger)
    left = ledger.remaining(cells)

    if args.status:
        return report_status(cells, ledger)

    print(f"\n  campaign: {len(cells)} runs, {len(cells) - len(left)} already done")
    print(f"  left    : {len(left)} x {args.minutes:.0f} min = "
          f"{len(left) * args.minutes / 60:.1f} hours")
    print(f"  ledger  : {args.ledger}\n")

    if args.dry_run:
        for cell in left:
            print(f"    {cell.key:28} {cell.label}")
        return 0

    if not left:
        print("  Nothing left to drive.")
        return 0

    return run_all(left, ledger, args)


def run_all(cells, ledger, args) -> int:
    """Drive each remaining cell, writing the ledger as it goes."""
    for index, cell in enumerate(cells, start=1):
        spec = drive.find_behaviour(cell.spec_hash)
        if spec is None:
            ledger.failed(cell.key, f"behaviour {cell.spec_hash} is gone")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {index}/{len(cells)}  {cell.label}")
        print(f"{'=' * 70}")

        # The ambient temperature is the campaign's, not the behaviour's.
        at_ambient = replace(spec, ambient_temp_c=cell.ambient_c)
        before = newest_run()
        try:
            code = drive.drive(at_ambient, run_args(args))
        except KeyboardInterrupt:
            # Deliberate: the ledger already holds everything finished, so
            # stopping here loses only the run in progress.
            print("\n  campaign stopped. Run the same command to carry on.")
            return 0
        except Exception as error:
            ledger.failed(cell.key, f"{type(error).__name__}: {error}")
            print(f"  failed: {error}")
            continue

        produced = newest_run()
        if code != 0 or produced is None or produced == before:
            ledger.failed(cell.key, f"drive.py exited {code} with no new run")
            print(f"  failed: no data produced")
            continue
        ledger.finished(cell.key, csv=str(produced), rows=count_rows(produced),
                        ambient_c=cell.ambient_c, behaviour=cell.name)

        # A moment between runs. Back-to-back MCP sessions on a game that has
        # just been driven hard is where it tends to fall over.
        if index < len(cells):
            time.sleep(5.0)

    print(f"\n  campaign finished. {len(ledger.done_keys())} runs on disk.")
    print(f"  Check one:  python3 does_the_game_matter.py <a csv from runs/>")
    return 0


def run_args(args):
    """The subset of the campaign's arguments that `drive.drive` understands."""
    import types

    return types.SimpleNamespace(
        minutes=args.minutes, to=args.to, start=args.start, mode=args.mode,
        refuge=args.refuge, seed=0,
        repair_above=2500.0, stuck_after=45.0, crashes_before_moving=3,
        endpoint=args.endpoint or _default_endpoint(),
    )


def _default_endpoint() -> str:
    from sim.mcp_client import DEFAULT_ENDPOINT

    return DEFAULT_ENDPOINT


def newest_run() -> Path | None:
    runs = sorted(RUNS.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def count_rows(csv_path: Path) -> int:
    with open(csv_path) as handle:
        return max(0, sum(1 for _ in handle) - 1)


def report_status(cells, ledger) -> int:
    done = ledger.done_keys()
    for cell in cells:
        if cell.key in done:
            mark, note = "done", ""
        else:
            why = ledger.why_failed(cell.key)
            mark, note = ("failed", f"  {why}") if why else ("todo", "")
        print(f"  {mark:7} {cell.label:34}{note}")
    print(f"\n  {len(done)}/{len(cells)} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
