"""Watching the car's condition during a run.

BeamNG's damage is persistent: a car that has been pranged keeps driving with
bent suspension and a dragging panel, so its engine load and thermals shift.
Left alone over a long unattended run that quietly corrupts the data, and the
corrupted part looks exactly like the clean part.

So damage is watched, the car is repaired when it gets bad, and the repair is
recorded as a discontinuity rather than smoothed over.
"""

import pytest

from control.health import HealthMonitor
from sim.backend import VehicleState


def a_state(damage=0.0, speed=10.0, t=0.0):
    return VehicleState(damage=damage, speed_mps=speed, sim_time_s=t)


def a_monitor(**kwargs):
    monitor = HealthMonitor(**kwargs)
    monitor.update(a_state(damage=0.0, t=0.0))
    return monitor


# -- damage ---------------------------------------------------------------


def test_an_undamaged_car_needs_nothing():
    monitor = a_monitor()
    monitor.update(a_state(damage=5.0, t=1.0))
    assert not monitor.needs_repair


def test_a_badly_damaged_car_needs_repair():
    monitor = a_monitor(repair_above=100.0)
    monitor.update(a_state(damage=450.0, t=1.0))
    assert monitor.needs_repair


def test_damage_present_before_the_run_is_not_this_run_s_fault():
    monitor = HealthMonitor(repair_above=100.0)
    monitor.update(a_state(damage=800.0, t=0.0))
    monitor.update(a_state(damage=820.0, t=1.0))
    assert not monitor.needs_repair


def test_repairing_clears_the_debt():
    monitor = a_monitor(repair_above=100.0)
    monitor.update(a_state(damage=450.0, t=1.0))
    monitor.after_repair(a_state(damage=0.0, t=2.0))
    monitor.update(a_state(damage=10.0, t=3.0))
    assert not monitor.needs_repair


def test_the_repair_threshold_is_configurable():
    monitor = a_monitor(repair_above=10.0)
    monitor.update(a_state(damage=40.0, t=1.0))
    assert monitor.needs_repair


def test_damage_taken_is_reported():
    monitor = a_monitor()
    monitor.update(a_state(damage=250.0, t=1.0))
    assert monitor.damage_taken == pytest.approx(250.0)


# -- stuck ----------------------------------------------------------------


def test_a_moving_car_is_not_stuck():
    monitor = a_monitor()
    for second in range(1, 60):
        monitor.update(a_state(speed=8.0, t=float(second)))
    assert not monitor.is_stuck


def test_a_car_that_stops_briefly_is_not_stuck():
    # Waiting at a junction is not being stuck.
    monitor = a_monitor(stuck_after_s=30.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    assert not monitor.is_stuck


def test_a_car_that_never_moves_again_is_stuck():
    monitor = a_monitor(stuck_after_s=30.0)
    for second in range(1, 60):
        monitor.update(a_state(speed=0.0, t=float(second)))
    assert monitor.is_stuck


def test_moving_again_clears_being_stuck():
    monitor = a_monitor(stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    monitor.update(a_state(speed=9.0, t=21.0))
    assert not monitor.is_stuck


def test_recovering_clears_being_stuck():
    monitor = a_monitor(stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    monitor.after_repair(a_state(speed=0.0, t=21.0))
    assert not monitor.is_stuck


# -- what to do about it ---------------------------------------------------


def test_a_damaged_car_is_repaired_where_it_stands():
    monitor = a_monitor(repair_above=100.0)
    monitor.update(a_state(damage=450.0, speed=12.0, t=1.0))
    assert monitor.recommended_action() == "repair"


def test_a_stuck_car_is_recovered_to_the_road_instead():
    # Repairing in place leaves it exactly where it is wedged.
    monitor = a_monitor(stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    assert monitor.recommended_action() == "recover"


def test_a_car_that_is_both_stuck_and_damaged_is_recovered():
    monitor = a_monitor(repair_above=100.0, stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(damage=900.0, speed=0.0, t=float(second)))
    assert monitor.recommended_action() == "recover"


def test_a_healthy_car_is_left_alone():
    monitor = a_monitor()
    monitor.update(a_state(damage=5.0, speed=10.0, t=1.0))
    assert monitor.recommended_action() is None


# -- giving up on a car that cannot be rescued -----------------------------


def test_a_car_that_moves_after_a_repair_is_fine():
    monitor = a_monitor(stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    monitor.after_repair(a_state(speed=0.0, t=21.0))
    monitor.update(a_state(speed=9.0, t=22.0))
    assert monitor.futile_repairs == 0


def test_repairs_that_change_nothing_are_counted():
    # An unrecoverable car -- flipped into a ravine, wedged under a bridge --
    # would otherwise be recovered every few seconds for the whole run while
    # collecting nothing.
    # Built stationary from the start: a monitor that has seen the car move is
    # entitled to treat its first repair as worth trying.
    monitor = HealthMonitor(stuck_after_s=5.0)
    monitor.update(a_state(speed=0.0, t=0.0))
    t = 1.0
    for _ in range(3):
        for _ in range(6):
            monitor.update(a_state(speed=0.0, t=t)); t += 1.0
        monitor.after_repair(a_state(speed=0.0, t=t)); t += 1.0
    assert monitor.futile_repairs == 3


def test_a_car_that_cannot_be_rescued_is_reported():
    monitor = HealthMonitor(stuck_after_s=5.0, give_up_after=2)
    monitor.update(a_state(speed=0.0, t=0.0))
    t = 1.0
    for _ in range(3):
        for _ in range(6):
            monitor.update(a_state(speed=0.0, t=t)); t += 1.0
        monitor.after_repair(a_state(speed=0.0, t=t)); t += 1.0
    assert monitor.beyond_help


def test_a_healthy_run_is_never_beyond_help():
    monitor = a_monitor()
    for second in range(1, 100):
        monitor.update(a_state(speed=10.0, t=float(second)))
    assert not monitor.beyond_help


# -- a neighbourhood the car cannot cope with ------------------------------


def prang(monitor, t, damage=4000.0):
    """A crash bad enough to matter -- above the default repair threshold,
    which counts bent panels and so sits in the thousands, not the hundreds."""
    monitor.update(a_state(damage=damage, speed=8.0, t=t))
    monitor.after_repair(a_state(damage=0.0, speed=8.0, t=t))


def test_one_prang_is_not_a_pattern():
    monitor = a_monitor(crashes_per_window=3, crash_window_s=120.0)
    prang(monitor, 10.0)
    assert not monitor.in_trouble


def test_repeated_crashes_close_together_are_a_pattern():
    # Somewhere the AI cannot cope: a tight neighbourhood, a junction it keeps
    # misjudging. Repairing it there just repeats the crash.
    monitor = a_monitor(crashes_per_window=3, crash_window_s=120.0)
    for t in (10.0, 40.0, 70.0):
        prang(monitor, t)
    assert monitor.in_trouble


def test_crashes_spread_out_are_not_a_pattern():
    monitor = a_monitor(crashes_per_window=3, crash_window_s=120.0)
    for t in (10.0, 400.0, 800.0):
        prang(monitor, t)
    assert not monitor.in_trouble


def test_relocating_gives_the_car_a_clean_slate():
    monitor = a_monitor(crashes_per_window=3, crash_window_s=120.0)
    for t in (10.0, 40.0, 70.0):
        prang(monitor, t)
    monitor.after_relocation(a_state(t=80.0))
    assert not monitor.in_trouble


def test_how_many_crashes_count_as_trouble_is_configurable():
    monitor = a_monitor(crashes_per_window=2, crash_window_s=120.0)
    for t in (10.0, 40.0):
        prang(monitor, t)
    assert monitor.in_trouble


def test_moving_the_car_is_recommended_when_it_is_in_trouble():
    monitor = a_monitor(crashes_per_window=2, crash_window_s=120.0)
    for t in (10.0, 40.0):
        prang(monitor, t)
    monitor.update(a_state(damage=500.0, speed=8.0, t=45.0))
    assert monitor.recommended_action() == "relocate"


def test_relocating_outranks_a_plain_repair():
    monitor = a_monitor(repair_above=100.0, crashes_per_window=2,
                        crash_window_s=120.0)
    for t in (10.0, 40.0):
        prang(monitor, t)
    monitor.update(a_state(damage=900.0, speed=8.0, t=45.0))
    assert monitor.recommended_action() == "relocate"


# -- the loop that hung a real run ----------------------------------------


def test_repairing_is_not_itself_recorded_as_a_crash():
    # Repairs used to append a crash time, so three repairs of ONE incident
    # looked like three crashes, which triggered "relocate", which repaired
    # again, which recorded another crash. That is the loop.
    monitor = a_monitor(repair_above=100.0, crashes_per_window=3,
                        crash_window_s=180.0)
    monitor.update(a_state(damage=500.0, speed=8.0, t=10.0))
    for repair in range(5):
        monitor.after_repair(a_state(damage=0.0, speed=8.0, t=11.0 + repair))
        monitor.update(a_state(damage=0.0, speed=8.0, t=11.5 + repair))
    assert monitor.recent_crashes == 1
    assert not monitor.in_trouble


def test_separate_crashes_are_still_counted_separately():
    monitor = a_monitor(repair_above=100.0, crashes_per_window=3,
                        crash_window_s=180.0)
    for t in (10.0, 40.0, 70.0):
        monitor.update(a_state(damage=500.0, speed=8.0, t=t))
        monitor.after_repair(a_state(damage=0.0, speed=8.0, t=t + 1))
    assert monitor.recent_crashes == 3
    assert monitor.in_trouble


def test_a_single_long_crash_counts_once():
    monitor = a_monitor(repair_above=100.0, crashes_per_window=2,
                        crash_window_s=180.0)
    for t in range(10, 40):
        monitor.update(a_state(damage=500.0, speed=0.0, t=float(t)))
    assert monitor.recent_crashes == 1


def test_nowhere_to_go_means_no_relocation_is_recommended():
    # With no refuge configured there is nowhere better to send it, and
    # recommending a move that cannot happen produced an action every tick.
    monitor = a_monitor(repair_above=100.0, crashes_per_window=2,
                        crash_window_s=180.0)
    for t in (10.0, 40.0):
        monitor.update(a_state(damage=500.0, speed=8.0, t=t))
        monitor.after_repair(a_state(damage=0.0, speed=8.0, t=t + 1))
    monitor.update(a_state(damage=0.0, speed=8.0, t=45.0))
    assert monitor.recommended_action(can_relocate=False) is None


def test_a_refuge_still_gets_a_relocation():
    monitor = a_monitor(repair_above=100.0, crashes_per_window=2,
                        crash_window_s=180.0)
    for t in (10.0, 40.0):
        monitor.update(a_state(damage=500.0, speed=8.0, t=t))
        monitor.after_repair(a_state(damage=0.0, speed=8.0, t=t + 1))
    monitor.update(a_state(damage=0.0, speed=8.0, t=45.0))
    assert monitor.recommended_action(can_relocate=True) == "relocate"


def test_a_stuck_car_is_still_recovered_with_nowhere_to_relocate_to():
    monitor = a_monitor(stuck_after_s=10.0)
    for second in range(1, 20):
        monitor.update(a_state(speed=0.0, t=float(second)))
    assert monitor.recommended_action(can_relocate=False) == "recover"


def test_cosmetic_damage_does_not_warrant_a_repair():
    # BeamNG's damageSum counts every bent panel. A scraped wing changes
    # nothing about engine load or thermals; stopping to repair it interrupts
    # the run for no gain.
    monitor = a_monitor()
    monitor.update(a_state(damage=200.0, speed=8.0, t=10.0))
    assert not monitor.needs_repair
