"""Watching the car's condition during a run.

BeamNG's damage is persistent. A car that has been pranged keeps driving with
bent suspension and a dragging panel, so its engine load, drag and thermals all
shift -- and over a long unattended run that quietly corrupts the data, with the
corrupted part looking exactly like the clean part.

Two things go wrong on a long run and they want different answers. A damaged car
is repaired where it stands, because moving it would throw away the route
position for nothing. A *stuck* car -- flipped, or wedged against a wall -- has
to be recovered to the road, because repairing it in place leaves it exactly
where it is wedged.

Either way the repair is a discontinuity in the data, not something to smooth
over: the run loop records it as the start of a new trip.
"""

from __future__ import annotations

from sim.backend import VehicleState

#: Damage above which the car is repaired. BeamNG's damageSum is unitless and
#: counts every bent panel, so it rises steadily from scrapes that change
#: nothing: a kerb strike registers in the tens, a scraped wing in the low
#: hundreds, a structural impact in the thousands.
#:
#: Only the last of those is worth stopping for. A dented car drives the same
#: -- same engine load, same thermals, same data -- and repairing it costs a
#: trip boundary and a discontinuity in the log for no gain. The threshold is
#: set where damage starts to change how the car actually drives.
DEFAULT_REPAIR_ABOVE = 2500.0

#: Below this the car counts as stationary.
STOPPED_SPEED_MPS = 0.5

#: How long stationary before it is stuck rather than waiting. Long enough to
#: sit at a junction or a stop, short enough not to waste a run.
DEFAULT_STUCK_AFTER_S = 45.0

#: How many repairs that change nothing before the car is declared beyond help.
#: A car flipped into a ravine can be recovered every few seconds for an entire
#: run while collecting nothing at all.
DEFAULT_GIVE_UP_AFTER = 3

#: Repeated crashes close together mean the car is somewhere it cannot cope --
#: a tight neighbourhood, a junction it keeps misjudging. Repairing it on the
#: spot just repeats the crash, so it is moved somewhere easier instead.
DEFAULT_CRASHES_PER_WINDOW = 3
DEFAULT_CRASH_WINDOW_S = 180.0


class HealthMonitor:
    def __init__(
        self,
        repair_above: float = DEFAULT_REPAIR_ABOVE,
        stuck_after_s: float = DEFAULT_STUCK_AFTER_S,
        give_up_after: int = DEFAULT_GIVE_UP_AFTER,
        crashes_per_window: int = DEFAULT_CRASHES_PER_WINDOW,
        crash_window_s: float = DEFAULT_CRASH_WINDOW_S,
    ) -> None:
        self.repair_above = repair_above
        self.stuck_after_s = stuck_after_s
        self.give_up_after = give_up_after
        self.crashes_per_window = crashes_per_window
        self.crash_window_s = crash_window_s
        self.crash_times: list[float] = []
        #: Whether the crash currently being suffered has already been counted.
        #: Damage is persistent, so without this one impact is recorded on
        #: every tick until it is repaired -- and every repair recorded another.
        self._crash_counted = False
        self._now_s = 0.0
        #: Repairs after which the car still did not move.
        self.futile_repairs = 0
        #: Whether the car has moved since the last repair. Starts False: a car
        #: that has never moved is not a car worth repairing repeatedly.
        self._moved_since_repair = False
        self._baseline_damage: float | None = None
        self.damage_taken = 0.0
        self.stationary_since_s: float | None = None
        self.stationary_for_s = 0.0

    def update(self, state: VehicleState) -> None:
        self._now_s = state.sim_time_s
        # Damage already on the car when the run began is not this run's fault.
        if self._baseline_damage is None:
            self._baseline_damage = state.damage
        self.damage_taken = max(0.0, state.damage - self._baseline_damage)

        # One impact, one crash. The car stays damaged until it is repaired,
        # so counting per tick -- or per repair -- turned a single accident
        # into a pile-up and sent the run relocating in circles.
        if self.damage_taken > self.repair_above:
            if not self._crash_counted:
                self.crash_times.append(state.sim_time_s)
                self._crash_counted = True
        else:
            self._crash_counted = False

        if state.speed_mps >= STOPPED_SPEED_MPS:
            self.stationary_since_s = None
            self.stationary_for_s = 0.0
            self._moved_since_repair = True
            self.futile_repairs = 0
        else:
            if self.stationary_since_s is None:
                self.stationary_since_s = state.sim_time_s
            self.stationary_for_s = state.sim_time_s - self.stationary_since_s

    def after_repair(self, state: VehicleState) -> None:
        """Re-baseline. A repair sets damage back to zero, and the car that
        comes out of it is a different car as far as this run is concerned."""
        self._now_s = state.sim_time_s
        if not self._moved_since_repair:
            self.futile_repairs += 1
        self._moved_since_repair = False
        self._baseline_damage = state.damage
        self.damage_taken = 0.0
        self._crash_counted = False
        self.stationary_since_s = None
        self.stationary_for_s = 0.0

    @property
    def needs_repair(self) -> bool:
        return self.damage_taken > self.repair_above

    @property
    def is_stuck(self) -> bool:
        return self.stationary_for_s >= self.stuck_after_s

    def after_relocation(self, state: VehicleState) -> None:
        """A clean slate somewhere else. The crash history belonged to the old
        place and says nothing about the new one."""
        self.crash_times.clear()
        self.after_repair(state)
        self.futile_repairs = 0
        self._moved_since_repair = True

    @property
    def recent_crashes(self) -> int:
        return sum(
            1 for t in self.crash_times if self._now_s - t <= self.crash_window_s
        )

    @property
    def in_trouble(self) -> bool:
        """Crashing repeatedly in one place, rather than having one accident."""
        return self.recent_crashes >= self.crashes_per_window

    @property
    def beyond_help(self) -> bool:
        """Repeated recoveries have not got the car moving again."""
        return self.futile_repairs >= self.give_up_after

    def recommended_action(self, can_relocate: bool = True) -> str | None:
        """`"relocate"`, `"recover"`, `"repair"`, or None.

        Repeated crashes come first, then stuck -- a wedged car repaired in
        place is still wedged -- then plain damage.

        `can_relocate` is False when no refuge has been saved. Recommending a
        move that the caller cannot make returns an action on every tick and
        nothing ever changes, which is a busy loop rather than a recovery.
        """
        # Moving it outranks patching it up: somewhere it keeps crashing, a
        # repair on the spot only buys another crash.
        if can_relocate and self.in_trouble:
            return "relocate"
        if self.is_stuck:
            return "recover"
        if self.needs_repair:
            return "repair"
        return None
