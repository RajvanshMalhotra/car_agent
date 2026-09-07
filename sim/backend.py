"""The sim boundary.

This module is one interface wide. Two implementations exist: the fake
kinematic backend (testable anywhere) and, on the Windows machine, the
gamepad + UDP backend. A `beamngpy` backend drops in here unchanged if the
BeamNG.tech licence lands. Nothing outside `sim/` imports anything
BeamNG-specific.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlInput:
    """What the controller asks the vehicle to do, in normalised units."""

    throttle: float  # 0..1
    brake: float  # 0..1
    steering: float  # -1..1, positive is left (ISO 8855 yaw)

    def __post_init__(self) -> None:
        for name, low, high in (
            ("throttle", 0.0, 1.0),
            ("brake", 0.0, 1.0),
            ("steering", -1.0, 1.0),
        ):
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"{name}: {value} outside [{low}, {high}]")


@dataclass
class VehicleState:
    """Vehicle state as read back from the simulator.

    ISO 8855: x forward, y left, heading counter-clockwise from +x.
    """

    sim_time_s: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0
    speed_mps: float = 0.0

    # Engine and thermal channels. These -- not traction -- are what the
    # battery layer consumes: grid corrosion tracks under-bonnet temperature,
    # sulfation tracks cranking and short trips.
    rpm: float = 0.0
    coolant_temp_c: float = 20.0
    underbonnet_temp_c: float = 20.0
    oil_temp_c: float = 20.0
    gear: int = 0
    fuel_fraction: float = 1.0
    engine_on: bool = False
    crank_count: int = 0

    #: Cumulative damage as the simulator reports it. Unitless and monotonic;
    #: a kerb strike registers in the tens, a real impact in the hundreds.
    damage: float = 0.0

    #: What the vehicle actually did, as reported back by the simulator -- not
    #: what the controller asked for. Engine load follows the actual pedals.
    throttle: float = 0.0
    brake: float = 0.0

    def snapshot(self) -> "VehicleState":
        """A detached copy, safe to hold across further simulation steps."""
        return dataclasses.replace(self)


class SimBackend(ABC):
    """Apply a control, read the resulting state. Everything else is detail."""

    @abstractmethod
    def apply_control(self, control: ControlInput) -> None:
        """Send one control input to the vehicle."""

    @abstractmethod
    def read_state(self) -> VehicleState:
        """Return the most recent vehicle state, as a detached snapshot.

        Implementations must not hand out a reference to live internal state:
        callers routinely hold one state to compare against the next.
        """

    @abstractmethod
    def reset(self) -> VehicleState:
        """Return the vehicle to its starting condition and report it."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources. Safe to call more than once."""
