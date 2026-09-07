"""Find out which way the car is pointing, by driving it.

A synthetic route has to start where the car is and run the way the car faces.
Taking that from the reported orientation would mean knowing the vehicle's
forward axis and the sign convention of yaw, and neither is documented. Getting
it wrong is not subtle: the car drives off at an angle from the first metre and
the run aborts within seconds.

So it is measured. Roll forward with the wheels straight and see which way the
car actually goes. Displacement cannot disagree with the frame the positions are
reported in, which is the same argument used for heading on both backends.
"""

from __future__ import annotations

import math

from control.path import Path
from sim.backend import ControlInput, SimBackend

#: Enough movement for the direction to be unambiguous, short enough to happen
#: in a couple of seconds.
DEFAULT_MIN_DISTANCE_M = 5.0

#: Gentle: this is a measurement, not part of the behaviour being studied.
DEFAULT_THROTTLE = 0.25


def measure_heading(
    backend: SimBackend,
    dt: float,
    throttle: float = DEFAULT_THROTTLE,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    timeout_s: float = 15.0,
) -> tuple[float, float, float]:
    """Roll forward briefly. Returns `(x, y, heading_rad)` where it ended up.

    Raises if the car will not move -- better than silently aligning a route to
    a heading measured from noise.
    """
    start = backend.read_state()
    origin = (start.x_m, start.y_m)

    elapsed = 0.0
    state = start
    while elapsed < timeout_s:
        backend.apply_control(ControlInput(throttle=throttle, brake=0.0, steering=0.0))
        state = backend.read_state()
        elapsed += dt
        if math.dist((state.x_m, state.y_m), origin) >= min_distance_m:
            break

    backend.apply_control(ControlInput(0.0, 0.0, 0.0))

    dx, dy = state.x_m - origin[0], state.y_m - origin[1]
    travelled = math.hypot(dx, dy)
    if travelled < min_distance_m * 0.5:
        raise RuntimeError(
            f"the vehicle did not move ({travelled:.2f} m in {elapsed:.1f} s). "
            "Check it is in gear, on the ground, and not against a wall."
        )
    return state.x_m, state.y_m, math.atan2(dy, dx)


def straight_route_from(
    origin: tuple[float, float],
    heading_rad: float,
    length_m: float,
    spacing_m: float = 5.0,
) -> Path:
    """A straight route starting at `origin`, running along `heading_rad`."""
    steps = max(2, int(length_m / spacing_m) + 1)
    return Path(
        [
            (
                origin[0] + i * spacing_m * math.cos(heading_rad),
                origin[1] + i * spacing_m * math.sin(heading_rad),
            )
            for i in range(steps)
        ]
    )
