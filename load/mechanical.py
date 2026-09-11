"""Road load, and the mechanical power the engine has to produce for it.

None of this reaches the battery as current. Parent spec section 2: applying
the EV chain `P_elec = P_mech / eta + P_acc` to a combustion car yields battery
currents around 8000 A, three orders of magnitude out, because in a petrol car
the engine supplies the traction and the battery supplies none of it.

Mechanical power matters through a different route entirely -- engine load
heats the engine bay, and bay temperature drives grid corrosion. So this
module's output goes to `load/thermal.py`, never to `load/electrical.py`.

`mass_kg` is read from the trajectory rather than the scenario because the
canonical schema measures it live, as a node-mass sum. On the recorded file it
is an assumed constant, and `load.legacy.assumed_inputs` is how a caller finds
that out.
"""

from __future__ import annotations

import math

from load.spec import BatteryScenario

GRAVITY_MPS2 = 9.81

READS: tuple[str, ...] = ("speed_mps", "a_long_mps2", "grade_rad", "mass_kg")


def road_load_n(sample, scenario: BatteryScenario) -> float:
    """Total tractive force: inertia, gravity, aerodynamic drag, rolling."""
    mass = float(sample["mass_kg"])
    speed = float(sample["speed_mps"])
    grade = float(sample["grade_rad"])
    inertia = mass * float(sample["a_long_mps2"])
    gravity = mass * GRAVITY_MPS2 * math.sin(grade)
    drag = 0.5 * scenario.air_density * scenario.cd_a_m2 * speed * speed
    rolling = scenario.crr * mass * GRAVITY_MPS2 * math.cos(grade)
    return inertia + gravity + drag + rolling


def mech_power_w(sample, scenario: BatteryScenario) -> float:
    """Force times speed. Zero when stationary, however hard the engine works."""
    return road_load_n(sample, scenario) * float(sample["speed_mps"])
