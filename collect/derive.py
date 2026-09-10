"""The two channels the game does not report and we compute.

Both exist because the raw measurement is not the quantity wanted.

`dir_z` is the z component of a unit forward vector, so its arcsine is the road
grade -- and a grade in radians is what the road-load equation takes.

`sensors.gx` is a real accelerometer and therefore reads gravity as well as
motion. A car parked on a 30 degree slope shows 4.9 m/s^2 of longitudinal
acceleration while going nowhere. Subtracting the gravity component along the
forward axis leaves what the vehicle is actually doing.
"""

from __future__ import annotations

import math

GRAVITY_MPS2 = 9.81


def add_derived(sample: dict) -> dict:
    dir_z = max(-1.0, min(1.0, float(sample["dir_z"])))
    sample["grade_rad"] = math.asin(dir_z)
    # The forward axis picks up g * sin(pitch), and dir_z is exactly sin(pitch).
    sample["a_long_mps2"] = float(sample["ax_mps2"]) - GRAVITY_MPS2 * dir_z
    return sample
