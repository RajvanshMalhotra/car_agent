import math

from collect.derive import GRAVITY_MPS2, add_derived


def a_sample(**overrides):
    sample = {"dir_x": 1.0, "dir_y": 0.0, "dir_z": 0.0,
              "ax_mps2": 0.0, "ay_mps2": 0.0, "az_mps2": -GRAVITY_MPS2}
    sample.update(overrides)
    return sample


def test_level_ground_is_zero_grade():
    assert add_derived(a_sample())["grade_rad"] == 0.0


def test_grade_is_the_arcsine_of_the_forward_z_component():
    assert math.isclose(add_derived(a_sample(dir_z=0.5))["grade_rad"], math.asin(0.5))


def test_grade_is_negative_downhill():
    assert add_derived(a_sample(dir_z=-0.1))["grade_rad"] < 0


def test_a_dir_z_outside_the_domain_is_clamped_not_crashed():
    # Numerical drift can put a unit vector marginally over 1.
    assert math.isclose(add_derived(a_sample(dir_z=1.0000001))["grade_rad"],
                        math.pi / 2)


def test_a_parked_car_on_the_level_has_no_longitudinal_acceleration():
    assert math.isclose(add_derived(a_sample())["a_long_mps2"], 0.0, abs_tol=1e-9)


def test_a_parked_car_on_a_slope_still_has_none():
    # The whole reason this channel is derived: on a 30 degree slope the
    # accelerometer reads 4.9 m/s2 that is entirely gravity.
    grade = math.radians(30)
    sample = a_sample(dir_z=math.sin(grade),
                      ax_mps2=GRAVITY_MPS2 * math.sin(grade),
                      az_mps2=-GRAVITY_MPS2 * math.cos(grade))
    assert math.isclose(add_derived(sample)["a_long_mps2"], 0.0, abs_tol=1e-9)


def test_real_acceleration_on_the_level_passes_through():
    assert math.isclose(add_derived(a_sample(ax_mps2=2.0))["a_long_mps2"], 2.0)


def test_the_measured_channels_are_left_alone():
    sample = add_derived(a_sample(dir_z=0.5, ax_mps2=2.0))
    assert sample["dir_z"] == 0.5 and sample["ax_mps2"] == 2.0
