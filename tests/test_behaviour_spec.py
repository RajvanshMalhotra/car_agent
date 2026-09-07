"""BehaviourSpec is the code gate: bounds are enforced on construction, not in a prompt."""

import pytest

from behaviour.spec import BehaviourSpec, SpecValidationError


def a_valid_spec(**overrides):
    fields = dict(
        name="commuter",
        target_speed_factor=1.0,
        accel_limit_mps2=2.0,
        decel_limit_mps2=3.0,
        jerk_limit_mps3=4.0,
        following_distance_s=1.8,
        corner_speed_factor=0.9,
        reaction_lag_s=0.4,
        erraticness=0.1,
        trip_duration_s=1200.0,
        idle_fraction=0.15,
        hvac_setting=0.5,
        ambient_temp_c=22.0,
        cold_start=True,
        start_stop_enabled=False,
    )
    fields.update(overrides)
    return BehaviourSpec(**fields)


def test_valid_spec_keeps_its_field_values():
    spec = a_valid_spec(idle_fraction=0.45)
    assert spec.idle_fraction == 0.45
    assert spec.name == "commuter"


def test_rejects_braking_harder_than_tyres_allow():
    # An LLM asked for "aggressive" emits sustained 0.9g braking (8.83 m/s^2).
    with pytest.raises(SpecValidationError, match="decel_limit_mps2"):
        a_valid_spec(decel_limit_mps2=8.83)


def test_rejects_negative_idle_fraction():
    with pytest.raises(SpecValidationError, match="idle_fraction"):
        a_valid_spec(idle_fraction=-0.1)


def test_rejects_idle_fraction_above_one():
    with pytest.raises(SpecValidationError, match="idle_fraction"):
        a_valid_spec(idle_fraction=1.5)


def test_error_names_the_field_and_the_allowed_range():
    with pytest.raises(SpecValidationError) as exc:
        a_valid_spec(accel_limit_mps2=99.0)
    message = str(exc.value)
    assert "accel_limit_mps2" in message
    assert "99.0" in message


def test_reports_every_out_of_bounds_field_not_just_the_first():
    with pytest.raises(SpecValidationError) as exc:
        a_valid_spec(accel_limit_mps2=99.0, erraticness=7.0)
    message = str(exc.value)
    assert "accel_limit_mps2" in message
    assert "erraticness" in message


def test_rejects_empty_name():
    with pytest.raises(SpecValidationError, match="name"):
        a_valid_spec(name="")


def test_spec_is_immutable():
    spec = a_valid_spec()
    with pytest.raises(Exception):
        spec.idle_fraction = 0.9


def test_identical_specs_hash_identically():
    assert a_valid_spec().spec_hash == a_valid_spec().spec_hash


def test_differing_specs_hash_differently():
    assert a_valid_spec().spec_hash != a_valid_spec(idle_fraction=0.2).spec_hash


def test_spec_hash_ignores_the_human_readable_name():
    # Reproducibility keys on (spec_hash, scenario, seed); renaming a behaviour
    # must not invalidate a cached run.
    assert a_valid_spec(name="commuter").spec_hash == a_valid_spec(name="courier").spec_hash


def test_round_trips_through_dict():
    spec = a_valid_spec()
    assert BehaviourSpec.from_dict(spec.to_dict()) == spec
