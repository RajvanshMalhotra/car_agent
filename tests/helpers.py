"""Shared test fixtures."""

from behaviour.spec import BehaviourSpec


def a_behaviour_spec(**overrides):
    fields = dict(
        name="test",
        target_speed_factor=1.0,
        accel_limit_mps2=2.0,
        decel_limit_mps2=3.0,
        jerk_limit_mps3=4.0,
        following_distance_s=1.8,
        corner_speed_factor=0.9,
        reaction_lag_s=0.0,
        erraticness=0.0,
        trip_duration_s=1200.0,
        idle_fraction=0.0,
        hvac_setting=0.5,
        ambient_temp_c=22.0,
        cold_start=True,
        start_stop_enabled=False,
    )
    fields.update(overrides)
    return BehaviourSpec(**fields)
