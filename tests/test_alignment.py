"""Working out which way the car is actually pointing.

A route has to start where the car is and run the way the car faces. Getting
that from the reported orientation means knowing the vehicle's forward axis and
the sign of yaw, neither of which is documented -- and being wrong sends the car
off at an angle from the first metre.

So it is measured instead: roll forward with the wheels straight and see which
way the car goes. Displacement cannot disagree with the frame the positions are
in.
"""

import math

import pytest

from control.alignment import measure_heading, straight_route_from
from sim.backend import ControlInput
from sim.fake import FakeBackend


class AngledBackend(FakeBackend):
    """A car that is not conveniently pointing along +x."""

    def __init__(self, heading_rad, **kwargs):
        super().__init__(**kwargs)
        self.initial_heading = heading_rad

    def reset(self):
        state = super().reset()
        self.state.heading_rad = self.initial_heading
        return self.state.snapshot()


def test_a_car_pointing_along_x_measures_as_zero():
    backend = FakeBackend(dt=0.02)
    backend.reset()
    _, _, heading = measure_heading(backend, dt=0.02)
    assert heading == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("truth", [0.5, 2.309, -1.2, math.pi])
def test_the_measured_heading_matches_where_the_car_points(truth):
    backend = AngledBackend(truth, dt=0.02)
    backend.reset()
    _, _, heading = measure_heading(backend, dt=0.02)
    assert math.cos(heading - truth) > 0.99


def test_the_measurement_reports_where_the_car_ended_up():
    backend = FakeBackend(dt=0.02)
    backend.reset()
    x, y, _ = measure_heading(backend, dt=0.02, min_distance_m=6.0)
    assert x >= 6.0
    assert y == pytest.approx(0.0, abs=0.1)


def test_the_controls_are_released_afterwards():
    backend = FakeBackend(dt=0.02)
    backend.reset()
    measure_heading(backend, dt=0.02)
    assert backend.read_state().throttle == pytest.approx(0.0)


def test_a_car_that_will_not_move_is_reported_rather_than_hanging():
    class Stuck(FakeBackend):
        def apply_control(self, control):
            super().apply_control(ControlInput(0.0, 1.0, control.steering))

    backend = Stuck(dt=0.02)
    backend.reset()
    with pytest.raises(RuntimeError, match="did not move"):
        measure_heading(backend, dt=0.02, timeout_s=3.0)


# -- building the route ---------------------------------------------------


def test_the_route_starts_where_the_car_is():
    route = straight_route_from((100.0, 50.0), heading_rad=0.0, length_m=500.0)
    assert route.point_at(0.0) == pytest.approx((100.0, 50.0))


def test_the_route_runs_the_way_the_car_faces():
    route = straight_route_from((0.0, 0.0), heading_rad=math.pi / 2, length_m=100.0)
    x, y = route.point_at(50.0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(50.0)


def test_the_route_is_the_requested_length():
    route = straight_route_from((0.0, 0.0), heading_rad=1.0, length_m=750.0)
    assert route.length_m == pytest.approx(750.0, rel=0.01)


def test_a_car_pointed_anywhere_tracks_its_own_route():
    # The whole point: an aligned route means no initial heading error at all.
    for truth in (0.5, 2.309, -1.2):
        backend = AngledBackend(truth, dt=0.02)
        backend.reset()
        x, y, heading = measure_heading(backend, dt=0.02)
        route = straight_route_from((x, y), heading, length_m=1000.0)
        assert route.closest_arc_length((x, y)) == pytest.approx(0.0, abs=0.1)
