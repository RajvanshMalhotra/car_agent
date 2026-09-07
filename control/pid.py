"""Textbook PID with anti-windup, emitting a normalised signed effort."""

from __future__ import annotations


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = 10.0,
        output_limit: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        """Return effort in [-output_limit, +output_limit]."""
        if dt > 0.0:
            self.integral += error * dt
            self.integral = min(
                self.integral_limit, max(-self.integral_limit, self.integral)
            )
            derivative = (
                0.0
                if self.previous_error is None
                else (error - self.previous_error) / dt
            )
            self.previous_error = error
        else:
            derivative = 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return min(self.output_limit, max(-self.output_limit, output))
