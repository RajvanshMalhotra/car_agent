"""Sensor noise: make the emitted rows look like a real logger, not an ODE.

`cell/integrate.py` and `load/` produce exact numbers -- a deterministic
physics model with no measurement in the loop. A real vehicle never reports
those numbers exactly: every channel a logger reads passes through an ADC,
a thermistor, a shunt, or a bus that quantises and drifts. This module adds
that layer back in, on emission only, so a model trained on the dataset sees
the kind of noise a real deployment would actually hand it.

**The central design rule.** Noise goes ONLY on channels a real car can
measure. Everything else is latent ground truth and must stay exactly clean:

    MEASURABLE -- a logger or the vehicle bus reads these directly:
        speed_mps, rpm, throttle, coolant_c, i_bat_a, v_bat_v, t_bat_c
    LATENT -- nothing on a real car measures these:
        soc, r_int_ohm, t_bay_c, soh, and every damage/aging field

`t_bay_c` is latent on purpose: the project's whole premise (see CLAUDE.md) is
that under-bonnet temperature is NOT measured and must be estimated, so
noising it would misrepresent what the estimator itself sees. `soc` is the
label a downstream model is meant to predict; noising the label would just be
noising the answer key.

This is a measurement-noise layer, not a data-generation trick: it does not
make the underlying process any less deterministic. The physics is still an
exact function of its inputs, and a large enough network trained on enough
noisy rows can still recover it -- averaging over repeated noisy observations
is exactly what a network does. What changes is realism, not information
content.
"""

from __future__ import annotations

import random

#: Channels a logger or the vehicle bus can actually read. Noise applies here,
#: and only here.
MEASURABLE: tuple[str, ...] = (
    "speed_mps", "rpm", "throttle", "coolant_c", "i_bat_a", "v_bat_v", "t_bat_c",
)

#: Ground truth with no real-world sensor. Must stay exactly clean; noising
#: these would corrupt training labels rather than simulate a measurement.
LATENT: tuple[str, ...] = ("soc", "r_int_ohm", "t_bay_c", "soh")

#: Per-channel noise spec. `sigma` is either a constant (in the channel's
#: units) or a callable of the clean reading, for noise that scales with the
#: signal. `clamp` is an optional (low, high) physical bound applied after
#: noise -- `None` in a slot means unbounded on that side.
NOISE: dict[str, dict] = {
    #: A good 12-bit ADC over 0-16 V is ~4 mV/count; real automotive voltage
    #: sensing is quoted around +/-0.5% accuracy, which is a similar order.
    "v_bat_v": {"sigma": 0.010, "clamp": (None, None)},
    #: Shunt or hall-effect current sensors are typically quoted ~0.5% of
    #: reading, with a floor near zero set by offset drift rather than the
    #: reading itself -- a 0 A true current does not read exactly 0 A.
    "i_bat_a": {"sigma": lambda x: max(0.1, 0.005 * abs(x)), "clamp": (None, None)},
    #: NTC thermistor on the battery case: self-heating and mounting/placement
    #: error dominate over the ADC quantisation.
    "t_bat_c": {"sigma": 0.5, "clamp": (None, None)},
    #: The coolant-temp sender is a cheap NTC read through a coarse ECU gauge
    #: circuit -- looser than a dedicated battery-temp sensor.
    "coolant_c": {"sigma": 1.0, "clamp": (None, None)},
    #: Wheel-speed-derived, quantised as it crosses the vehicle bus.
    "speed_mps": {"sigma": 0.05, "clamp": (0.0, None)},
    #: Crank position sensor: one of the most accurate signals on the bus.
    "rpm": {"sigma": 5.0, "clamp": (0.0, None)},
    #: Pedal position sensor, roughly 0.5% of full scale (0-1).
    "throttle": {"sigma": 0.005, "clamp": (0.0, 1.0)},
}


def _sigma_for(channel: str, value: float) -> float:
    spec_sigma = NOISE[channel]["sigma"]
    return spec_sigma(value) if callable(spec_sigma) else float(spec_sigma)


class SensorNoise:
    """Applies `NOISE` to the measurable channels of one row at a time.

    Deterministic given `(seed, scale, enabled)` and the sequence of `apply`
    calls made against it -- construct once per run and call `apply` per row,
    the way a real sensor's noise process advances one sample at a time.
    """

    def __init__(self, seed: int = 0, enabled: bool = True, scale: float = 1.0):
        self.seed = seed
        self.enabled = enabled and scale != 0.0
        self.scale = scale
        self._rng = random.Random(seed)

    def apply(self, row: dict) -> dict:
        """Return a NEW dict: `row` with noise added to measurable channels.

        Latent channels (and anything not in `NOISE` at all, such as
        `engine_load` or `grade_rad`) pass through completely untouched.
        """
        out = dict(row)
        if not self.enabled:
            return out
        for channel in MEASURABLE:
            if channel not in out:
                continue
            clean = float(out[channel])
            sigma = self.scale * _sigma_for(channel, clean)
            noisy = clean + self._rng.gauss(0.0, sigma) if sigma > 0.0 else clean
            low, high = NOISE[channel]["clamp"]
            if low is not None:
                noisy = max(low, noisy)
            if high is not None:
                noisy = min(high, noisy)
            out[channel] = noisy
        return out

    def manifest(self) -> dict:
        """What was applied, for the dataset sidecar."""
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "scale": self.scale,
            "measurable_channels": list(MEASURABLE),
            "latent_channels": list(LATENT),
            "sigmas": {
                channel: (
                    "0.005 * |reading|, floor 0.1"
                    if callable(NOISE[channel]["sigma"])
                    else NOISE[channel]["sigma"] * self.scale
                )
                for channel in MEASURABLE
            },
        }
