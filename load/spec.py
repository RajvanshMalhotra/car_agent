"""Every constant the battery layer needs and the game does not supply.

BeamNG simulates no 12 V system at all -- no battery, no alternator, no
accessories. So all of this is declared rather than measured, which is exactly
why it is bounds-checked here and content-hashed: a plausible-looking number
outside physical range produces a run that looks fine and means nothing.

Ambient temperature and accessory load are the two fields that also appear on
`collect.ScenarioSpec`, because the drive itself had to assume them. They are
not redeclared as new truth here; `from_sidecar` reads them back off the
trajectory so the value is stated in one place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

#: Frontal area times drag coefficient, m^2. A mid-size saloon.
CD_A_RANGE_M2 = (0.1, 2.0)
#: Rolling resistance coefficient. Tarmac and road tyres sit near 0.012.
CRR_RANGE = (0.001, 0.05)
MASS_RANGE_KG = (500.0, 4000.0)
#: A 12 V SLI battery. Anything outside this is a different application.
CAPACITY_RANGE_AH = (20.0, 200.0)
R0_RANGE_OHM = (0.001, 0.05)
ALTERNATOR_RANGE_A = (40.0, 250.0)
ACCESSORY_RANGE_A = (0.0, 120.0)
AMBIENT_RANGE_C = (-40.0, 60.0)
#: Cranking is hundreds of amps for a second or two. The largest current the
#: battery ever sees, and what the sulfation pathway tracks.
CRANK_RANGE_A = (100.0, 800.0)
CRANK_RANGE_S = (0.2, 10.0)
#: Key-off draw: alarm, clock, ECU keep-alive. Tens of milliamps.
PARASITIC_RANGE_A = (0.0, 1.0)
#: Charge acceptance coefficient, per hour. Around 2 gives ~1.2 A at 99% state
#: of charge on a 60 Ah battery and ~12 A at 90%.
C_ACCEPT_RANGE_PER_H = (0.1, 20.0)
C_TH_RANGE_J_PER_K = (1000.0, 100000.0)
H_RANGE_W_PER_K = (0.1, 50.0)
AIR_DENSITY_RANGE = (0.8, 1.5)

_UNIT_FRACTIONS = ("hvac",)


@dataclass(frozen=True)
class BatteryScenario:
    name: str
    # vehicle
    mass_kg: float = 1510.62  # ETK I-Series node-mass sum, measured over MCP
    cd_a_m2: float = 0.75
    crr: float = 0.012
    air_density: float = 1.225
    # electrical
    alternator_rated_a: float = 120.0
    accessory_base_a: float = 22.0  # ECU, ignition, fuel pump, instruments
    lights: bool = False
    hvac: float = 0.0  # blower and condenser fans, 0 to 1
    parasitic_a: float = 0.03
    crank_a: float = 350.0
    crank_s: float = 1.5
    # battery
    capacity_ah: float = 60.0
    r0_ohm: float = 0.006
    c_th_j_per_k: float = 15000.0
    h_w_per_k: float = 2.0
    c_accept_per_h: float = 2.0
    # environment
    ambient_c: float = 25.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        for field, (low, high) in (
            ("mass_kg", MASS_RANGE_KG),
            ("cd_a_m2", CD_A_RANGE_M2),
            ("crr", CRR_RANGE),
            ("air_density", AIR_DENSITY_RANGE),
            ("alternator_rated_a", ALTERNATOR_RANGE_A),
            ("accessory_base_a", ACCESSORY_RANGE_A),
            ("parasitic_a", PARASITIC_RANGE_A),
            ("crank_a", CRANK_RANGE_A),
            ("crank_s", CRANK_RANGE_S),
            ("capacity_ah", CAPACITY_RANGE_AH),
            ("r0_ohm", R0_RANGE_OHM),
            ("c_th_j_per_k", C_TH_RANGE_J_PER_K),
            ("h_w_per_k", H_RANGE_W_PER_K),
            ("c_accept_per_h", C_ACCEPT_RANGE_PER_H),
            ("ambient_c", AMBIENT_RANGE_C),
        ):
            value = getattr(self, field)
            if not low <= value <= high:
                raise ValueError(f"{field}: {value} outside [{low}, {high}]")
        for field in _UNIT_FRACTIONS:
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field}: {value} outside [0.0, 1.0]")

    @property
    def scenario_hash(self) -> str:
        """Content address. The name labels the scenario, it does not define it."""
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryScenario":
        return cls(**data)

    @classmethod
    def from_sidecar(cls, path: Path | str, name: str, **overrides) -> "BatteryScenario":
        """Take ambient and accessory load from a trajectory's own sidecar.

        The drive already had to assume both. Restating them here would let the
        two drift apart without anything noticing.
        """
        recorded = json.loads(Path(path).read_text()).get("scenario", {})
        fields = {"name": name}
        if "ambient_temp_c" in recorded:
            fields["ambient_c"] = float(recorded["ambient_temp_c"])
        if "accessory_load_a" in recorded:
            fields["accessory_base_a"] = float(recorded["accessory_load_a"])
        fields.update(overrides)
        return cls(**fields)
