# Stages 3-5: the load model, the cell model, and the synthetic dataset

**Date:** 2026-09-11
**Status:** approved design, not yet implemented
**Parent spec:** `docs/superpowers/specs/2026-09-09-battery-rul-pipeline-design.md`
**Implements:** parent sections 6 (stage 3), 7 (stage 4), 8 (stage 5)

## 1. What this is

The parent spec draws seven stages. Stage 1 (`collect/`) is built. This spec
covers the three that turn a driving trajectory into a battery dataset:

```
trajectory CSV  ->  load/   ->  I_bat, T_bay
                       |
                    cell/   ->  SoC, V_bat, T_bat, R_int, SOH, RUL
                       |
                 two tables + sidecar  ->  world model, then NN
```

Out of scope: the world model (stage 2) and the networks (stages 6-7). Each
gets its own spec.

## 2. The input we actually have

`telemetry.csv`, 127 MB, recorded from BeamNG over OutGauge and MotionSim
before `collect/` existed. Profiled directly:

- 181,099 rows, 1,929.9 s, ~94 Hz, no timestamp gaps over 1 s
- **one trip then one soak**: engine on for 1,192 s, off for 738 s
- **zero cranks observed** -- the engine is already running at t=0
- 72.8% of rows stationary (< 0.5 m/s)
- coolant (`engine_temp`) mean 101 C, max 130 C; `oil_temp` max 212 C

Three columns are unusable and must be recorded as such rather than silently
consumed:

| column | state | cause |
|---|---|---|
| `brake` | 32-byte blob | the recorder captured OutGauge's `display1` field |
| `oil_pressure`, `clutch`, `game_time` | constant zero | never populated |

It is the 34-column OutGauge shape, not `collect/schema.py`'s 48-column
canonical trajectory. Missing entirely: `mass_kg`, `engine_load`,
`engine_torque_nm`, `ignition_level`, `engine_running`, the four wheel speeds,
the four brake temperatures, `downforce_fl`.

Present and useful: `speed`, `rpm`, `engine_temp`, `oil_temp`, `throttle`,
`gear`, `fuel`, and full pose -- so road grade is recoverable from `pitch`, and
`acc_x/y/z` gives the vibration proxy.

The file lives in `runs/`, which is gitignored. Nothing in this spec commits it.

## 3. Module boundaries

Two packages. `battery/corrosion.py` and `battery/electrical.py` **move** here
and are extended, per parent section 11. The bay-temperature model moves out of
`sim/engine.py`.

```
load/                                    cell/
  scenario.py   ScenarioSpec               ecm.py        V_oc, R_int, V_term
  legacy.py     OutGauge CSV -> canonical  aging.py      corrosion, sulfation, shedding
  trips.py      trip and crank segments    integrate.py  the 1 Hz within-trip loop
  mechanical.py F_total, P_mech            life.py       trip damage -> SOH(t) -> RUL
  thermal.py    T_bay                      dataset.py    stage-5 assembly
  electrical.py I_acc, I_alt, I_bat        leakage.py    the section 8 leakage report
```

`load/` never imports `cell/`. `cell/` reads no CSV except through
`dataset.py`. Stdlib only -- the parent plan's global constraint stands, and
nothing here needs more.

## 4. The legacy adapter and its honesty rules

`collect/schema.py` defines exactly two provenance values and stage 1's tests
pin that. This spec does **not** change it. Instead `load/legacy.py` emits the
canonical 48 columns together with its own sidecar, whose channel table uses a
wider vocabulary:

| provenance | meaning | columns in this file |
|---|---|---|
| `measured` | straight from the game | `speed_mps rpm coolant_c oil_c throttle gear_index fuel`, full pose |
| `derived` | computed from measurements | `grade_rad` (from `pitch`), `a_long_mps2`, `engine_running` (rpm > 50) |
| `assumed` | filled from a constant we chose | `mass_kg` = 1510.62 (ETK I-Series node-mass sum), `engine_load` (throttle x rpm proxy) |
| `absent` | not recoverable from this file | `brake`, `clutch`, `oil_pressure`, the wheel speeds, the brake temperatures, `downforce_fl`, `engine_torque_nm`, `ignition_level` |

Two rules, both enforced in code rather than by comment:

1. **A computation that reads an `absent` column raises.** No silent zero.
2. **A headline number whose dependency graph includes an `assumed` column
   carries a flag saying so.** "Absolute bay temperature is indicative only"
   stops being a sentence in a document and becomes an attribute on the value.

`brake` being absent costs nothing: parent section 2 already drops hard braking
as a counterfactual axis.

## 5. Stage 3 -- the load model

### 5.1 Mechanical

    F_total = m a_long + m g sin(theta) + 0.5 rho Cd A v^2 + Crr m g cos(theta)
    P_mech  = F_total v

`Cd A`, `Crr`, `rho` are per-vehicle constants in `ScenarioSpec`. Where
`engine_torque_nm` and `engine_load` are measured they give an independent
check; in this file both are absent or assumed, so the check is deferred to
canonical data.

### 5.2 Thermal, and the heat-soak correction

    coupling     = C_static (1 + C_load load) / (1 + C_airflow v)
    T_bay_target = T_ambient + coupling (T_coolant - T_ambient)
    dT_bay/dt    = (T_bay_target - T_bay) / tau_bay

Measured coolant is used verbatim. The model never integrates over a
measurement.

**Correction to `sim/engine.py`.** It sets `coupling = 0.0` the instant the
engine stops. That is wrong in the direction that matters most. At shutdown
airflow stops while the block is still near 100 C, so bay temperature *rises*
for several minutes before decaying -- the well-known heat soak, and the
hottest the battery ever gets. Corrosion is Arrhenius, so this peak is
disproportionately damaging. The recorded file is precisely this case: a 738 s
soak immediately after 1,192 s of driving with coolant at ~101 C.

`load/thermal.py` replaces the hard zero with a raised no-airflow coupling
`C_soak > C_static`, which then decays as the coolant does.

### 5.3 Electrical

**Sign convention, correcting the parent spec.** Parent section 6 writes
engine-off as `I_bat = -I_parasitic`, while section 7 writes
`dSoC/dt = -I_bat / Q` and `battery/electrical.py` documents positive as
discharge. These disagree. This spec standardises on **positive = discharge**
throughout. Engine-off parasitic is `+0.03 A`; cranking is `+350 A`. The parent
spec is corrected, not worked around.

    engine off      I_bat = +I_parasitic                     ~30 mA
    cranking        I_bat = +I_crank                         350 A for 1.5 s
    engine running  I_bat = I_acc - I_alt_out

with

    I_accept  = C_accept Q (1 - SoC) f_T(T)
    I_alt_out = min( alt_capability(rpm), I_acc + I_accept )

`C_accept` around 2 /h gives ~1.2 A acceptance at 99% SoC on a 60 Ah battery
and ~12 A at 90% -- the right shape. `f_T` is the temperature multiplier on
acceptance: a cold battery accepts charge poorly, so `f_T` falls below 1.0
under about 10 C and saturates at 1.0 once warm. It is not the same function
as `f_temp` in section 6.1, which acts on resistance.

**Charge acceptance is not optional.** Without the `(1 - SoC)` taper the
alternator refills the battery instantly after every crank, and both the
recharge-deficit and sulfation pathways disappear from the model entirely.

`I_accessories` is a scenario input, not a measurement: BeamNG simulates no
accessories and no 12 V system at all. Declared per run, recorded in the
sidecar, 20-60 A total.

### 5.4 Trips and cranks

`load/trips.py` segments the trajectory on `engine_running`, and emits a crank
event at each off-to-on transition. On canonical data this reads
`ignition_level` and `engine_running`; on this file `engine_running` is derived
from `rpm > 50`, which yields one trip, one soak, and zero cranks.

Zero observed cranks is why `ScenarioSpec` carries a trip *schedule* -- see
section 7.

## 6. Stage 4 -- the cell model

### 6.1 Fast states, 1 Hz

    dSoC/dt    = -I_bat / Q(SOH)
    V_bat      = V_oc(SoC) - I_bat R_int            (battery carrying the load)
    V_bat      = V_reg - sag(I_bat)                 (alternator carrying it)
    C_th dT/dt = I_bat^2 R_int - h (T_bat - T_bay)
    R_int      = R_0 f_soh(SOH) f_temp(T) f_soc(SoC)

`h` is the battery-to-bay thermal conductance, in W/K. The three `f_*` are
dimensionless multipliers on internal resistance, each 1.0 at the reference
condition: resistance rises as the battery ages, as it gets cold, and as it
discharges.

`T_bay`, not outside air: the battery sits in a 60-100 C engine bay.

Self-heating was checked against the currents this battery actually sees. At
20 A through 6 mOhm it is 2.4 W; even a 350 A crank is 735 W for 1.5 s into a
~15 kJ/K mass, which is 0.07 K. So `h (T_bat - T_bay)` is essentially the
entire thermal model, exactly as the parent spec states. The `I^2 R` term is
kept because it costs nothing and becomes visible as `R_int` rises with age.

**`R_int` feeds back into `I_bat`.** As SOH falls, `R_int` rises, cranking sags
harder and dissipates more heat. That feedback is what bends the fade curve;
without it the model ages linearly and says nothing interesting.

### 6.2 Aging states, per trip

Schiffer et al. (2007), weighted Ah throughput, with corrosion and sulfation as
**states rather than summary statistics** -- both rates depend on their own
accumulated value, so a summary statistic cannot express them.

- **corrosion**: layer thickness, Arrhenius in `T_bat`. `battery/corrosion.py`
  supplies the temperature dependence, already anchored to the +10 C halves
  life rule and landing at ~62 kJ/mol.
- **sulfation**: crystal size, growing with time at low SoC and with cranks not
  followed by a full recharge, shrinking on full charge.
- **shedding**: vibration dose from `ax ay az`.

SOH is the combination; end of life is SOH = 0.8.

### 6.3 Two rates, because a battery life is 10^8 seconds

Within a trip, SOH is held fixed and the fast states integrate at 1 Hz --
roughly 2,000 steps for the recorded file. At trip end, `cell/integrate.py`
emits a damage vector: corrosion equivalent-hours, SoC-weighted Ah throughput,
time at low SoC, vibration dose.

`cell/life.py` then advances the aging states over a declared trip schedule,
recomputing the per-trip damage whenever SOH has drifted past a threshold --
necessary because `R_int` rises, which changes both current and temperature.

The alternative, integrating everything at 1 Hz to end of life, is 10^8 steps
and is not reachable in stdlib Python. Over 1,930 s SOH moves by about 10^-7,
so the single-rate loop would spend every cycle recomputing a constant.

### 6.4 The scale gate

Parent section 10 and open question 4: no aging rate constant is fitted,
because no battery in this project has reached end of life and the open
full-life dataset has not been obtained.

Therefore `RUL_days` is emitted with `scale_unfitted: true` in the sidecar
whenever the rate constant is the unfitted default, which is the only case
available today. `cell/life.py` returns an absolute figure with the flag
cleared only once a constant fitted against full-life data is supplied.
Relative claims are unaffected either way: they are ratios, and the unfitted
constant cancels.

A parameter ensemble over declared rate-constant ranges supplies the interval
required by the parent's output contract. It represents **parameter
uncertainty, not the world model's stochastic latent**, and the sidecar says so
in those words.

## 7. ScenarioSpec

Frozen dataclass, bounds-checked on construction, content-hashed for
reproducibility. Replaces `BehaviourSpec`; no LLM, no cache, no generation.

    vehicle      mass_kg, cd_a, crr, alternator_rated_a
    battery      capacity_ah, r0_ohm, c_th_j_per_k, h_w_per_k, c_accept_per_h
    accessories  base_a, lights, hvac
    environment  ambient_c
    trips        an ordered list of (trajectory, soak_s) with a crank at each start
    seed

The `trips` field is what gives the sulfation and parasitic pathways any data
at all. The recorded file is one trip with zero cranks; a schedule of that trip
repeated with declared soak durations produces crank events, SoC recovery
during driving and decay during soak.

**Soak durations are declared, never measured.** The sidecar records them as
scenario inputs. No trip boundary is fabricated inside a recorded trace.

## 8. Stage 5 -- the dataset, in two tables

| table | rate | columns | trains |
|---|---|---|---|
| `within_trip.csv` | 1 Hz | canonical trajectory + `I_bat V_bat T_bat SoC R_int` | SoC and voltage models |
| `trip_damage.csv` | per trip | scenario + damage vector + `SOH RUL_days` | the RUL model |

**Why two and not one.** Flattening `RUL_days` onto every 1 Hz row broadcasts a
per-trip constant across ~2,000 near-identical rows, where a network recovers
it from the row index. That is the `True_SoC` trap in new clothes. Separating
the tables is what makes the leakage report meaningful rather than decorative.

### The leakage report

Parent section 8 requires the generator to emit a report listing any column
pair whose relationship it can recover analytically: exact linear and ratio
relationships, and cumulative-sum identities.

**The load-bearing test is on the detector itself.** It is fed the real
dataset's known identity,

    True_SoC = 100 (1 - cumsum(current dt) / 16.069411)

and must *catch* it. A leakage detector nobody has ever seen fire is not
evidence of anything. Only then is our generated dataset asserted to pass.

## 9. Testing

Stdlib only, no network, runs on macOS with no BeamNG and no Windows machine --
the parent plan's global constraints stand unchanged.

- per-module unit tests against hand-computed values
- property tests: SoC monotone under pure discharge; corrosion rate doubles per
  +10 C (`test_corrosion.py` already covers this); charge acceptance goes to
  zero at SoC = 1
- adapter tests: an `absent` column raises on read; an `assumed` column
  propagates its flag to a headline number
- the leakage-detector regression test of section 8
- one end-to-end pass over the recorded 181k-row file asserting physical bounds
  (SoC in [0, 1], V in [10, 15], `T_bay > T_ambient` while the engine runs) and
  that idling with HVAC on produces an actual deficit

## 10. What this spec changes in the parent

1. **Sign convention** standardised to positive = discharge (section 5.3).
   Parent section 6 is internally inconsistent with parent section 7.
2. **Bay coupling at engine-off** is no longer zero (section 5.2). The parent
   inherited `sim/engine.py`'s hard zero, which discards the heat-soak peak --
   the hottest the battery ever gets, in an Arrhenius process.
3. **Charge acceptance** is promoted from a clause to a required term
   (section 5.3), because without it two of the five aging pathways do not
   exist in the model.
4. **The dataset is two tables, not one** (section 8).

## 11. Open questions inherited, not resolved

Parent open questions 1 (ambient not writable in BeamNG), 3 (no real current
clamp measurements) and 4 (full-life dataset not obtained) all bear directly on
this spec and none are resolved by it. Question 4 is the binding one: it sets
the absolute scale, and section 6.4 is the mechanism that stops us claiming a
number we cannot support.
