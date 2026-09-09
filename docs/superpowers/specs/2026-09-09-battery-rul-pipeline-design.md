# Driving simulation to lead-acid RUL: pipeline design

**Date:** 2026-09-09
**Status:** approved design, not yet implemented
**Supersedes:** the behaviour-agent framing in `CLAUDE.md`

## 1. What this is

A seven-stage pipeline that turns simulated driving into a lead-acid battery
dataset, and uses that dataset to train and compare a baseline neural network
against a physics-informed one.

```
BeamNG.drive  ->  world model  ->  counterfactual driving futures
                                             |
                                    vehicle load model
                                             |
                                   battery physics model
                                             |
                                   synthetic battery dataset
                                             |
                                   baseline NN  |  PINN
                                             |
                              unseen scenarios + real measurements
```

The battery is a **12 V SLI lead-acid battery in a combustion car**. Not an EV
traction pack, not lithium. That single fact determines most of what follows.

### Scope

In scope: stages 1 through 7 as drawn. Out of scope: the LLM behaviour agent,
the campaign sampler, and the hand-written pure-pursuit controller, all of
which are removed (section 11).

## 2. The one physical fact everything hangs on

In a combustion car the battery supplies **none** of the traction power. The
engine burns fuel for that. The battery cranks the engine -- 200-600 A for
1-3 s -- and the alternator then carries the accessory load and recharges it.
While driving, battery current is small and usually *negative*.

The consequence is that the naive chain

    P_elec = P_mech / eta + P_accessories ;  I_bat = P_elec / V_bat

is wrong here. It is the EV formula. Applied to 20 m/s, +2 m/s^2, 1500 kg and a
3 degree grade it yields `I_bat` around 8000 A sustained, which is three orders
of magnitude out.

Mechanical power still matters enormously, but through a different pathway:
engine load heats the engine bay, and bay temperature drives **grid corrosion**,
which is the dominant aging mechanism for this battery. So `P_mech` is computed,
and routed to the thermal model rather than to battery current.

### Aging pathways, in order

1. **Grid corrosion** -- Arrhenius in under-bonnet temperature. Dominant.
2. **Sulfation** -- repeated cranking without a full recharge; short trips.
3. **Recharge deficit** -- idling with accessories on, where alternator output
   at low RPM does not cover the electrical load.
4. **Plate shedding** -- vibration.
5. **Cycle count** -- only with start-stop, which is a configurable flag and
   defaults off.

### Consequence for the counterfactual axes

The axes in the original sketch do not all survive:

| Axis | Verdict |
|---|---|
| Uphill / sustained load | keep -- high engine load, hot bay, real corrosion signal |
| Aggressive acceleration | weak -- reaches the battery only as bay heat |
| Hard braking | drop -- essentially no effect on an SLI battery |

Replaced by the axes that actually move this battery: **hot ambient**,
**sustained high load**, **idle-heavy stop-go with accessories on**, and
**short-trip cycling**.

The last one has a structural implication: the world model must be able to roll
out **trip patterns**, not only seconds of continuous driving. One 40-minute
drive and eight 5-minute drives cover the same distance and are entirely
different for the battery.

## 3. Architecture and interfaces

Five modules, four interfaces. Each interface is a plain tabular contract so
that any stage can be replaced without touching its neighbours.

| Module | Stages | Consumes | Produces |
|---|---|---|---|
| `collect/` | 1 | BeamNG over MCP | trajectory CSV + provenance sidecar |
| `worldmodel/` | 2 | trajectory CSV | rollouts in the same schema |
| `load/` | 3 | trajectory schema | `I_bat`, `T_bay` series |
| `cell/` | 4, 5 | `I_bat`, `T_bay` | `SoC`, `V`, `T`, `SOH`, `R_int`, and the assembled dataset |
| `models/` | 6, 7 | synthetic dataset | trained NN and PINN, plus evaluation |

Two rules carried over from the existing codebase because they earned their
place:

- **The simulator boundary is one interface wide.** Only `sim/` imports
  anything BeamNG-specific. A `beamngpy` backend must be a drop-in if the
  BeamNG.tech licence ever lands.
- **A fake backend is core infrastructure, not a nicety.** Development is on
  macOS; the game runs on a Windows laptop belonging to a collaborator.
  Anything untestable without that laptop is a design smell.

## 4. Stage 1 -- trajectory collection

### What the game can and cannot give

Probed directly against BeamNG 0.39's first-party MCP server (86 tools, no
BeamNG.tech licence, no mod). Findings recorded here so they are not
re-litigated:

- `run_lua_vehicle` executes arbitrary Lua **inside the vehicle physics VM**,
  which is where the full state lives. `get_electrics` exposes a curated slice
  of it and is not the ceiling.
- That tool is **asynchronous**: it answers "queued in vehicle VM(s)" and
  delivers the result on a later call, keyed by vehicle id, with nothing
  identifying which request it answers. Results must be tagged and drained.
- `enablePhysicsStepHook` and `onPhysicsStep` exist in the vehicle VM, so Lua
  can accumulate samples at physics rate. `jsonEncode` is available.
- `obj:getTotalMass()` does **not** exist. Mass comes from summing
  `obj:getNodeMass(i)` over `obj:getNodeCount()` -- 783 nodes, 1510.62 kg on
  the ETK I-Series.
- Ambient temperature is **readable** (`core_environment.getState().temperatureC`
  = 25) and **not writable** by that route: `setState` accepts the call and
  changes nothing. See open question 12.1.
- There is **no 12 V electrical system in BeamNG**. The ~200 keys of
  `electrics.values` contain nothing matching volt, batt, amp, current,
  alternator or charge. The only electrical signal is `electricalLoadCoef`.
  Battery current and voltage are therefore modelled, in stage 3, always.

### Design

Replace the current 5 Hz HTTP polling loop with a Lua-side sampler:

- a sampler installed via `enablePhysicsStepHook` appends a packed record to a
  ring buffer at 100 Hz;
- Python calls `run_lua_vehicle` once a second to drain it;
- the drain is tagged and idempotent, so a dropped call loses nothing.

This yields more data at fewer HTTP round trips than the current loop, and it
makes longitudinal acceleration and road grade real measurements rather than
finite differences of a 1 Hz speed signal.

BeamNG's own AI drives (`set_ai`, `drive_to`). It stays on the road and avoids
traffic, which nothing we write does. Driving style is the single `aggression`
scalar the AI accepts; that coarseness is accepted deliberately, because
diversity now comes from the world model's counterfactuals rather than from the
recorded drives.

### Schema

Measured, straight from the game:

    t_s x_m y_m z_m vx vy vz speed_mps
    ax ay az                          sensors.gx/gy/gz, includes gravity
    dir_x dir_y dir_z                 obj:getDirectionVector()
    mass_kg                           node-mass sum, live
    rpm engine_torque_nm engine_av engine_load exhaust_flow
    throttle brake steering gear_index clutch_ratio
    wheel_av_fl wheel_av_fr wheel_av_rl wheel_av_rr
    brake_temp_fl brake_temp_fr brake_temp_rl brake_temp_rr
    coolant_c oil_c fuel_volume_l odometer_m trip_m
    ignition_level engine_running altitude_m damage

Derived inside stage 1, and labelled as such:

    grade_rad = asin(dir_z)
    a_long                            gravity removed using dir

Recorded once per run, not per sample: BeamNG build, level, vehicle jbeam and
config, ambient temperature as read from the game, accessory-load scenario,
`aggression`, git commit, sampler rate.

**Every column carries a `measured` or `derived` flag in the sidecar.** This is
not documentation. Downstream code reads it, and no plot or paper may present a
derived column as a measurement.

Battery-facing consumers downsample to 1 Hz. The 100 Hz record is kept because
the world model needs it and it cannot be recovered later.

## 5. Stage 2 -- the world model

**Its job is counterfactual query, not data generation.** It cannot add
information the simulator lacks, and it is slower than the ODE it would be
surrogating, so speed is not the justification. It earns its place by answering
per-vehicle counterfactuals -- "what would this battery have done had this
driver made fewer short trips" -- which requires Pearl's
abduction -> action -> prediction. Plain action-conditioned rollout yields an
interventional average, not a counterfactual.

Learns: trajectory history -> next state, over the stage 1 schema, with a
stochastic latent so rollouts carry a distribution rather than a point.

Must roll out at two timescales: within-trip dynamics at 10-100 Hz, and
trip patterns (start, duration, soak, restart) at trip granularity. The second
is what the sulfation pathway needs and is easy to forget.

Reproducibility: runs are identified by `(scenario_hash, seed)`. BeamNG's
`get_simulation_state` reports a `deterministic` flag; whether it delivers
reproducible physics is open question 12.2.

## 6. Stage 3 -- the vehicle load model

Two paths out of the trajectory, both feeding stage 4.

### Mechanical, then thermal

    F_total = m a + m g sin(theta) + 0.5 rho Cd A v^2 + Crr m g
    P_mech  = F_total v

`m`, `a`, `theta` and `v` are measured. `Cd A`, `Crr` and `rho` are constants
per vehicle configuration, fitted where possible against BeamNG's own
`engine_torque_nm` and `engine_load`, which give an independent handle on
engine output.

    T_bay = f(coolant_c, engine_load, airflow(v), T_ambient)

The structure is defensible; the coefficients are not calibrated. Ratios
between conditions are the usable output. **Absolute bay temperatures are
indicative only, and must be reported as such.**

On a real car the measured coolant temperature is used verbatim. The model
never integrates over a measurement.

### Electrical

    engine off      I_bat = -I_parasitic                     ~30 mA
    cranking        I_bat = -I_crank                         200-600 A, 1-3 s
    engine running  I_bat = I_accessories - I_alternator(rpm)

`I_alternator` is an RPM-dependent output curve, regulator-limited to a ~14.4 V
setpoint and further limited by the battery's charge acceptance, which itself
depends on SoC and temperature. Near idle it may not cover the load, which is
the recharge-deficit pathway.

`I_accessories` is a **scenario input**, not a measurement -- BeamNG simulates
no accessories. Headlights, blower, AC, ECU, fuel pump: 20-60 A total. It is
declared per run and recorded in the sidecar.

Crank events are detected from `ignition_level`, `engine_running` and the RPM
rise, and they define trip boundaries.

## 7. Stage 4 -- the battery physics model

Equivalent circuit, integrated at 1 Hz:

    dSoC/dt   = -I_bat / Q
    V_bat     = V_oc(SoC) - I_bat R_int
    C_th dT/dt = I_bat^2 R_int - h (T - T_bay)
    Q         = Q_0 (SOH)
    R_int     = R_0 (SOH, T, SoC)

**`T_bay`, not outside air.** For a battery under the bonnet the ambient is the
engine bay at 60-100 degrees C. At the currents an SLI battery actually sees,
self-heating is negligible and `h (T - T_bay)` is essentially the whole thermal
model. Getting that term wrong leaves nothing.

Aging follows **Schiffer et al. (2007)**, the weighted-Ah-throughput model for
lead-acid, with explicit corrosion and sulfation terms. Not the Wang power law,
which is lithium and does not apply here.

- corrosion: Arrhenius in battery temperature, which tracks `T_bay`
- sulfation: accumulated time at low SoC, and cranks not followed by full
  recharge
- shedding: vibration proxy from `ax ay az` and per-wheel `downForce`

## 8. Stage 5 -- the synthetic dataset

Columns: the stage 1 trajectory schema, plus `I_bat`, `V_bat`, `T_bat`, `SoC`,
`SOH`, `R_int`, plus scenario metadata.

**A hard lesson is encoded here.** The existing real dataset has

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 (1 - total_ah_used / 16.069411)

exact on every row. Its reported ~99.7% model accuracy is label leakage -- the
model recovers a division. The synthetic dataset must not repeat this. No
target may be a closed-form function of the features presented alongside it,
and the generator emits a leakage report listing any column pair whose
relationship it can recover analytically.

## 9. Stage 6 -- baseline NN versus PINN

Both learn: driving history -> battery state. The PINN additionally penalises
violation of `dSoC/dt = -I/Q`.

**The PINN physics loss must use time-varying capacity `Q(SOH)`, not nominal.**
With nominal capacity a correct model carries a systematic residual that grows
as the cell ages, and the physics term then fights the data term.

The honest claim available from this comparison is narrow: both are trained on
data generated by the stage 4 ODE, so agreement with that ODE demonstrates
fitting, not physical correctness. What the comparison *can* show is sample
efficiency, extrapolation behaviour outside the training envelope, and
robustness under distribution shift.

## 10. Stage 7 -- evaluation, and what may be claimed

The endpoint is unobservable. No battery in this project has reached end of
life, so absolute RUL accuracy cannot be checked. The parts are validated
separately:

- **aging law** -- against an open full-life flooded lead-acid dataset, noting
  flooded-versus-AGM and grid-versus-vehicle regime differences before
  assuming transfer;
- **load model** -- against the existing bulb data, which is one operating
  point: a constant ~3.0 A resistive load at 31 degrees C. It pins a constant
  term and identifies nothing about load dependence;
- plus held-out tails, rank-ordering of load severities, and a sanity band from
  the literature: SLI batteries last 3-5 years, 2-3 in hot climates. A model
  predicting 12 years is broken.

### Output contract

Never a bare point estimate. Every RUL prediction is a calibrated conservative
bound, and the headline figure is the safe one:

    847 days  (90% interval: 620 - 1150)  ->  plan for 620

**What is validated is calibration, not point accuracy.** If a 90% interval is
claimed, truth must land inside it about 90% of the time on held-out data.
That is checkable with modest data. Point accuracy at end of life is not.

Three relative claims carry less validation burden and are often more useful:

- profile A ages the battery roughly 2x faster than profile B
- this battery is aging faster than expected for its usage
- fewer short trips would have extended it by roughly X%

### Known-thin argument, stated rather than hidden

The NN and PINN are trained on data generated by our own ODE, so the loop is
partly circular. Masked latents learned from the real bulb data are the only
thing breaking it, and that data is early-life, two-channel and single
-condition. Circularity is **reduced, not eliminated**. The claim is stated at
that strength.

Absolute scale is the weak link: physics gives the fade curve's shape, but the
rate constant must be fitted to cells that actually reached end of life. No
dead batteries are available, so the open full-life dataset is load-bearing,
not optional.

Expect wide intervals and say so plainly.

## 11. What is removed

Deleted, with the reasoning recorded so it is not rebuilt by accident:

| Path | Why |
|---|---|
| `behaviour/`, `agent.py` | LLM behaviour generation. Diversity now comes from stage 2 counterfactuals. The game's AI takes one `aggression` scalar regardless, so eight generated parameters were collapsed to one on the way in. |
| `campaign/`, `collect.py` | Campaign sampling over a behaviour x ambient grid. The ambient axis was never simulated, and the behaviour axis is superseded. |
| `control/driver.py`, `pid.py`, `pure_pursuit.py`, `path.py`, `policy.py`, `policy_driver.py`, `learning.py`, `tuner.py`, `demonstration.py`, `calibration.py`, `alignment.py`, `roam_driver.py` | The hand-written controller and the policy-learning stack. Both existed because BeamNG.drive had no API. The MCP server ended that. |
| `learn.py`, `policies/` | as above |
| `battery/corrosion.py`, `battery/electrical.py`, `sim/engine.py` | Not deleted -- **moved** and extended into `load/` and `cell/`, where the diagram puts them. They stop being called inline during driving. |
| `analysis/game_value.py`, `does_the_game_matter.py` | Answered their question; the answer is recorded in section 2. |

Kept: `sim/mcp_client.py`, `sim/backend.py`, `sim/fake.py`, `control/ai_driver.py`
(route and aggression), `control/places.py`, `control/health.py`,
`control/journey.py`, `control/interventions.py`, `datalog/`, `drive.py`.
`sim/mcp_backend.py` is reshaped around the Lua sampler.

`BehaviourSpec` is replaced by a plain `ScenarioSpec`: aggression, route,
duration, accessory load, ambient, vehicle configuration, seed. Bounds-checked
on construction and content-hashed for reproducibility -- no LLM, no cache, no
generation step.

## 12. Open questions

1. **Ambient temperature control.** `core_environment.setState` does not apply
   `temperatureC`. Another route may exist -- a level environment object via
   `set_object_field`, or a preference variable via `set_var`. Until one is
   found, ambient is a parameter of our thermal model rather than a simulated
   axis, and thermal counterfactuals are extrapolation, not simulation. Given
   ambient is the dominant aging driver, this is the highest-value unknown
   remaining.
2. **Determinism.** `get_simulation_state` reports a `deterministic` flag,
   currently false. Whether enabling it gives byte-reproducible physics decides
   how train and test splits can be constructed.
3. **Real measurements.** The final evaluation box needs true `I_bat` against
   true dynamics -- a current clamp and voltage logger on an actual car. Not
   available yet. Until it is, the electrical model is unvalidated.
4. **Full-life dataset.** Identified but not yet obtained, and it is the only
   thing setting absolute scale.
5. **Start-stop.** Assumed absent. If the target vehicle has it, the chemistry
   is EFB or AGM rather than flooded and cycle count enters the aging law.
