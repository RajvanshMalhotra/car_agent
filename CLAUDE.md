# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**Phase 1 partially implemented** (behaviour agent + controller + fake backend).
Built: `behaviour/` (spec, generator, cache, DeepSeek client), `control/` (path,
pure pursuit, PID, driver), `sim/` (backend interface, fake backend, engine and
thermal model, OutGauge/OutSim decoding, gamepad+UDP backend), `battery/`
(Arrhenius grid corrosion), `datalog/` (CSV + provenance sidecar), `agent.py`,
`windows_probe.py`, `windows_drive.py`.

Not built: `campaign/` (sampling + resumable ledger), the electrical/alternator
model for the recharge-deficit pathway, and trip cycling -- a run is one trip
with one crank, so the sulfation pathway has no data yet. Nothing has ever run
against the real game.

```bash
python3 -m pytest -q          # 227 tests, ~20 s, no network, no Windows machine
./agent.py list               # cached behaviours
./agent.py drive <name|hash>  # run one through the fake backend
export DEEPSEEK_API_KEY=...   # only needed for `./agent.py generate`

# on the Windows machine, in this order:
py windows_probe.py           # verifies ViGEmBus + decodes whatever UDP arrives
py windows_drive.py --list
py windows_drive.py "Delhi Courier" --seconds 300
```

No linter or formatter configured yet. Requires `openai` (DeepSeek is
OpenAI-compatible); everything else is stdlib.

### LLM provider: DeepSeek, not Claude

`behaviour/deepseek.py` targets `https://api.deepseek.com` via the `openai` SDK.
Models available on the current key: `deepseek-v4-pro` (default),
`deepseek-v4-flash`. Its JSON mode guarantees *parseable* JSON only -- no schema
enforcement -- which is why the schema is restated in the prompt and the real
gate stays in `BehaviourSpec.__post_init__`. Generation takes ~25 s per
behaviour, which is fine because it is offline and cached. The client sits
behind a `complete_json(system, user, schema)` seam, so swapping providers is a
one-file change.

## What this project is

Generate diverse, physics-constrained driving data in a vehicle simulator to improve the
generalization of a battery state model. An LLM emits driving-behaviour parameter sets, a
classical controller executes them in BeamNG, and the resulting vehicle-dynamics traces
feed a battery physics layer downstream.

The full concept spans nine stages (existing dataset -> behaviour agent -> simulation ->
physics model -> world model -> expanded dataset -> retrain -> evaluate on unseen
conditions). These are separate projects. **The current slice is the simulation harness
plus the behaviour agent only.**

## Hard constraints

- **The simulator runs on a different machine.** Development happens on macOS; BeamNG runs
  on a Windows laptop belonging to a collaborator. All sim-touching code runs on Windows.
  Anything that cannot be tested without that laptop is a design smell -- see the fake
  backend requirement below.
- **It is BeamNG.drive, not BeamNG.tech.** `beamngpy` requires BeamNG.tech, the academic
  fork (free for research on application to licensing@beamng.gmbh). Until that licence
  lands, control is via a virtual gamepad (ViGEmBus / `vgamepad`) and telemetry via
  BeamNG's native UDP outputs (OutGauge, Motion Sim) or a Lua mod streaming vehicle state.
  Exact setting names on the current build are unverified.
- **Full access to the Windows machine** is available: driver installs and mod folders are
  permitted.

## Architecture decisions

### The sim boundary is one interface wide
`sim/backend.py` defines `apply_control()` / `read_state()` / `reset()` / `close()`. Two
implementations: the gamepad+UDP backend (now) and a `beamngpy` backend (when licensed).
No other package imports anything BeamNG-specific. The .tech upgrade must be a drop-in.

### A fake backend is mandatory, not a nicety
A kinematic-bicycle fake backend lets the controller, campaign and logging layers be tested
on macOS with no Windows machine and no game. Without it every test needs someone else's
laptop. Treat it as core infrastructure.

### The LLM is offline, never in the control loop
Behaviour generation runs once per behaviour, is bounds-validated, and is cached to disk.
Runs must be reproducible from `(spec_hash, scenario, seed)`. No API call happens while a
vehicle is moving: it would add latency, nondeterminism, and per-run cost.

### Under-bonnet temperature is estimated, never measured
Neither OutGauge nor OutSim carries engine bay temperature, and BeamNG does not
model it. `sim/engine.py` estimates it from coolant temperature, road speed
(airflow) and ambient, and the same estimator runs on both backends. On the real
car the measured coolant temperature is used verbatim -- the model must never
integrate over a measurement -- and only the bay estimate stays modelled. The
structure is defensible; the coefficients are not calibrated. Ratios between
behaviours are the usable output, absolute temperatures are indicative.

**Measured effect sizes (model, not reality):** holding ambient fixed and the
car driving continuously, driving style moves corrosion 1.2-1.5x while ambient
25 C -> 42 C moves it ~2.9x. Ambient dominates style roughly 2:1. This is
consistent with "make the next experiment the hot one" but it also weakens the
case for driving-style diversity as the main source of variation. Caveat: the
behaviours all settled at 16-17 m/s because route and speed limit constrained
them, and `idle_fraction` is not yet enforced -- idling is where style should
matter most.

### Reaction lag is not one delay
`reaction_lag_s` governs response to discrete events and feeds the longitudinal
and corner-anticipation path only. Lane keeping is a continuous compensatory
tracking task limited by neuromuscular delay (`NEUROMUSCULAR_LAG_S`, 0.15 s).
Feeding the full reaction lag into the steering loop makes it oscillate past
~0.4 s and diverge entirely by 1.5 s -- and the LLM routinely emits 0.5-1.0 s.

### Behaviours are parameters, not learned policies
An LLM emits a `BehaviourSpec` (target speed factor, accel/decel limits, jerk limits,
following distance, corner speed, reaction lag, erraticness). A pure-pursuit steering +
PID speed controller executes it. RL was considered and rejected for this slice: millions
of steps in a near-real-time soft-body sim on one laptop, to produce a policy that drives
worse than a controller, for behaviour variation we can specify directly.

### Validation is a code gate, not a prompt instruction
`BehaviourSpec` bounds-checks every field against physical limits on construction. An LLM
asked for "aggressive" will emit sustained 0.9g braking. Physics guardrails begin at
behaviour generation, not at the downstream physics model.

### Campaigns must be sampled and resumable
The sweep space (behaviour x route type x ambient temperature x traffic density x weather)
is combinatorially large -- a full grid is on the order of 1000+ runs at ~10 minutes of
real time each. Sample it (full coverage on behaviour x route, stratified over the rest),
and persist a run ledger so an interrupted campaign resumes instead of restarting.

## The existing battery dataset

Columns: `current, voltage, temp, dt_hours, ah_consumed, total_ah_used, True_SoC`.

**Known-critical finding.** The target is an exact closed-form function of the inputs:

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 * (1 - total_ah_used / 16.069411)

Verified exact on every sampled row; the capacity constant is stable to six decimals. The
reported ~99.7% model accuracy therefore reflects **label leakage, not generalization** --
the model is recovering a division. Adding data diversity will not change this number and
will not demonstrate anything. The target variable must change (e.g. predict voltage or
terminal behaviour under load, or SoH, from dynamics) before expanded data can prove
anything. Do not treat "improve accuracy on True_SoC" as a success criterion.

How the data was produced: a light bulb was attached to the battery and left on. That is
a constant resistive load -- ~3.0 A flat, 31.0 degC pinned, 1 Hz -- which is a legitimate
capacity-test protocol but yields **one point in the stress space**. Load-dependence
parameters (current exponent, Arrhenius term) are unidentifiable from a single condition.
The battery reached only ~99.7%, so there is very little degradation content.

Useful columns after removing derived and constant ones: **`current` and `voltage` only**.
`temp` and `dt_hours` are constant; `ah_consumed`, `total_ah_used` and `True_SoC` are all
derived and all leak. Any masked-autoencoder over the full column set will just learn the
arithmetic identities.

## Target application: SLI lead-acid in combustion cars

**This is a 12 V starter battery in a petrol/diesel car, not an EV traction pack.** The
chemistry is lead-acid, not lithium. Both facts invalidate large parts of the standard
literature and several earlier decisions.

An SLI battery is held near 100% SoC by the alternator whenever the engine runs, and never
deep-cycles: cranking draws 200-600 A for 1-3 s, recharged within minutes, 1-3% DoD per
start. So "aggressive acceleration drains the battery" is NOT the mechanism. The real
pathways, in order of importance:

1. **Grid corrosion (dominant)** -- driven by under-bonnet temperature. Hard driving means
   higher engine load, hotter engine bay, faster corrosion. Arrhenius: +10 degC roughly
   halves life.
2. **Sulfation** -- short-trip patterns; repeated cranking without a full recharge.
3. **Recharge deficit** -- idling and stop-go with AC/lights/blower, where alternator
   output at low RPM may not cover electrical load.
4. **Plate shedding** -- vibration.
5. **Cycle count** -- only if the vehicle has start-stop (then EFB/AGM, not flooded).
   Currently unknown; design for conventional ICE with start-stop as a configurable flag.

### Consequences for the design

- **BeamNG is the right simulator** -- combustion cars, engine load, RPM and engine
  thermals are exactly what it models.
- **The load model is thermal and trip-pattern driven, not traction driven.** Regenerative
  braking is irrelevant (there is none). What the sim must supply: engine bay temperature,
  RPM, idle fraction, trip segmentation, crank events, electrical accessory state.
- **`BehaviourSpec` fields shift accordingly.** Trip length, idle time, ambient temperature
  and accessory usage matter more than acceleration aggressiveness.
- **Aging law: Schiffer et al. (2007)** weighted-Ah-throughput model with sulfation and
  corrosion terms. NOT the Wang graphite/LFP power law -- that is lithium and does not
  apply.
- The existing lamp data maps to the **parasitic-drain / standing-sulfation** pathway, not
  the corrosion pathway that dominates in service. Useful but narrow.
- If only one more experiment is run, make it the **hot** condition, not the high-current
  one -- temperature is the primary aging driver for this application.

## Research framing

Target is **RUL**, via early-life prediction: real data covers only the first sliver of
degradation and cannot be aged further, so simulation plus a world model must bridge to
end-of-life. Precedent: Severson et al. (2019) predict cycle life before capacity
degradation.

**The world model's job is counterfactual query, not data generation.** Benchmarked, a
GRU rollout is ~26x slower than the 4-state ODE (144.9 ms vs 5.6 ms for one hour at 1 Hz),
so it cannot be justified as a surrogate for speed, and it cannot add information the
generator lacks. It earns its place through per-cell counterfactuals -- requiring Pearl's
abduction -> action -> prediction, since plain action-conditioned rollout yields an
interventional average, not a counterfactual -- and through fusing representations learned
from real data.

Novelty position: latent world models are absent from RUL, per-cell counterfactual RUL has
no located prior work, and SLI lead-acid RUL under driving patterns is far less studied
than EV lithium. Established and therefore cited as apparatus, never claimed: the
driving-style/degradation link, LLM scenario generation (ROADGPT targets BeamNG.tech),
and synthetic-to-real pretraining.

### Validation strategy

The endpoint is unobservable, so validate the parts separately:
- **Aging law** -- against real full-life lead-acid data (an open 8-year flooded lead-acid
  grid-storage dataset exists at 1 s resolution; note flooded-vs-AGM and grid-vs-vehicle
  regime differences before assuming transfer).
- **Load model** -- against the existing early-life bulb data.
- Plus: hold out the tail of existing data, check rank-ordering of load severities, and
  sanity-band against published lead-acid cycle life.

Do not claim absolute RUL accuracy at end-of-life. It cannot be checked.

### Output contract — what the model returns

**Never a bare point estimate.** Every RUL prediction is a calibrated conservative
bound, and the headline figure is the safe one:

    847 days  (90% interval: 620 - 1150)  ->  plan for 620

The stochastic latent supplies the distribution; report a low quantile, not the mean.
"Safe estimate" means conservative lower bound, and that is what a fleet operator acts on.

Alongside the absolute figure, three relative claims that carry less validation burden
and are frequently more useful:

- "Profile A kills the battery roughly 2x faster than profile B"
- "This battery is aging faster than expected for its usage"
- "Fewer short trips would have extended it by roughly X%"

**What gets validated is calibration, not point accuracy.** If a 90% interval is claimed,
the truth must land inside it ~90% of the time on held-out data. That is checkable with
modest data; point accuracy at end-of-life is not.

Absolute scale is the weak link: physics gives the fade curve's *shape*, but the rate
constant must be fitted to cells that actually reached end-of-life. **No dead batteries
are available** -- so the open full-life lead-acid dataset is load-bearing, not optional,
and it is the only thing setting the absolute scale. Back it with a literature sanity
band: SLI batteries typically last 3-5 years, 2-3 in hot climates. A model predicting
12 years is broken.

Expect wide intervals and say so plainly.

### Known-thin argument

Step "generalise the PINN on expanded data" remains largely circular -- expanded data comes
from the world model, trained on the ODE. The masked real-data latents are the only thing
breaking the loop, and they are early-life, two-channel, single-condition. Circularity is
reduced, not eliminated. State the claim accordingly.

Also: the PINN physics loss must use **time-varying capacity**, not nominal, or a correct
model carries systematic residual that grows as the cell ages.
