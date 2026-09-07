# Project conversation history

Record of design discussion. Newest sections at the bottom.

---

## Session 1 — 2026-09-07

### Starting point

User shared a poster describing a 9-stage pipeline: "Generating Diverse,
Physics-Constrained Driving Data to Improve Battery Model Generalization."
Stages: existing dataset -> problem (overfitting) -> LLM+RL behaviour agent ->
BeamNG.tech simulation -> battery physics model -> world model -> expanded
dataset -> retrain battery model -> test on unseen conditions.

Working directory was empty. No code existed at any point in this session.

### Scope decomposition

The poster is not one project. Identified five independently buildable pieces:
behaviour agent, sim harness, battery/physics layer, world model, and
training/evaluation. **Decision: scope the current slice to harness +
behaviour agent** (boxes 3 and 4).

### Environment constraints established

- Simulator runs on a collaborator's **Windows laptop**; development is on macOS.
- **Decision: all sim-touching code runs on the Windows machine.** Rejected
  driving it remotely over LAN (firewall, latency in a closed control loop).
- The installed game is **BeamNG.drive, not BeamNG.tech**. Confirmed via
  Context7 docs that `beamngpy` requires BeamNG.tech, the academic fork
  (free for research on application to licensing@beamng.gmbh).
- Full access to the Windows machine: driver installs and mods permitted.

### Control/telemetry decision

User initially suggested driving the game via OS-level input. Raised that this
solves *input* but not *output* -- the deliverable is the telemetry, and
synthetic keyboard input is binary, collapsing behaviour distinctions.

Three options weighed:
- A: OS control + screen OCR. Rejected -- brittle, low-rate, binary input.
- B: virtual gamepad (ViGEmBus/`vgamepad`) for analog control + native UDP
  telemetry (OutGauge, Motion Sim) or a Lua mod streaming vehicle state.
- C: apply for BeamNG.tech and use `beamngpy`.

**Decision: build B now behind a pluggable `SimBackend` interface, apply for C
in parallel, swap the backend in when the licence lands.**

### Agent architecture decision

Poster specified "LLM generates behaviours -> RL agent executes." Argued against
the RL half: millions of steps in a near-real-time soft-body sim on one laptop,
producing a policy that drives worse than a controller, for behaviour variation
that can be specified directly.

**Decision: LLM emits a validated `BehaviourSpec` parameter set; a pure-pursuit
(lateral) + PID (longitudinal) controller executes it.** Deterministic,
reproducible, works immediately. RL can be added later behind the same
Controller interface.

### Architecture presented (Section 1 of 6, approval pending)

Packages: `llm_behaviour/`, `control/`, `sim/`, `campaign/`, `logging/`.
Three properties bought deliberately:
1. The sim boundary is one file wide (`sim/backend.py`) -- the .tech swap
   touches nothing else, and a fake kinematic backend makes everything
   testable on macOS with no game and no Windows machine.
2. The LLM is offline and cached, never in the control loop -- no latency,
   no nondeterminism, runs reproducible from `(spec_hash, scenario, seed)`.
3. Validation is a code gate in `BehaviourSpec.__post_init__`, not a prompt
   instruction.

### Scenario diversity

User selected all four dimensions: route type, ambient temperature, traffic
density, weather/friction. Flagged the combinatorial cost -- a full grid is
~1260 runs at ~10 min real time each (~210 hours). **Decision: sample the
space (full coverage on behaviour x route, stratified elsewhere) and make
campaigns resumable via an on-disk run ledger.**

### Existing dataset analysis — critical finding

User shared the dataset: columns `current, voltage, temp, dt_hours,
ah_consumed, total_ah_used, True_SoC`.

Verified numerically that the target is an exact closed-form function of the
inputs:

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 * (1 - total_ah_used / 16.069411)

The capacity constant is stable to six decimals across every sampled row.
**The reported ~99.7% accuracy is label leakage, not overfitting to a narrow
distribution.** Adding data diversity cannot change a number that is already
exact. Also noted: the data is a constant-current bench discharge (~3.0 A flat,
31.0 degC pinned, 1 Hz) containing no driving at all; and 12.15 V x 16.07 Ah
(~195 Wh) is an e-bike/scooter/lab-cell scale, not an EV traction pack.

### /init

Wrote `CLAUDE.md` recording constraints, architecture decisions and the dataset
finding. No codebase existed to analyse; no invented build commands. Noted the
presence of a Codex config and a Gemini directory and pointed the user at
`/import` rather than reading them directly.

### Full formulation supplied — target is RUL

User clarified the real target is **RUL**, and supplied the full mathematical
formulation: vehicle-dynamics-to-current interface, coupled SOC/SOH ODEs with
an Arrhenius stress law, terminal voltage with health-dependent internal
resistance, lumped thermal dynamics, a GRU + stochastic-latent world model, a
PINN loss on the decoder, and RUL extraction at the 80%-capacity threshold.

Corrections raised, in order of cost:

**Structural:** the ODE *defines* the degradation law, so RUL labels are a
deterministic functional of an assumed equation. Training and testing a
surrogate on that ODE measures interpolation fidelity, not battery science.
Box 9 cannot be evaluated on synthetic data without the pipeline being
self-referential. Real degradation data (Severson 2019, NASA PCoE, Oxford)
must enter.

**Concrete errors:**
1. EOL threshold inconsistent -- `0.8*C_nom` with `C_max(N_EOL)=5.6 Ah` implies
   C_nom = 7.0 Ah, contradicting the dataset's 16.069411 Ah.
2. `dC/dt = -lambda*C` gives exponential decay, so fade *decelerates*; real
   cells show sqrt-like early fade then a knee into accelerated collapse.
   The knee is the hard part of RUL prediction, and this form cannot produce
   one. Recommended the Wang et al. (2011) power law on Ah-throughput,
   `Q_loss = B*exp(-Ea/RT)*(Ah)^z`, z ~= 0.55.
3. `DoD(t)` is ill-defined in a continuous ODE -- it is a per-cycle quantity.
   Recommended rainflow counting plus Palmgren-Miner accumulation.
4. `V_pol(I,T)` written as instantaneous is just a second resistor.
   Recommended a 1-RC Thevenin state.
5. Thermal source term undercounts heat: should be `I*(OCV-V)`, i.e.
   `I^2*R_int + I*V_pol`.
6. `I_demand` omits regenerative braking (sign, direction-dependent
   efficiency) and auxiliary/HVAC load -- the latter matters especially
   because ambient temperature is a scenario variable.
7. RUL extraction references `C_max` but the decoder outputs only
   `[V, I, T, SOC]` -- the quantity is not in the observation space.
8. The PINN loss uses constant `C_nom` while the generator uses time-varying
   `C_max(t)`, so a perfect model has systematically nonzero residual.

**Recommended:** split timescales -- world model for fast within-cycle
dynamics, physics ODE for slow cross-cycle degradation on aggregated stress
features. Avoids latent drift over thousand-cycle horizons.

### World model justification examined

Benchmarked the premise that the world model "replaces slow ODE integration":

    ODE  (pure-Python loop, 4 states):     5.6 ms   (1 hour @ 1 Hz)
    GRU  (256 hidden, sequential rollout): 144.9 ms
    -> GRU is 26x SLOWER, against an unoptimised baseline

Neural surrogates pay off against PDE-scale physics (DFN/PyBaMM, thermal FEM),
not a lumped 4-state ODE. The "expand the distribution" justification is also
weak when you own the generator -- extrapolating a learned model outside its
training support yields less reliable data, not more.

Where a world model does earn its place: (1) learned from real data, (2) as a
learned residual on the ODE fitted to real cells, (3) with a genuinely
expensive teacher such as PyBaMM's DFN, (4) for uncertainty quantification.

User reaffirmed wanting the world model, for learning value and for RUL
accuracy. **Decision: keep it, but repoint the teacher and add real data.**

Three distinct mechanisms by which it can help were separated: representation
pretraining (the accuracy mechanism), long-horizon rollout stability, and
action-conditioned counterfactuals (the novelty mechanism).

### Literature scan

Findings:
- **Driving style -> degradation is settled science.** Published quantitative
  results: ~2.5x faster aging, 18.4x immediate aging stress vs eco-driving,
  21% vs 53% capacity loss over 1000 cycles, RMS current 66 A vs 21 A.
  Must be cited as motivation, never claimed as a finding.
- **LLM -> driving scenarios is done**, including ROADGPT which targets
  BeamNG.tech specifically. Box 3 is engineering, not novelty.
- **Synthetic physics pretraining -> fine-tune on real cells is done**
  (electrochemical-model data transferred to NASA Ames cells).
- **Gap 1: latent world models are essentially absent from RUL.** The field is
  transformers, CNN-LSTM, ODE-LSTM, GPR, XGBoost, survival analysis.
- **Gap 2: nobody does per-cell counterfactual RUL.**

Key paper found: *How Can Driving World Models Do Counterfactual Prediction?*
(arXiv 2608.11601). Action-conditioned rollout is **not** counterfactual
prediction -- it discards the factual continuation, which carries the
episode-specific latent state. Pearl's **abduction -> action -> prediction** is
required. Without abduction you get an interventional average, not a
counterfactual for a specific cell. This directly justifies the stochastic
latent, which would otherwise be decoration.

**Resulting framing:** counterfactual RUL prediction for individual cells via
abduction in a behaviour-conditioned latent world model. Contribution is the
abduction machinery plus the first world model for RUL; the LLM, the
driving-style finding and synthetic pretraining are apparatus.

Venue: applied venues (eTransportation, Applied Energy, IEEE TTE) suit a
composition-of-known-parts paper; user confirmed IEEE Transactions is the
target.

### Top-venue guidance and compute

Discussed what would be needed for NeurIPS/ICML/ICLR with a future project
combining world models and voice models: mechanism over application,
rollout-error-vs-horizon rather than one-step, parameter- and compute-matched
baselines, human MOS evaluation and ethics/provenance for speech, and the
"two cool things stapled together" risk.

User has **ongoing access to 1x H100**. Noted this buys experimental
throughput (multi-seed runs, full ablation grids, compute-matched baselines)
rather than scale, and that world-model training is typically
environment-bound rather than GPU-bound -- vectorise environments or the card
idles.

### Chemistry and application resolved — major redirect

Two clarifications landed late and invalidated several earlier decisions.

**It is lead-acid, not lithium.** The Wang graphite/LFP power law recommended
earlier is the wrong chemistry and was retracted. Lead-acid ages by sulfation,
grid corrosion, water loss and plate shedding, not SEI growth. Correct model:
**Schiffer et al. (2007)** weighted-Ah throughput with sulfation and corrosion
terms. Noted that Peukert's law is far stronger in lead-acid than lithium, and
that the Li-ion-dominated RUL literature largely does not apply -- which
improves the novelty position.

**It is an SLI starter battery in a combustion car**, not traction. This is the
case previously flagged as potentially fatal to the premise. Verdict: the
premise survives but the causal pathway changes entirely. An SLI battery is
alternator-held near 100% SoC and never deep-cycles (200-600 A cranking for
1-3 s, 1-3% DoD per start). Degradation pathways, ranked: grid corrosion from
under-bonnet temperature (dominant, Arrhenius), sulfation from short-trip
patterns, recharge deficit at idle with accessory load, plate shedding from
vibration, and cycle count only if start-stop equipped (unknown; designed
around a configurable flag).

Consequences: BeamNG becomes a *better* fit (combustion cars, engine load, RPM,
engine thermals); the load model is rewritten from traction to thermal and
trip-pattern driven, with regen braking dropped entirely; `BehaviourSpec` shifts
toward trip length, idle time, ambient temperature and accessory usage.
Earlier advice to prioritise a high-current experiment was reversed -- the
**hot** condition matters more, since temperature is the primary aging driver.

### Data provenance

Degradation was produced by attaching a light bulb and leaving it on. A valid
capacity-test protocol, but it yields **one point in the stress space**, so
load-dependence parameters are unidentifiable. The battery reached only ~99.7%,
so degradation content is minimal. After removing derived and constant columns,
only `current` and `voltage` carry information -- masked autoencoding over the
full column set would learn arithmetic identities instead of signal structure.

Recommended (not yet actioned): additional batteries under differing load and
temperature to unlock load-dependence, and repeated cycling rather than single
discharge. User reported no materials available to degrade further, which is
what motivates the simulation-plus-world-model bridge in the first place.

### Framing settled

Early-life RUL prediction: real data covers only the first sliver of
degradation, so simulation and the world model bridge to end-of-life
(precedent: Severson et al. 2019). **World model's role fixed as counterfactual
query, not data generation**, per the benchmark showing it 26x slower than the
ODE. Validation splits into aging law (against open full-life lead-acid grid
data) and load model (against the bulb data), since the endpoint itself is
unobservable. Absolute end-of-life RUL accuracy is explicitly not claimable.

Acknowledged as thin: the final PINN step remains largely circular, broken only
by early-life two-channel single-condition real latents.

### Output contract settled

Discussion of whether the deliverable is an absolute RUL figure or a relative
estimate. Position raised: absolute day-counts are unverifiable because the
endpoint is never observed. User confirmed absolute prediction is the core
intent.

**Resolution: predict absolute days, but as a calibrated conservative bound
rather than a point** -- e.g. "847 days (90% interval 620-1150), plan for 620".
This matches the user's own phrase "safe estimate", is what an operator acts
on, and is defensible where a bare point estimate is not. The stochastic latent
supplies the distribution, which is a third justification for it alongside
abduction. Validation target is **calibration** (does the truth fall inside the
claimed interval at the claimed rate) rather than point accuracy.

User also accepted the three relative claims as legitimate outputs: severity
ratios between profiles, aging-faster-than-expected detection, and
counterfactual percentage extensions.

Suggested and rejected: sourcing used car batteries of known age from a
scrapyard or mechanic to obtain real (age, remaining capacity) pairs cheaply.
User cannot obtain dead batteries. **Consequence: the open full-life lead-acid
dataset becomes load-bearing** -- it is the only source of absolute scale, since
physics supplies the fade curve's shape but not its rate constant. Backed by a
literature sanity band of 3-5 years typical SLI life, 2-3 in hot climates.

### Deliverables produced

- `CLAUDE.md` -- written, then substantially revised for lead-acid/SLI.
- `history.md` -- this file.
- Research brief artifact for supervisor review:
  https://claude.ai/code/artifact/2a1d425c-1096-47e2-93e9-8cea2b1e77e5
  (predates the lead-acid/SLI redirect -- needs updating before it is shared.)

### State at end of session

- No code written. Design Section 1 of 6 presented; approval still pending.
- Agreed plan, three parallel tracks: (A) real data -- acquire open lead-acid
  degradation data, calibrate fade law, masked latents on current/voltage;
  (B) simulation -- BeamNG harness, behaviour agent, sweep; (C) model -- world
  model with gated fusion of real latents, then PINN.
- Open: start-stop or conventional; whether the supervisor brief gets updated.
