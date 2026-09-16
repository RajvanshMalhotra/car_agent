# What is new here

**In one line:** a model that infers a specific battery's hidden wear from
ordinary measurements, answers *"what if this car had been used differently"*,
and keeps working in a climate it was never trained on.

Most battery-life research asks **"is this battery dying?"** on a population.
This asks **"what would have happened to *this* battery under a different
owner?"** — and then, with the bridge, turns that into a life estimate.

---

## The three claims, strongest evidence first

### 1. How to make a learned world model survive conditions it never saw

**The claim.** A world model that predicts absolute values fails badly under a
shift in conditions. One that predicts a *correction to a physically sensible
reference* does not — and **where** that reference enters the network decides
whether it works at all.

**The evidence.** Same model, same data, only the representation changed:

| | Accuracy on an unseen 42 °C climate |
|---|---|
| Predicting absolute values | **31.9%** |
| Predicting a correction to a reference | **88.3%** |

The failure is a clean bias, not noise: the model forecast 37 °C for batteries
actually at 43 °C — *after seeing 43 °C in its own input*. It pulls everything
back toward the climates it trained on.

**Three failed attempts that sharpen the claim** (negative results, all
recorded):

| Attempt | Result on an unseen climate |
|---|---|
| Feed the reference into hidden layers | 38.4% — collapse |
| Let attention learn the reference | 77.2% — worse |
| Feed in smoothed history, relative to the reference | loses 8 of 8 paired runs |

So the reference works **because** it is added at the very end, untouched by
any nonlinearity. Everything routed deeper cost more than it gained.

**A diagnostic that predicts failure before training.** Measure what share of
the inputs a network sees fall outside the range it trained on:

| Input | Outside training range |
|---|---|
| Absolute temperature | **100%** |
| Smoothed temperature | 26% |
| Temperature *differences* | **0.1–0.4%** |

Differences transfer; levels do not. This is a general lesson about world
models under distribution shift, not a battery trick.

### 2. Per-battery counterfactual reasoning (the original goal)

**The claim.** Existing life prediction gives population averages. Answering
*"what if this owner had driven differently"* requires inferring the hidden
state of this specific cell first — abduction, then intervention.

**The evidence.** The model was tested against a deliberately strong control:
not "beats an empty model" (trivial), but "beats a copy of itself whose memory
of the *order of events* has been erased". It won 69.7% of the time, and won
in **every** wear regime. The simpler frozen-memory model managed 53.6% and
lost outright in the low-wear regime.

Under noise and a climate shift, the day-by-day model's "what if" predictions
were **twice as accurate** as the frozen-memory model's, on 96.5% of test
batteries.

**The honest caveat.** The right answer comes from the same physics simulator
that generated the training data. This shows the model recovers *our
simulator's* hidden state, not a real battery's.

### 3. The application itself

12 V starter batteries in petrol cars, aged by driving and parking patterns.
The literature is overwhelmingly electric-vehicle lithium. Good motivation,
thin as a standalone contribution — use it as framing.

---

## What must be cited, never claimed as new

- Driving style affects battery wear (settled, with published numbers)
- World models, RSSM/Dreamer architectures
- Residual and delta prediction in time series — **the "predict a correction"
  trick is a cousin of this.** The contribution is the evidence about
  out-of-distribution behaviour and the range diagnostic, not the trick
- LLM-generated driving scenarios, synthetic-to-real pretraining

---

## The cherry on top: the bridge

Everything above predicts **measurements** — voltages, currents, temperatures,
charge level. That is a very good simulator, not a life predictor.

**The bridge** feeds those predicted measurements into the physical wear law,
turning a forecast into **days of life**, and turning a counterfactual into a
sentence an owner can act on:

> *"Fewer short trips would have extended this battery's life by about 20%."*

**Why it is credible.** The wear law is not learned — it is the same
Arrhenius corrosion and sulfation physics used to generate the data. Feasibility
is measured, not assumed:

| Wear pathway | Recoverable from the model's own daily outputs? |
|---|---|
| Corrosion (the dominant one) | Yes — within **3–4%** at 25 °C and 42 °C |
| Sulfation | Yes — implied low-charge hours land at **0.97–1.00×** the true value |

**What the bridge still cannot say.** The wear law's absolute rate constant was
never fitted to batteries that actually died, and the code deliberately
*refuses* to print a life in days because of it. Absolute figures also carry a
known ~3× bias. So the trustworthy output is **ratios and relative claims** —
"2.3× faster", "20% longer" — which are exactly the claims that do not depend
on the unfitted constant.

Making absolute days honest needs one more thing: fitting the rate constant to
an open full-life lead-acid dataset. That is the next milestone, not a
detail.

---

## What the numbers rest on

The headline was validated properly, which most of the earlier work was not:
**16 training runs**, holding out each of four climates in turn, two random
seeds each.

| | Normal forecast | "What if" forecast |
|---|---|---|
| **This design** | **89.2%** (±2.1) | **72.6%** (±6.2) |
| Best non-learning baseline | 82.3% (±0.9) | 60.1% (±2.3) |

The headline 88.3% reproduced at 87.8% and 88.2% on re-runs.

**An honest correction that came out of this.** Results swing by up to 6 points
from random seed alone. Several small wins reported earlier from single runs
(1–2 point margins) were never real differences, and have been retracted in the
detailed report. The large effects stand.

---

## Limitations, stated up front

1. **All data is simulated.** One recorded drive, replayed through a physics
   model. No real battery was aged.
2. **The wear law's scale is unfitted**, so absolute lifetimes are not claimable.
3. **The action space is thin** — essentially "drove today" or "parked today",
   plus per-battery constants. Driving style and trip length never vary within
   a battery, so "fewer short trips" is not yet answerable from this dataset.
4. **Uncertainty is not yet honest.** When the model claims a 90% confidence
   range, the truth falls inside it only 10–63% of the time depending on
   conditions. Calibration is the metric this project says it will be judged
   on, and it is not there yet.
5. **Abduction and calibration have not been re-measured** on the current best
   model — those numbers come from an earlier version.

---

## The paper sentence

> *Counterfactual world models for battery ageing that generalise to unseen
> thermal conditions.* We show that a latent world model can infer hidden,
> path-dependent sulfation state and answer per-battery counterfactuals; that
> naive latent world models fail catastrophically under thermal shift; and
> that anchored relative prediction fixes it, with a diagnostic that predicts
> the failure before training. A physics bridge converts these forecasts into
> relative life estimates.
