# Can a learned world model abduct a hidden, path-dependent battery state?

**Research question.** Can a learned world model perform ABDUCTION — infer a
hidden, path-dependent physical state (the sulfation crystal size,
`AgingState.crystal` in `cell/aging.py`) from an observation window, well
enough to answer counterfactual queries about a different future action
sequence?

**Headline answer, after four rounds of tightening the methodology: a real
but partial, localized signal — not general abduction, and not a clean
negative either.** This report went through four distinct stages, each one a
genuine methodological correction (or, in round 4, an attempted fix) rather
than a retuning toward a preferred result, and all four are preserved below
because the progression is itself the most honest part of this experiment:

1. **A void result** (Section 2): the first training run's clean-looking
   "not abducting" finding was an artifact of a collapsed posterior — the
   model had nothing to abduct with, so its agreement with the
   no-information baseline was arithmetic, not evidence.
2. **An over-claim from a weak control** (Sections 3–5, first pass): after
   fixing the collapse, the abducted latent beat a **zero-information**
   baseline by a wide margin (48.4% of the factual-to-zero gap closed), and
   this was reported as "genuine, partial abduction." That control was too
   weak: beating a latent that carries *nothing* is expected of *any*
   non-dead latent, whether or not what it carries is the hidden,
   path-dependent state specifically, or just the battery's cheap,
   currently-observable condition (state of charge, temperature).
3. **A properly controlled negative result** (Section 5, second pass):
   replacing the zero-information control with a **last-day-only** control —
   same encoder, same architecture, same amount of per-day information, all
   history erased — collapsed the finding. The abducted latent closed only
   14.1% of the factual-to-lastday gap on average, and its per-example win
   rate against the last-day control was **46.5%, below chance**. The
   crystal-bin breakdown made it worse, not better, for the abduction
   hypothesis: win rate was *lowest* in the low-crystal bins and *highest*
   in the high-crystal bin — the opposite of the mid-transition advantage
   genuine path-dependent recovery would predict. Step 4's SAE probe agreed:
   the best crystal-tracking feature (r=0.168) did not clearly beat SoC
   (r=0.155), and both were beaten by a directly observable target
   (battery temperature, r=0.256).
4. **A training curriculum fix, tested rigorously rather than assumed to
   work** (Section 7): the suspected cause of round 3's negative result was
   that training only ever saw the FACTUAL action continuation, so nothing
   forced the latent to distinguish "history" from "current state."
   `experiment/build_counterfactual_targets.py` generates a randomized
   counterfactual continuation per training window, with EXACT physics-ODE
   ground truth (re-running `calendar_sim.roll_days` from the window's true
   resumed hidden state, the same generator-owned trick Step 3 uses for
   eval), and training now requires one encoded latent to correctly predict
   BOTH the factual and this counterfactual future. Re-run against the SAME
   properly-specified lastday control, with a two-sided binomial
   significance test replacing an arbitrary win-rate threshold: the overall
   win rate rose to **53.6% (p=0.0053, statistically real, not noise)**, and
   Step 4's SAE probe now shows crystal (r=0.628) clearly beating SoC
   (r=0.541) by more than the required margin. But the crystal-bin
   breakdown shows this is **not a uniform effect** — it is strongly
   significant in the saturated-crystal bin (70.0% win rate, p<0.0001) and
   *significantly reversed* in the lowest-crystal bin (39.0% win rate,
   p=0.0001, the abducted latent doing WORSE than the lastday control
   there) — so the honest characterization is a real, localized, partial
   signal, not general path-dependent abduction.

**Corrected verdict: after the curriculum fix, the model shows a genuine
but partial and non-uniform ability to use history beyond what the current
state reveals — concentrated specifically in fully saturated batteries, and
absent (or reversed) for batteries early in the aging process.** This is a
meaningfully different, more interesting finding than either "yes, general
abduction" or "no, pure state-conditioning" — see Section 7 and the final
verdict in Section 8 for the full account and what it plausibly means
mechanistically.

---

## 1. Dataset design (unchanged from the first run)

### 1.1 Why a new, literal calendar simulation was needed

`cell/life.py`'s `project()` and `sweep_dataset.py`'s existing sweep
(`runs/sweep_cont/`) deliberately fold a periodic layup pattern into a single
**per-day-average** `Damage`, sampled every 30 days — the right move for
projecting the scalar health/hidden-state curve cheaply over a decade, and
the wrong move for this experiment. The brief's window needs the literal
day-to-day *observable* pattern (trip count, driving minutes, voltage,
current, temperature) that an external observer would actually see, and the
existing tables don't carry it: `aging_trajectory.csv` has only the hidden
state at 30-day resolution, and `within_trip.csv` has one reference replay
per ambient, not per scenario per day.

So `experiment/calendar_sim.py` reimplements the day-stepping loop
literally: it walks real calendar days, alternates `layup_gap_days`-long
stretches of ordinary two-trips-a-day driving with `layup_days`-long
continuous parks, and calls straight into `cell.integrate.run_trip` /
`run_soak` for each real day (which call `cell.aging.accumulate()`
internally, so the hidden state advances on real per-day damage, not an
averaged one). Benchmarked directly: an ordinary day (two ~10-minute trips
plus an 8 h soak at 60 s resolution) costs 5–10 ms; a layup day (24 h at
300 s resolution) costs under 2 ms. This made literal day-by-day simulation
of an 80-scenario, 2000-day dataset (160,000 day-rows) tractable at ~22
minutes total, rather than needing the period-averaging shortcut.

**Deliberate departure from the main project's measurable/latent
convention.** `cell/sensors.py` treats state of charge and bay temperature
as LATENT (nothing on a real car measures them; see CLAUDE.md's "under-bonnet
temperature is estimated, never measured"). This experiment's brief
explicitly asks for state of charge and bay temperature as *observable*
window features. That instruction is followed literally here — this is a
self-contained experiment about a different question (hidden-state
abduction, not the main project's sensor-noise-realism layer) and is flagged
so the two are never confused.

### 1.2 Scenario generation

80 scenarios: the same discrete ambient grid as `sweep_dataset.py`
(-10, 10, 25, 42 °C) × 20 scenarios each, with `trip_minutes`,
`layup_days`, `layup_gap_days`, and `parasitic_a` drawn uniformly per
scenario from the same ranges `sweep_dataset.py` uses, under an independent
sampling seed (42) — a fresh draw, not a re-read of `runs/sweep_cont/`.
Horizon: 2000 days per scenario (long enough to comfortably cover the
measured 540–1170 day, median-630 crystal transition band with margin on
both sides).

Output: `runs/experiment/daily_trajectory.csv` (160,000 rows: 80 scenarios ×
2000 days), `runs/experiment/scenarios.json` (per-scenario axes), and
`runs/experiment/daily_dataset.json` (sidecar/provenance).

### 1.3 Windowing and the target-flattening gate

Each example: a 60-day window of daily `ACTION_FEATURES`
(`is_layup`, `driving_minutes`, `soak_hours`, `ambient_c` — exogenous,
schedule-controlled) and `OBSERVABLE_FEATURES` (`v_mean`, `v_min`, `v_max`,
`i_mean`, `i_mean_abs`, `t_bat_mean`, `t_bat_max`, `t_bay_mean`, `t_bay_max`,
`soc`), paired with the true `crystal` at the window's end (the abduction
target) and `corrosion_hours`/`shedding`/`soh` retained as ground truth for
scoring only. The model never sees any of the four hidden fields. Every
window is also guaranteed a `ROLLOUT_K = 30`-day clean runway beyond its end
day (`experiment/windows.py`) — the same 30 days both the new multi-step
training (Section 3) and the abduction test (Section 5) roll out over.

The prior sweep (this brief's own critical context) found that sampling a
window end day *uniformly* reproduces the same near-binary skew a fixed
10-year horizon sample showed — the transition is real but narrow relative
to a battery's full life. So window end days are drawn by **stratified
sampling on the crystal target itself**: bin the candidate pool into 10
crystal bins and cap how many windows are drawn per bin
(`experiment/windows.py`).

Scenario-level holdout: 64 scenarios (80%) → train, 16 (20%) → test, split
*before* any window is drawn (`experiment/windows.py:scenario_split`), so no
test window shares a scenario with any training window.

**GATE 1 result — PASS** (unchanged from the first run; the dataset was
never the problem).

| bin | train (n=6000) | test (n=1500) |
|---|---|---|
| [0.0, 0.1) | 600 (10.0%) | 150 (10.0%) |
| [0.1, 0.2) | 600 (10.0%) | 150 (10.0%) |
| [0.2, 0.3) | 600 (10.0%) | 150 (10.0%) |
| [0.3, 0.4) | 600 (10.0%) | 150 (10.0%) |
| [0.4, 0.5) | 600 (10.0%) | 150 (10.0%) |
| [0.5, 0.6) | 600 (10.0%) | 150 (10.0%) |
| [0.6, 0.7) | 600 (10.0%) | 150 (10.0%) |
| [0.7, 0.8) | 600 (10.0%) | 150 (10.0%) |
| [0.8, 0.9) | 600 (10.0%) | 150 (10.0%) |
| [0.9, 1.0) | 600 (10.0%) | 150 (10.0%) |

Every bin hit its cap exactly (largest bin fraction 10.0%, well under the
~25% gate). 122,304 train and 30,576 test candidate (scenario, end-day)
pairs were available before sampling.

---

## 2. History: the first run's void result, and why it was void

### 2.1 The apparently clean negative result

The world model (`experiment/model.py`, encoder-decoder GRU with an
8-dimensional stochastic latent, ~11,130 parameters) was first trained for
60 epochs on **single-step next-day prediction only**: the decoder predicted
tomorrow's `OBSERVABLE_FEATURES` conditioned on the abducted latent, the
next day's action, and — critically — the **true** previous-day observable,
with a plain KL penalty (`kl_to_standard_normal`, weight 0.01) pulling the
posterior toward `N(0,I)`.

That run passed Gate 1 and its own Gate 2 decisively (model MSE 0.00375 vs.
persistence 0.11566, 31× better) and then ran Step 3, the abduction test,
producing:

| quantity | value |
|---|---|
| FACTUAL error (model vs. ODE, same actions) | **1.746** |
| COUNTERFACTUAL error (model vs. ODE, flipped actions, abducted latent) | **2.4875** |
| INTERVENTIONAL baseline (model vs. ODE, flipped actions, prior/zero latent) | **2.4875** |
| fraction of the interventional→factual gap closed by abduction | **≈ −0.006%** |

Step 4's sparse-autoencoder probe pointed the same way: best crystal-feature
`|r| = 0.264` vs. best SoC-feature `|r| = 0.577` — SoC more than twice as
recoverable as the hidden state.

### 2.2 Why this was VOID, not a real negative

The giveaway was that the counterfactual and interventional numbers agreed
to four decimal places (2.4875 vs. 2.4875, a gap of −0.00006). That is not
what "the model tried and failed to abduct" looks like; it is what
"the model was handed the same input twice" looks like. Measuring the
posterior directly on that checkpoint, over the 500-window sample used for
the diagnostic, confirmed it:

```
mean |mu|                 0.025028
per-dim std of mu          [0.0322, 0.036, 0.0271, 0.0406, 0.0248, 0.0734, 0.0258, 0.0894]
mean sigma                 0.9926      (prior sigma = 1.0)
```

Every one of these numbers says "prior, not posterior": `mu` averaged
±0.025 around zero with almost no spread across examples (the encoder
produced nearly the same latent for every window, regardless of content),
and `sigma` sat at 0.9926 against a prior of 1.0 — essentially untouched.
The "abducted" latent handed into both the counterfactual and the
interventional rollout in Step 3 **was** the prior/zero vector in both
cases. Their agreement was arithmetic identity, not evidence that abduction
doesn't work.

**Root cause.** A daily-aggregate observable vector is nearly determined by
the previous day's: conditioning the decoder on the TRUE previous-day
observable turns next-day prediction into something close to a one-step
extrapolation the decoder can do almost entirely from `[action_t, prev_obs]`
alone. The model reached MSE 0.00375 that way with an empty latent, and the
KL penalty — with nothing pulling the other way — then had every incentive
to push `q(z|window)` all the way to the prior. Nothing in the pipeline
checked whether the resulting latent carried anything before Step 3 ran.
That missing check is Gate 2b (Section 4).

---

## 3. The fix: multi-step closed-loop training

`experiment/model.py` was updated with two changes (see its module
docstring for the full rationale):

- **`encode_and_rollout()`** — the training-time call now encodes the window
  once, then rolls the SAME `rollout()` path Step 3 evaluates forward
  `ROLLOUT_HORIZON` days in **closed loop**: each day's `prev_obs` is the
  model's own previous prediction, not ground truth. Once day 30 of the
  prediction depends on getting days 1–29 right using only what got packed
  into `z`, the single-step shortcut (read off `prev_obs`, ignore `z`) stops
  being sufficient — the process drifts under model error, and only the
  latent can correct for divergence from the true trajectory.
- **`kl_free_bits()`** — per-dimension KL floored at 0.15 nats
  (Kingma et al., 2016). `torch.clamp(..., min=free_bits)` has zero gradient
  below the floor, so once a dimension's raw KL reaches 0.15 nats the
  optimizer has no further way to push it toward the prior. This is a
  **permanent structural floor**, not a schedule — collapse to `mu=0,
  sigma=1` on any dimension becomes unreachable regardless of how long
  training runs.

**KL scheme actually used, stated plainly: both.** `kl_free_bits` is the
structural fix — the one that makes collapse impossible rather than merely
unlikely. On top of it, `experiment/train_world_model.py` also linearly
ramps the KL weight from 0 up to `KL_WEIGHT = 0.01` over the first
`KL_WARMUP_EPOCHS = 10` epochs, so early training optimizes reconstruction
before any regularization pressure arrives. The warmup is a convenience on
top of the floor, not a substitute for it.

`experiment/train_world_model.py` was rewired end to end:

- **Rollout horizon**: `ROLLOUT_HORIZON = 30` days, read from
  `windows_meta.json`'s `rollout_k` — the exact runway every window was
  built with, and the exact horizon `experiment/abduction.py` rolls out over
  in Step 3. Training and evaluation now agree on what "the rollout" means
  by construction, not by convention.
- **Future targets**: for each window (identified by `scenario` + `end_day`
  in `windows_train.npz` / `windows_test.npz`), the 30 days of
  `ACTION_FEATURES`/`OBSERVABLE_FEATURES` immediately after the window are
  looked up from `daily_trajectory.csv` (`experiment/data.py`), mirroring
  exactly the slice `experiment/abduction.py` already used
  (`arr[f][end+1:end+1+k]`) — the new `build_future_arrays()` helper is
  shared logic Gate 2b (Section 4) reuses too.
- **Loss**: `mse(rollout_prediction, true_future_observables) +
  kl_weight(epoch) * kl_free_bits(mu, logvar, free_bits=0.15)`.
- **Architecture and hyperparameters otherwise unchanged**: 8-dimensional
  latent, 32-unit encoder/decoder hidden sizes, batch size 128, Adam
  lr 1e-3, seed 0, 11,130 parameters total — same "deliberately small"
  model, trained on a harder task. Epoch count raised from 60 to 100
  (multi-step training converges more slowly; each epoch takes well under a
  second on CPU at this scale — full 100-epoch run: ~72 s).

**New Gate 2 result — the multi-step rollout diagnostic — PASS,
decisively.**

| predictor | standardized MSE, 30-day rollout (test) |
|---|---|
| world model | **0.01021** |
| persistence (repeat last observed day for all 30 days) | 0.76849 (75× worse) |
| training-set mean future observable | 1.01023 (99× worse) |

(Diagnostic only, never trained against: the old single-step `forward()`
path scores 0.01420 on this checkpoint — worse than the rollout-trained
model's own multi-step number, confirming the multi-step objective didn't
sacrifice single-step quality to get there.)

Per-feature, the model dominates persistence everywhere, including on
`t_bat_mean`/`t_bat_max` (where the *previous* single-step-trained model had
been marginally worse than persistence) — battery temperature barely moves
day to day, but a model that has to stay on-distribution for 30 self-fed
steps can no longer afford even that small a shortcut.

---

## 4. Gate 2b — the latent-informativeness gate, run before trusting Step 3 again

`experiment/gate2b.py` is the check the first run never had. It tests two
independent ways the pipeline could still be meaningless even with a good
rollout MSE, and both must pass before Step 3 is trusted.

### (a) Is the posterior different from the prior?

Measured on the retrained checkpoint, over all 1500 held-out test windows:

| quantity | value | prior value | verdict |
|---|---|---|---|
| mean \|mu\| | 0.3803 | 0 | clearly nonzero |
| per-dim std of mu | [0.4035, 0.4029, 0.4129, 0.4000, 0.4063, 0.3966, 0.3874, 0.4061] | 0 (if collapsed) | all 8 dims **> 0.1** threshold |
| per-dim mean sigma | [0.7878, 0.7876, 0.7997, 0.7928, 0.7992, 0.7955, 0.7845, 0.7984] | 1.0 | all 8 dims **< 0.9** threshold |
| mean sigma | 0.7932 | 1.0 | clearly below prior |

Every one of these numbers now says the opposite of Section 2.2's collapse
signature: `mu` varies by ~0.40 across examples on every one of the 8
dimensions (versus 0.025–0.089 collapsed), and `sigma` sits at ~0.79–0.80
on every dimension (versus 0.9926 collapsed) — a real, if partial, posterior
contraction relative to the prior. **8/8 dimensions pass both thresholds.
PART A: PASS.**

### (b) Does the decoder actually use the latent it's given?

30-day closed-loop rollout error, same held-out windows, abducted latent
(`mu`) vs. a zeroed latent (`z = 0`, the same prior/zero vector Step 3's
interventional condition uses):

| condition | rollout MSE |
|---|---|
| true abducted latent | **0.01021** |
| zeroed latent | **0.01934** |
| relative degradation from zeroing | **+89.3%** |

Zeroing the latent nearly doubles the rollout error (threshold for a pass
was +15%). The decoder is not ignoring `z`. **PART B: PASS.**

**GATE 2b: PASSED overall.** Both the encoder and the decoder are doing
real work with the latent on held-out data. Step 3 is now safe to run and
trust.

---

## 5. Step 3 — the abduction test, twice: an over-claim, then the correction

### 5.1 First pass: abducted vs. zero-information latent (the over-claim)

Procedure (`experiment/abduction.py`, first version of the fix): for every
one of the 1500 held-out test windows, encode the factual window (`mu`),
roll the decoder forward 30 days under a flipped is-layup schedule, and
compare against the exact ground-truth counterfactual — `CellState` and
`AgingState` rebuilt from the window's true end-of-day resume fields, then
`calendar_sim.roll_days` reruns the *same physics ODE* from that *same real
hidden state* under the flipped schedule. The `INTERVENTIONAL` condition
used the prior mean (a zero vector) as the "no abduction" latent.

| quantity | value |
|---|---|
| FACTUAL error (model vs. ODE, same actions) | **0.01021** |
| COUNTERFACTUAL error (model vs. ODE, flipped actions, abducted latent) | **0.05150** |
| INTERVENTIONAL baseline (model vs. ODE, flipped actions, prior/zero latent) | **0.09022** |
| fraction of the interventional→factual gap closed | **48.4%** |

This was reported as "the model IS abducting, partially": the abducted
latent nearly halved the error relative to a zero latent, and the brief's
decision rule (`counterfactual_mse < interventional_mse` and
`|counterfactual − factual| < |interventional − factual|`) was satisfied.

**Why this was an over-claim, not a second void result.** Unlike the first
run, the posterior here was genuinely non-collapsed (Gate 2b, Section 4)
and the decoder genuinely used it (+89% worse when zeroed). Both of those
checks were real and correctly passed. The mistake was in what the
INTERVENTIONAL control could prove: a zero latent carries *no* information
of any kind, so beating it only shows the abducted latent carries *some*
information — it says nothing about *what kind*. A latent that just encodes
"today's SoC is 61% and battery temp is 34°C" would ALSO beat a zero latent
by a wide margin under a counterfactual schedule, because knowing the
current state obviously helps predict a few weeks forward regardless of
whether any history was ever inferred. Beating zero is necessary for
abduction but nowhere close to sufficient, and reporting it as "genuine,
partial abduction" conflated the two.

### 5.2 The corrected control: last-day-only latent

**The sharper test**: build a control latent from a version of the SAME
window with all path dependence removed, then compare the abducted latent
against THAT, not against zero. Construction used
(`flatten_to_last_day()` in `experiment/abduction.py`): re-encode a window
in which every one of the 60 days is a copy of the window's FINAL day. This
was chosen over the alternative (a small learned map from last-day
observables to latent space) because it holds the architecture, the
encoder, and the per-day information volume completely fixed and changes
only one thing — history. A learned map would confound "what the map
learned to predict" with "what history actually contains," and would need
its own training run with its own capacity and convergence questions. The
flattened-window re-encoding needs neither: it is the same trained encoder,
asked a question its architecture can answer directly, with no new
training. An encoder that reads off "what does the battery look like right
now" gets IDENTICAL input under this construction and under the real
window; an encoder that integrates *how the window got there* gets a
degenerate, historyless one instead. That is exactly the axis this
experiment is about.

**The four numbers** (standardized MSE, averaged over the 30-day rollout
and all 10 observable features, n=1500, same checkpoint, same test
windows):

| quantity | value |
|---|---|
| FACTUAL (model vs. ODE, same actions) | **0.01021** |
| COUNTERFACTUAL abducted (flipped actions, full-window latent) | **0.05150** |
| COUNTERFACTUAL lastday (flipped actions, last-day-only latent) | **0.05828** |
| INTERVENTIONAL zero (flipped actions, prior/zero latent) | **0.09022** |
| fraction of factual→lastday gap closed by the abducted latent | **14.1%** |
| fraction of factual→zero gap closed (the weak control, kept for continuity) | 48.4% |
| per-example win rate, abducted vs. lastday | **698/1500 = 46.5%** |

**The headline number collapses.** Against the correct control, the
abducted latent closes only 14.1% of the gap — well under a third of the
48.4% the weak zero control implied. And the per-example win rate is
**46.5%, below the 50% chance line**: on more than half of the 1500 test
windows, the last-day-only control actually produced a LOWER error than the
full-window abducted latent. A mean that favors "abducted" while the median
individual comparison favors "lastday" is the signature of a small number
of high-error outliers dragging the average — not a consistent per-battery
effect. `experiment/abduction.py`'s verdict logic was corrected to require
`win_rate_vs_lastday > 0.55` (clearly above chance) in addition to the mean
comparison, precisely so this failure mode cannot pass silently again; on
this run it correctly reports **"NOT ABDUCTION -- AGGREGATE IS NOISE."**

**The crystal-bin breakdown makes the case, not just the aggregate.** If the
abducted latent were recovering genuine path-dependent information, it
should help MOST in the middle of the crystal transition — where the
hidden state is changing but SoC and temperature still look like an
ordinary battery, i.e. exactly where the last-day control has no way to see
what is different. That is not what the data shows:

| true crystal bin | n | factual | abducted | lastday | gap closed | win rate (abducted vs. lastday) |
|---|---|---|---|---|---|---|
| [0.0, 0.2) | 300 | 0.0070 | 0.0435 | 0.0534 | 21.3% | **40.3%** |
| [0.2, 0.4) | 300 | 0.0092 | 0.0431 | 0.0500 | 16.9% | **39.0%** |
| [0.4, 0.6) — mid-transition | 300 | 0.0097 | 0.0556 | 0.0607 | 10.0% | 48.0% |
| [0.6, 0.8) — mid-transition | 300 | 0.0090 | 0.0580 | 0.0672 | 15.7% | 48.0% |
| [0.8, 1.0) | 300 | 0.0162 | 0.0573 | 0.0602 | 6.6% | **57.3%** |

Win rate is *lowest* exactly where crystal is low (39–40%, i.e. the
abducted latent loses to lastday MORE often there) and *highest* where
crystal is saturated (57.3%) — the reverse of the mid-transition bump a
genuine path-dependent signal would produce, and the two mid-transition
bins sit almost exactly at chance (48.0% both). There is no bin where the
win rate is convincingly above 50%. Nothing in this breakdown supports
"the model recovers more path-dependent signal where the hidden state
diverges most from what the last day shows" — if anything, the pattern is
mildly the opposite.

**Bottom line for Step 3, corrected: the model is NOT abducting the hidden,
path-dependent crystal state.** The counterfactual improvement reported in
Section 5.1 is real but fully explained by cheap, currently-observable
state — the same information a last-day snapshot already contains — not by
any inference about the battery's history. This reverses the 5.1 finding,
and the reversal came entirely from tightening the control, not from
retraining or retuning the model.

---

## 6. Step 4 (rerun) — sparse autoencoder probe

Same procedure as the first run (`experiment/sae_probe.py`, unmodified): an
8→32 (4× overcomplete) linear sparse autoencoder (ReLU code, L1 penalty
1e-3), trained for 200 epochs on training-window posterior means, evaluated
on test-window codes.

**Sparsity achieved**: mean 15.2 of 32 features active per example (47.5%),
2 dead features.

**Best single-feature \|Pearson r\| against each target** (test set,
n=1500):

| target | best \|r\| | feature # |
|---|---|---|
| **crystal** (hidden, path-dependent — the thing we want) | **0.168** | 8 |
| **soc** (observable, trivially available) | **0.155** | 10 |
| t_bat_mean, window's last day (observable) | 0.256 | 8 |
| corrosion_hours (technically hidden, but ≈ a function of ambient × elapsed time) | 0.129 | 20 |

**Interpretation, stated plainly — and it now agrees with the corrected
Step 3, not just the over-claimed one.** The top crystal-tracking feature
(r=0.168) is *nominally* higher than the top SoC-tracking feature (r=0.155),
but `experiment/sae_probe.py`'s own comparison rule requires a **+0.05**
margin before calling that a clear win, and the actual gap is only +0.013 —
correctly "NO BETTER," not "clearly better." More tellingly, the top feature
for `t_bat_mean` (last-day battery temperature, a trivially observable
quantity) scores **higher than crystal** (0.256 vs. 0.168) — the single best
predictor found anywhere in this probe is a piece of currently-observable
state, not the hidden target. That is exactly consistent with Section 5.2's
finding that a last-day-only latent gets most of the way to the full
window's counterfactual performance: whatever the SAE is finding easiest to
extract linearly is the same kind of information the lastday control
already has by construction.

This was read, in the earlier (over-claiming) pass of this report, as a
puzzle — "Step 3 says the latent helps, Step 4 says no single feature
clearly tracks the hidden state, so the information must be distributed
across dimensions rather than sparse." That reading is no longer needed.
Once Step 3 itself is corrected (Section 5.2), there is no tension left to
explain: **both tests agree** that the latent's useful content is dominated
by cheap, currently-observable quantities (temperature most of all, then
SoC), and neither test finds convincing evidence that it is the
specifically hidden, path-dependent crystal state.

**Bottom line for Step 4, read together with the corrected Step 3: no
single interpretable feature "is" the crystal tracker, crystal-tracking is
not clearly better than SoC-tracking, and the single best-tracked target of
all is a directly observable one (battery temperature).** This is not a
tempering caveat on a positive Step 3 result anymore — it is independent
confirmation of the same negative finding.

---

## 7. The counterfactual-training curriculum (round 4) — does the obvious fix work?

Round 3 (Section 5.2) ended on a clean negative and one concrete hypothesis
for why: the model NEVER saw anything but the FACTUAL action continuation
during training (`experiment/train_world_model.py`). Under factual-only
training, "what state resulted" and "what happens next" are always
correlated with the real schedule, so nothing forces the latent to keep
information that is only useful under a schedule that did not happen. This
section tests that hypothesis directly, honestly reporting whatever came
out — which turned out to be neither a clean yes nor a clean no.

### 7.1 Method: train on ODE-true counterfactuals, not just the real trajectory

`experiment/build_counterfactual_targets.py` is new. For every one of the
6000 TRAIN windows (never touching the held-out test scenarios), it:

1. Draws a **randomized** counterfactual is-layup schedule
   (`random_perturbation_day_types`): each of the 30 future days
   independently flipped relative to the factual schedule with probability
   0.5. This is deliberately a DIFFERENT perturbation shape from Step 3's
   deterministic full flip (`build_counterfactual_actions`) — training
   augmentation is never literally "train on the eval transformation," so
   any generalization Step 3 measures is to a genuinely unseen kind of
   intervention, not memorization of one.
2. Reruns the exact physics ODE (`calendar_sim.roll_days`, via
   `experiment.abduction.ground_truth_counterfactual`) from the window's
   TRUE resumed hidden state under that randomized schedule — the same
   generator-owned trick the eval test uses, now used to manufacture a
   training signal instead of only a test one.
3. Saves the result as `windows_train_cf.npz` (cf_actions, cf_observables),
   row-aligned with `windows_train.npz`.

This took about 20 minutes for all 6000 windows (dominated by the ODE
reruns, not model computation) and is a one-time preprocessing cost.

`experiment/train_world_model.py` then trains a dual-rollout objective: one
`encode()` call, one sampled `z`, and TWO `rollout()` calls from that SAME
`z` — one under the factual continuation (against its true target) and one
under the randomized counterfactual continuation (against ITS true target,
from the ODE rerun). The loss is `recon_factual + CF_LOSS_WEIGHT *
recon_counterfactual + kl_weight * kl_free_bits(mu, logvar)`, with
`CF_LOSS_WEIGHT = 1.0` (equal weighting, so the model cannot satisfy the
objective by improving only the factual branch and coasting on the
counterfactual one). Everything else — architecture, latent dimension,
`kl_free_bits` floor, KL warmup, 100 epochs, batch size, seed — is
unchanged from round 3, so any difference in outcome is attributable to the
curriculum, not a confound from retuning something else at the same time.

### 7.2 New Gate 2 / Gate 2b: the model still learns the dynamics, and the latent is still used

| predictor | standardized MSE, 30-day rollout (test) | round 3 |
|---|---|---|
| world model (factual) | **0.00556** | 0.01021 |
| persistence | 0.76849 | 0.76849 |
| train mean | 1.01023 | 1.01023 |

Factual-rollout accuracy roughly DOUBLED versus round 3 (0.00556 vs.
0.01021) despite the model now also having to satisfy a second,
counterfactual objective from the same latent — the dual task did not cost
factual accuracy, it improved it, consistent with the curriculum forcing a
more informative `z` rather than just adding noise.

**Gate 2b — PASSED, more decisively than round 3:**

| quantity | round 4 | round 3 |
|---|---|---|
| mean \|mu\| | 0.3824 | 0.3803 |
| mean sigma | 0.7229 | 0.7932 |
| relative rollout degradation when latent zeroed | **+404.5%** | +89.3% |

Every dimension still clears both thresholds (8/8), and the decoder now
leans on the latent far more heavily than before. One dimension in
particular (index 2) shows a much sharper posterior contraction than the
rest (std of mu 0.978 vs. 0.28–0.41 on the other seven dimensions; mean
sigma 0.42 vs. 0.74–0.80 elsewhere) — a plausible early signature of a
latent dimension specializing on something the others don't carry, though
Gate 2b does not by itself say what.

### 7.3 Step 3, re-run against the SAME lastday control, with a proper significance test

Round 3's verdict logic used a fixed win-rate threshold (>55%) as a
stand-in for "clearly above chance." Round 4's result (53.6%) exposed why
that was too blunt an instrument: 53.6% is close to the 55% cutoff in
absolute terms but, at n=1500, is **statistically significant** (a
two-sided binomial test against 50% gives p=0.0053 — not noise). So
`experiment/abduction.py` was corrected to use a proper significance test
(`binom_test_pvalue`, a stdlib-only normal-approximation binomial test)
instead of an arbitrary threshold, applied both to the aggregate win rate
and to each crystal bin, so a real-but-small aggregate effect and a
real-but-non-uniform one are no longer conflated.

**The four numbers** (standardized MSE, n=1500, same held-out test
scenarios, same lastday control construction as round 3):

| quantity | round 4 | round 3 |
|---|---|---|
| FACTUAL | 0.00556 | 0.01021 |
| COUNTERFACTUAL abducted | 0.01064 | 0.05150 |
| COUNTERFACTUAL lastday | 0.03973 | 0.05828 |
| INTERVENTIONAL zero | 0.09304 | 0.09022 |
| fraction of factual→lastday gap closed | **85.2%** | 14.1% |
| win rate, abducted vs. lastday | **804/1500 = 53.6%** | 46.5% |
| p-value (two-sided binomial test vs. 50%) | **0.0053** | (not computed; round 3's 46.5% was below chance, no test needed to call it non-abducting) |

The mean gap-closed number jumped from 14.1% to 85.2%, and — unlike round
3 — the win rate is now both above chance AND statistically significant.
By the letter of round 3's own logic (mean improvement + win rate above
50%), this would read as a clean positive. It is not, once the crystal-bin
breakdown is examined.

**Crystal-bin breakdown, with per-bin significance:**

| true crystal bin | n | win rate | p-value | verdict |
|---|---|---|---|---|
| [0.0, 0.2) | 300 | 39.0% | **0.0001** | **significantly favors LASTDAY** — abducted latent does worse |
| [0.2, 0.4) | 300 | 51.7% | 0.564 | not significant |
| [0.4, 0.6) — mid-transition | 300 | 54.7% | 0.106 | not significant |
| [0.6, 0.8) — mid-transition | 300 | 52.7% | 0.356 | not significant |
| [0.8, 1.0) — saturated | 300 | **70.0%** | **<0.0001** | strongly significant, favors abducted |

**Interpretation, stated plainly.** This is neither round 3's clean
negative nor a clean positive. Three things are true simultaneously:

1. **The aggregate effect is real, not noise** (p=0.0053) — a meaningful
   change from round 3, where the win rate was actually below chance.
2. **The effect is not uniform across the crystal spectrum.** It is
   strongly concentrated in the SATURATED-crystal bin (win rate 70.0%,
   p<0.0001) — the model becomes clearly, robustly better than the
   lastday control specifically for batteries whose hidden state has
   fully saturated. In the LOWEST-crystal bin, the effect is not just
   absent but **significantly reversed** (39.0%, p=0.0001): the abducted
   latent does measurably WORSE than a control with no history at all.
   The two mid-transition bins — where a genuinely path-dependent signal
   would be expected to help most, per the reasoning in round 3's
   breakdown — show no significant effect either way.
3. **This is the opposite shape from what "clean, general path-dependent
   abduction" would predict**, but it is also not nothing. A plausible
   mechanistic reading: near saturation, the window's full 60-day shape
   (e.g., an extended run of park-heavy or drive-heavy days) may carry a
   genuinely path-dependent signature that the counterfactual curriculum
   taught the model to use — this is still "history that the last day does
   not show," even if it is coarser than full per-day path-dependence.
   Near zero crystal, the counterfactual curriculum may instead have
   introduced a *distraction*: perturbed schedules the model was trained
   on there may not resemble anything informative about a state that has
   barely moved from its start, and the extra objective made the latent
   WORSE at exactly the state-conditioning task the lastday control is
   already good at. Both readings are plausible; distinguishing them was
   not attempted here and is listed in Section 9.

### 7.4 Step 4, re-run: the SAE probe now shows a real crystal signal

| target | round 4 best \|r\| | round 3 best \|r\| |
|---|---|---|
| **crystal** | **0.628** (feature #18) | 0.168 |
| soc | 0.541 (feature #24) | 0.155 |
| t_bat_mean, last day | 0.704 (feature #22) | 0.256 |
| corrosion_hours | 0.171 (feature #26) | 0.129 |

Sparsity: 32 codes, mean 16.35 active per example (51.1%), zero dead
features.

Crystal (0.628) now clearly beats SoC (0.541) by margin 0.087 — comfortably
past the script's required +0.05 threshold, the opposite of round 3's
result and a genuinely large jump in absolute correlation for both. This is
independent, representational evidence (a linear probe on the latent, not
the decoder's behavior) that SOMETHING about crystal specifically, beyond
what SoC gives away, is now present in the latent — consistent with Step
3's finding of a real (if localized) effect. Temperature (0.704) remains
the single best-tracked target of all, so the latent is still dominated by
observable state overall; crystal has gone from "clearly the weakest
tracked target" to "second-best tracked, and clearly ahead of SoC," which
is a substantial, not marginal, change.

### 7.5 Verdict for round 4

**A real, statistically significant, but localized and non-uniform
signal.** The counterfactual-training curriculum measurably improved the
model's ability to use history beyond the current observable state — both
behaviorally (Step 3's aggregate win rate, SAE-independent) and
representationally (Step 4's crystal-vs-SoC correlation gap). But it did
not produce general path-dependent abduction: the effect is concentrated in
one crystal regime (saturated), reversed in another (near-zero), and absent
in the middle two of five bins — including both bins where a genuinely
general path-dependent signal would be expected to matter most. The honest
characterization is "partial, localized abduction in a specific regime,"
not "the fix worked" or "the fix didn't work."

---

## 8. Honest verdict on the research question

**Partially, and only in a specific regime — not the clean "yes" or "no"
either earlier round suggested.** With naive (factual-only) multi-step
training, this world model does NOT perform abduction of the hidden,
path-dependent crystal state: when tested against a control that holds
architecture and information volume fixed and removes only history (Section
5.2), the model's apparent counterfactual ability was fully explained by
conditioning on cheap, currently-observable state (SoC, temperature) — a
properly controlled negative, not an artifact.

**Adding a counterfactual training curriculum changes this, but not into a
clean positive.** Training the SAME encoded latent to correctly predict
BOTH the factual continuation and an exact-ODE-ground-truth counterfactual
one (Section 7) produced a statistically real behavioral effect (win rate
53.6% vs. lastday, p=0.0053) and a substantially strengthened
representational one (SAE crystal correlation 0.168 → 0.628, now clearly
ahead of SoC). But the crystal-bin breakdown shows this is **concentrated
in the saturated-crystal regime** (70.0% win rate, p<0.0001) and
**significantly reversed at low crystal** (39.0% win rate, p=0.0001) — the
opposite of, and more localized than, what a general path-dependent
recovery mechanism would produce. The two mid-transition bins, where
general path-dependent information would matter most, show no significant
effect in either direction.

**What the model most likely does, stated as the best-supported reading of
all four rounds together**: for a battery whose hidden state has NOT moved
far from its starting point (low crystal), the model still substantially
relies on cheap, currently-observable state, and the counterfactual
curriculum did not help — if anything it may have introduced noise the
lastday control doesn't have to deal with. For a battery near full
saturation, something about the window's longer-run SHAPE (not
necessarily fine-grained day-by-day path-dependence, but some
coarser signature — e.g., how much of the 60-day window was spent parked
vs. driving) appears to be recoverable and useful in a way the last day's
snapshot cannot provide, and the counterfactual curriculum taught the model
to extract and use it. That is a real, if narrower and coarser, form of
history-sensitivity than "abduction of the hidden crystal variable in
general" — closer to "abduction of one coarse feature of recent history,
usable specifically when the hidden state is already near an extreme."

**Why this result, unlike the first two, is trustworthy as reported.** The
void run's problem was that nothing was being tested at all (collapsed
posterior). The round-3 negative was a genuinely well-controlled result
that held up to scrutiny. This round-4 result is neither a repeat of either
failure mode: Gate 2b passes more decisively than before (+404.5% rollout
degradation when the latent is zeroed), the significance test is principled
(a two-sided binomial test, not an arbitrary threshold — see Section 7.3 for
why the threshold approach itself needed correcting), and the same test
that could have shown a clean positive (all five bins significant in favor
of abducted) instead honestly returned a mixed one. Nothing was tuned to
produce this outcome; the curriculum fix was implemented once, from the
concrete hypothesis in round 3's "what I would do differently," and
reported as it came out.

**A fair one-line summary: the widely-assumed capability — that training a
world model to predict counterfactual continuations gives it general
path-dependent abduction — does not simply appear once you add the obvious
training fix.** It produces a real but narrow, regime-specific effect, and
claiming general abduction from this result would be exactly the kind of
over-claim round 2 already made and had to walk back.

### The methodological lesson, stated for all three mistakes avoided along the way

1. **A clean-looking counterfactual-vs-baseline agreement (or disagreement)
   is not evidence about abduction until the posterior and the decoder have
   both been checked for whether either one is even using the latent** —
   this is what Gate 2b now enforces permanently, and it is what the first,
   void run skipped.
2. **A "no information" baseline (zero latent) is too weak a control for
   claiming path-dependent abduction specifically** — beating it only shows
   the latent carries *some* information, not the *right kind*. The control
   that matters is one that holds architecture and information volume fixed
   and removes only history, which is what Section 5.2's last-day-only
   latent does. The over-claimed second run skipped this, and its "genuine,
   partial abduction" verdict did not survive the sharper control.
3. **A fixed win-rate threshold (round 3's ">55%") is itself a weaker
   instrument than an actual significance test, and can mislead in EITHER
   direction** — round 4's 53.6% sat just under the threshold but was
   nonetheless statistically real (p=0.0053), and a threshold-only check
   would also have missed that the aggregate effect was concentrated in one
   crystal regime and reversed in another. A binomial test on the aggregate
   AND on each stratified subgroup is what actually distinguishes "real but
   small," "real but localized," and "not real" — three outcomes a single
   threshold collapses into two.

## 9. What I would do differently

1. ~~**Train the decoder on multi-step rollouts, not just one-step.**~~ Done
   (Section 3) — necessary to get a non-collapsed latent at all, though not
   sufficient on its own to demonstrate genuine abduction.
2. ~~**Add a gate for whether the latent carries information the decoder
   uses.**~~ Done (Gate 2b, Section 4) — catches the collapse failure mode
   but does not by itself catch a real, non-path-dependent latent.
3. ~~**Build the last-day-only control into the pipeline.**~~ Done (Section
   5.2) and reused unchanged for round 4 (Section 7.3) — it is now the
   permanent headline comparison, not a one-off check.
4. ~~**A curriculum that trains under perturbed, not just factual, action
   continuations.**~~ Done (Section 7) — the single highest-leverage item
   from this list, and the one that turned a clean negative into a real but
   localized, non-uniform effect. This is also the clearest lesson of the
   whole report: implementing the "obvious fix" did not produce the clean
   positive a first guess might expect, and reporting THAT honestly (rather
   than stopping at "the win rate went up") is what surfaced the
   regime-specific structure in Section 7.3's bin breakdown.
5. **Explain the low-crystal reversal, not just report it.** Section 7.5
   offers a plausible mechanism (the counterfactual curriculum may act as a
   distraction where the hidden state has moved little) but this was not
   tested. A direct check: train a variant where the counterfactual loss is
   down-weighted or held out specifically for low-crystal windows, and see
   whether the reversal in that bin disappears — this would distinguish
   "the curriculum actively hurts low-crystal state-conditioning" from "the
   low-crystal bin was always going to be near-chance and round 4 just
   drew an unlucky significant sample" (against which: p=0.0001 argues for
   a real effect, not luck, but not for which mechanism).
6. **Identify what the saturated-crystal bin's signal actually IS.** Section
   7.5 speculates it is a coarse window-shape feature (e.g. fraction of days
   parked) rather than fine-grained path-dependence. This is directly
   testable: hold out `driving_minutes`/`soak_hours` summary statistics of
   the window (mean, variance, longest run) as auxiliary targets and see how
   much of the saturated-bin win rate a model conditioned on THOSE (but
   still no full sequence) can recover — if most of it, the "coarse
   window-shape, not fine path-dependence" reading is supported directly
   rather than left as a plausible story.
7. **A supervised probe on the full latent, not just single SAE features**
   (e.g. ridge regression of crystal on the full 8-dim `mu`) — round 4's SAE
   result (crystal r=0.628, clearly beating SoC) already answers the
   "is it distributed vs. sparse" question round 3 left open, but a full
   linear probe would give an upper bound on how much of crystal is
   linearly recoverable at all, useful context for interpreting 0.628.
8. **A wider SAE sweep** (several L1 weights and expansion factors) rather
   than one fixed setting, now that there is a real, non-trivial crystal
   signal worth resolving more precisely (round 3 had no such motivation).
9. **A held-out validation split from the training scenarios**, used for
   early stopping — training here ran a fixed 100 epochs with no
   over/under-fitting check. Gate 2's large margins in both rounds 3 and 4
   make this unlikely to change the headline conclusion, but it remains a
   real gap in the training methodology as run.
10. **More than one counterfactual "shape" per test window**, and per
    training window. Both the eval flip (deterministic, full-window) and
    the training augmentation (randomized, per-day, p=0.5) are single fixed
    designs. Testing a gentler counterfactual (e.g., shift a single 10-day
    layup by 200 days) against the SAME lastday control, and training with
    a range of perturbation intensities rather than one fixed p=0.5, would
    show whether round 4's regime-specific pattern is a general property of
    counterfactual-curriculum training or an artifact of these two specific
    perturbation designs.

Items 1–4 were completed in this pass and are load-bearing for the report's
final verdict; nothing was done to rescue a positive OR a negative finding
— round 4's result was reported exactly as it came out, including that it
supports neither of the two cleaner stories (general abduction, or none at
all) that would have made a tidier headline.

---

## 10. Round 5 — an RSSM in place of the frozen latent (2026-09-15)

### 10.1 Why, and what was held fixed

Rounds 1–4 used one architecture ("gru_vae"): a GRU encoder squeezes the
60-day window into a single 8-dim latent `z`, frozen for the whole 30-day
rollout. Sulfation is not frozen; it builds day by day. Round 5 replaces it
with a Recurrent State-Space Model (PlaNet/Dreamer; `experiment/rssm.py`):
a deterministic GRU path `h_t` plus a stochastic state `s_t`, a learned prior
`p(s_t | h_t)` for imagination and a posterior `q(s_t | h_t, o_t)` for
filtering. Abduction is filtering over the window; counterfactual prediction
is imagining forward from the final belief `[h_60, s_60]`.

Everything except architecture was held identical to round 4
(`experiment/train_rssm.py` imports gru_vae's constants): same windows, split,
standardisation, counterfactual curriculum, 30-day horizon, KL weight 0.01
with 10-epoch warmup, free bits (1.2 nats per day = 0.15 x 8 dims), batch
128, lr 1e-3, 100 epochs, seed 0. RSSM additions: KL balancing 0.8 and
gradient clipping at 100. Parameters: 10,914 vs gru_vae's 11,130.
Success criteria were written down before the run
(`docs/superpowers/specs/2026-09-15-rssm-world-model-design.md`), and nothing
was retuned after seeing test results.

All evaluation scripts now take `--arch`; gru_vae keeps its filenames and the
RSSM writes `rssm_*`. Rerunning gate2b and the SAE probe on gru_vae through
the refactored code reproduced the saved results exactly.

A new Step 5, `experiment/calibration.py`, measures what the output contract
actually validates: how often a claimed 90% interval (100 sampled futures,
central band) contains the true factual 30-day continuation.

### 10.2 Results, one run each

| | gru_vae (round 4) | RSSM (round 5) |
|---|---|---|
| Gate 2 rollout MSE | **0.00556** | 0.00821 |
| Gate 2b: error increase when latent zeroed | 404% | 4710% |
| Step 3 win rate vs last-day control | 53.6% (p=0.005) | **69.7% (p≈1e-52)** |
| Crystal bins, win rate | 39 / 52 / 55 / 53 / 70% | **75 / 64 / 65 / 74 / 71%** |
| Any bin significantly favouring last-day | yes, [0.0,0.2) | **none** |
| Counterfactual / factual error | 1.91 | **1.60** |
| Step 3 verdict tier | significant, mixed | **genuine path-dependent** |
| Step 4 best \|r\| crystal / SoC | 0.63 / 0.54 | 0.62 / 0.95 |
| Step 5 coverage of claimed 90% | 47.0% | **62.9%** |
| Step 5 interval width, day 1 → day 30 | 0.095 → 0.096 | 0.104 → 0.142 |

By every criterion set in advance, the RSSM passes where gru_vae did not:
its own abducted belief beats its own last-day control in every crystal bin,
including the low-crystal bin where gru_vae significantly lost.

### 10.3 What that does NOT show — the head-to-head

The last-day control is internal to each model, so a higher win rate can
come from a better abducted state OR a worse control. Comparing the two
models directly, per test window, against the same ODE ground truth:

| Error | gru_vae | RSSM | RSSM lower on |
|---|---|---|---|
| Factual | 0.00556 | 0.00821 | 37.7% of windows |
| Counterfactual (abducted) | **0.01064** | 0.01316 | 43.0% |
| Counterfactual (last-day) | 0.03973 | 0.04977 | 35.5% |

| Crystal bin | cf gru_vae | cf RSSM | RSSM lower on |
|---|---|---|---|
| [0.0,0.2) | 0.0066 | 0.0114 | 30% |
| [0.2,0.4) | 0.0086 | 0.0127 | 43% |
| [0.4,0.6) | 0.0093 | 0.0147 | 33% |
| [0.6,0.8) | 0.0115 | 0.0123 | 42% |
| [0.8,1.0) | 0.0171 | **0.0147** | **67%** |

**gru_vae still predicts counterfactuals more accurately in absolute terms**,
everywhere except heavily sulfated batteries. Both readings are true at once:
the RSSM relies on history far more (its last-day control collapses harder,
and zeroing its state is catastrophic), but it is a less accurate forecaster
overall.

### 10.4 A likely confound: training instability

RSSM training was smooth to epoch 71 (train factual MSE ~0.0026), then spiked
repeatedly (epoch 72: 0.0089; 79: 0.0065; 91: 0.0065) with the KL term rising
from ~2.4 to ~3.4 nats. The final epoch landed on a spike (0.0056, against a
best of 0.0023 at epoch 76), and that checkpoint is the one evaluated. The
absolute-error gap in 10.3 is therefore partly, possibly mostly, an
optimisation artifact, not an architectural limit. This is not established:
it is the hypothesis a follow-up run would test.

### 10.5 Step 4 is not a like-for-like comparison

The RSSM's abducted state is 44-dim (`h` 36 + `s` 8) against gru_vae's 8, and
its deterministic path encodes current state directly (SoC r=0.95). The
probe's "crystal must beat SoC" rule was designed for a small latent that
had to choose what to keep; a larger state keeps both. The RSSM's crystal
r (0.62) is unchanged from gru_vae's (0.63), so the probe neither supports
nor contradicts 10.2. A ridge probe on the full state (section 9, item 7)
would be the fairer test.

### 10.6 Calibration

Both models are overconfident; the RSSM less so (62.9% vs 47.0% of a claimed
90%), and only its intervals widen with horizon (+37% from day 1 to 30,
against +1% for gru_vae). Neither meets the output contract. The gru_vae
result is structural: one posterior draw of `z` feeds a deterministic
rollout, so it has no way to express growing uncertainty.

### 10.7 Verdict for round 5

The architecture change did what it was predicted to do on the question this
experiment asks: the RSSM's counterfactuals depend on the window's history,
uniformly across sulfation regimes, which gru_vae's did not. It did not
produce a better forecaster. The honest claim is "a day-by-day belief state
recovers path dependence that a frozen latent does not", not "the RSSM
predicts counterfactuals better". Neither model is calibrated.

Next, in order:
1. Fix the instability (validation split from training scenarios with
   best-checkpoint selection, section 9 item 9; and/or learning-rate decay),
   rerun both models under it, and repeat 10.2–10.3. Disclosed as a change
   made after seeing round 5's test results.
2. Multiple seeds for both architectures: every number here is one run.
3. Ridge probe on the full state for both models (10.5).
4. Calibration: sample-based interval widening is necessary but not
   sufficient; a post-hoc conformal adjustment on a validation split is the
   cheap route to the claimed coverage.

---

## 11. Round 6 — sensor noise, an unseen hot climate, best-epoch selection (2026-09-15)

### 11.1 Why, and what changed

A leakage/overfitting audit after round 5 found neither: train and test
error matched, no scenario crossed the split, a nearest-neighbour copy of the
closest training window scored 77x worse than gru_vae, and a same-battery
day-type lookup 47x worse. The accuracy was still flattering, because the data
was exact ODE output and the test batteries came from the same four climates
as training. Round 6 (`docs/superpowers/specs/2026-09-15-noisy-heldout-climate-design.md`):

- **Noise:** `cell/sensors.py` sigmas on measurable daily channels only
  (voltage 0.010 V, current max(0.1 A, 0.5%), battery temperature 0.5 C);
  SoC and bay temperature clean, as sensors.py classes them latent.
  Realised noise matched spec to 3 significant figures.
- **Unseen climate:** all 20 batteries at 42 C form the test set; train is
  51 batteries and validation 9, from -10/10/25 C. Every test day's battery
  temperature lies above the training maximum (29.6 C mean).
- **Best epoch** by validation 30-day rollout error, identical rule for both
  models. gru_vae: epoch 78. RSSM: epoch 80. The RSSM's late instability
  recurred (val error rose from 0.0042 at epoch 80 to 0.0066 at 99); selection
  avoided it.

Output: `runs/experiment_hot_noisy/`. Hyperparameters unchanged from round 5.

### 11.2 In-distribution vs unseen climate

| | Validation (training climates) | Test (unseen 42 C) |
|---|---|---|
| gru_vae 30-day rollout MSE | 0.0058 | **0.150** (26x) |
| RSSM 30-day rollout MSE | 0.0042 | **0.147** (35x) |

Both models fail to extrapolate. Per feature, gru_vae on temperature is
WORSE than repeating the last day (standardized 0.22 vs persistence 0.005).
The errors are a bias, not scatter: gru_vae forecasts 37.1 C battery
temperature where the truth is 43.4 C -- after seeing 43.4 C on the window's
last day -- overstates SoC by 3.7 points, and predicts a cranking-voltage dip
0.77 V too mild. The models regress toward their training climates.

### 11.3 Real-unit accuracy, factual 30 days, against clean truth

| | gru_vae r4 (clean, seen climates) | gru_vae r6 | RSSM r6 |
|---|---|---|---|
| Battery temperature MAE | 0.32 C | 6.25 C | **5.58 C** |
| Bay temperature (mean) MAE | 0.55 C | 4.22 C | **4.05 C** |
| SoC MAE | 1.2 pts | 3.8 pts | **3.4 pts** |
| Lowest voltage MAE | 0.12 V | 0.77 V | **0.67 V** |
| Mean voltage MAE | 0.015 V | **0.045 V** | 0.050 V |
| Mean current MAE | 0.17 A | **0.51 A** | 0.60 A |

Temperature R^2 on the 42 C test is strongly negative for both (-107 and
-86): the within-climate spread is 0.6 C, so a 5-6 C offset dwarfs it.

### 11.4 Counterfactuals on the unseen climate — the round's main result

| | gru_vae | RSSM |
|---|---|---|
| Factual MSE | 0.1496 | 0.1467 |
| Counterfactual (abducted) MSE | 0.2827 | **0.1438** |
| Counterfactual / factual | 1.89 | **0.98** |
| Last-day control MSE | 0.3591 | 0.1953 |
| Win rate vs own last-day control | 62.5% | 57.5% |
| Bins significantly favouring last-day | none | none |

Head-to-head per test window, same ODE ground truth:

| Error | RSSM lower on |
|---|---|
| Factual | 56.9% (p≈8e-8) |
| Counterfactual (abducted) | **96.5% (p≈1e-283)** |
| Counterfactual (last-day) | 96.1% |

On an unseen climate the RSSM's counterfactual predictions are twice as
accurate as gru_vae's, on nearly every battery, and no worse than its own
factual predictions. gru_vae's counterfactual error roughly doubles its
factual error, exactly as in rounds 4-5. This reverses round 5's head-to-head
(10.3), where gru_vae was more accurate in absolute terms on seen climates
with a last-epoch RSSM checkpoint. Two changes separate them -- best-epoch
selection and the climate shift -- and this round does not isolate which
matters more.

Both models' win rates against their own last-day control are significant
and uniform across crystal bins. As 10.3 showed, that statistic is internal
to each model and does not rank them; the head-to-head does.

### 11.5 Latent checks and calibration

| | gru_vae r6 | RSSM r6 |
|---|---|---|
| Gate 2b: error increase when latent zeroed | 42% | 242% |
| SAE best \|r\| crystal / SoC | 0.35 / 0.76 | **0.56** / 0.92 |
| 90% interval coverage | 10.8% | 19.8% |
| Interval width day 1 -> 30 | 0.150 -> 0.103 | 0.131 -> 0.198 |

Calibration collapses under extrapolation for both: a biased forecast with
honest-looking spread misses the truth almost every time. Only the RSSM's
intervals widen with horizon. Neither is usable as the output contract's
"plan for" figure on an unseen climate.

### 11.6 Verdict for round 6

1. **Neither model generalizes to an unseen hot climate.** Errors grow
   25-35x, driven by a systematic cold bias in temperature. For this project
   that is the most important finding: 42 C is exactly the condition the
   deployed model has to handle, and a model trained without hot data will
   under-predict heat and over-predict SoC.
2. **Under that shift the RSSM's counterfactuals are far more robust**: half
   the error, lower on 96.5% of batteries, with no penalty for changing the
   schedule. This is the strongest architecture result so far.
3. **Still one seed per model, and noise on daily aggregates is a
   simplification** (11.1).

Next, in order:
1. Train on all four climates with the same noise and best-epoch rule, and
   test on held-out batteries, to separate "noise + best epoch" from
   "climate shift" in 11.4.
2. Multiple seeds for both models on both setups.
3. Physics-informed temperature: the bay/battery temperature offset from
   ambient is what extrapolates, not the absolute value. Predicting the
   offset (or conditioning the decoder on ambient additively) is the obvious
   structural fix for 11.2's bias.
4. Drop SoC and bay temperature from the model inputs, per `cell/sensors.py`.

---

## 12. Round 7 — generalizing to the unseen climate (2026-09-16)

### 12.1 The diagnosis that set the design

After round 6, two no-learning baselines were scored on the 42 C test
(tolerance accuracy: within +/-0.05 V, +/-0.5 A, +/-1 C, +/-2 SoC points,
against clean truth): persistence 68.6%, and **same-day-type** -- each future
day equals the latest window day with the same drive/layup type -- **83.3%**.
Both round 6 models (~30%) were far below copying. Temperatures as offsets
above ambient, unlike absolute temperatures, lay inside the training range.
Spec: `docs/superpowers/specs/2026-09-16-unseen-climate-generalization-design.md`.

### 12.2 Changes

- **Anchored residuals** (`--anchored`): predict `anchor + correction`, anchor
  = the same-day-type baseline computed from the observed window and the
  action sequence asked about (counterfactual schedules get counterfactual
  anchors). Output layer zero-initialised, so an untrained model reproduces
  the baseline. The RSSM's window reconstruction anchors on earlier window
  days only.
- **Ambient offsets** (`--offsets`): the four temperature observables as
  `value - ambient_c`, converted back for scoring.
- `experiment/pipeline.py` is the single data path for both training scripts
  and `experiment/accuracy.py`; with both flags off it reproduces rounds 4
  and 5 exactly (epoch-0 losses match to four decimals).

### 12.3 Protocol

Design chosen on a development split -- train -10/10 C (4,017 windows, 35
batteries), validate 5 batteries, score on every 25 C window (2,461, 20
batteries) -- with 42 C untouched. The dev baseline (83.5% / 60.5%) matched
42 C's (83.3% / 59.2%), so 25 C was a fair stand-in. The selection was
committed (`da375d5`) before the 42 C run, which was executed once. (A first
attempt was killed for memory before producing any checkpoint or score;
nothing was observed from it.)

### 12.4 Dev grid (25 C unseen), tolerance accuracy factual / what-if

| | gru_vae | RSSM |
|---|---|---|
| plain (round 6 design) | 14.5% / 8.8% | 23.0% / 21.0% |
| offsets | 53.1% / 45.0% | 42.6% / 48.0% |
| anchored | 82.0% / 55.7% | **87.5% / 65.8%** |
| offsets + anchored | 78.8% / 56.9% | 83.6% / 61.8% |
| same-day-type baseline | 83.5% / 60.5% | |

Only RSSM + anchored beat the baseline on both. Offsets added nothing on
top of anchoring and were dropped.

### 12.5 Final: unseen 42 C, trained on -10/10/25 C, run once

| Tolerance accuracy | Round 6 | Round 7 | Baseline |
|---|---|---|---|
| RSSM, factual | 31.9% | **88.3%** | 83.3% |
| RSSM, what-if | 36.0% | **71.8%** | 59.2% |
| gru_vae, factual | 29.9% | 65.8% | 83.3% |
| gru_vae, what-if | 18.7% | 55.4% | 59.2% |

RSSM per feature, factual (model / baseline, MAE):

| Feature | Model | Baseline | MAE (round 6 RSSM MAE) |
|---|---|---|---|
| v_mean | 88.3% | 77.5% | 0.023 V (0.050) |
| v_min | 69.2% | 66.1% | 0.098 V (0.672) |
| i_mean | 89.0% | 81.2% | 0.30 A (0.60) |
| t_bat_mean | 95.0% | 92.4% | 0.41 C (5.58) |
| t_bay_mean | 96.7% | 95.7% | 0.18 C (4.05) |
| soc | 77.2% | 70.4% | 2.2 pts (3.4) |

What-if: the RSSM beats the baseline on nine of ten features, most on voltage
(v_mean 56.2% vs 28.4%) and battery temperature (94.7% vs 86.3%); the one
loss is peak bay temperature (83.9% vs 84.1%). SoC under a changed schedule
remains weak for everything (RSSM 32.0%, baseline 19.5%).

gru_vae with identical anchoring stays below the baseline on both, with large
losses on bay temperature and v_max: its corrections hurt features the
anchor already predicts well.

### 12.6 Verdict for round 7

1. **The unseen-climate failure was representational, not a data ceiling.**
   Predicting a correction to a physically sensible anchor, rather than an
   absolute value, takes the RSSM from 31.9% to 88.3% on a climate it never
   saw, and its battery temperature error from 5.6 C to 0.4 C.
2. **The RSSM is the architecture that benefits.** Same anchoring, same data:
   RSSM +5.0 / +12.6 points above the baseline, gru_vae -17.5 / -3.8 below.
   This strengthens round 6's architecture result.
3. **Caveats, stated:** one seed per model (dev margins of ~4 points between
   variants are within plausible seed noise); tolerance bands are our choice;
   noise on daily aggregates is simplified; SoC and bay temperature are still
   model inputs; abduction, SAE-probe and calibration scripts are not yet
   anchor-aware, so round 7 reports accuracy only.

Next, in order:
1. Three seeds for RSSM anchored and the baseline comparison on 42 C.
2. Make gate2b / abduction / calibration anchor-aware and rerun the
   head-to-head and interval coverage on the final checkpoint.
3. Drop SoC and bay temperature from the inputs (`cell/sensors.py`).
4. SoC under changed schedules (32%) is the weakest remaining output.

---

## 13. Round 8 — residual skip connections injecting the anchor (2026-09-16)

### 13.1 Idea

Round 7's RSSM adds the anchor only at the decoder output, so its correction
cannot depend on the value it corrects. Round 8 projects the anchor into
every decoder hidden layer, `x_l = ELU(W_l x_{l-1} + P_l anchor)`, with the
head still zero-initialised (`--decoder-layers`, `--anchor-skip`; spec
`docs/superpowers/specs/2026-09-16-anchor-skip-connections-design.md`).
Defaults reproduce round 7's decoder parameter for parameter; round 7's
final checkpoint still scores 88.3444% exactly.

### 13.2 Dev results (train -10/10 C, score 25 C; 42 C untouched)

| Variant | Params | Val MSE (seen climates) | Factual | What-if |
|---|---|---|---|---|
| round 7: 1 layer, anchor at output | 10,914 | 0.00366* | **87.5%** | 65.8% |
| depth control: 2 layers, no skips | 11,970 | 0.00366 | 86.6% | **67.4%** |
| anchor skips: 2 layers + skips | 12,674 | **0.00234** | 38.4% | 49.5% |
| anchor skips + ambient offsets | 12,674 | 0.02085 | 78.9% | 64.2% |
| same-day-type baseline | | | 83.5% | 60.5% |

*round 7's val MSE is from its own dev run.

### 13.3 What happened

- **Raw-anchor skips overfit the training climates.** Best validation error
  of any variant, worst unseen-climate accuracy. 100% of the 25 C test's
  battery-temperature anchors lie outside the training range (29-32% for bay
  temperature); injecting them into hidden layers lets the correction key on
  absolute values the model never saw. Round 7 escaped this only because an
  output-added anchor passes through no nonlinearity.
- **With ambient offsets the injected values are in range** (0-1% outside for
  battery temperature, 8% for bay and SoC), and temperatures become nearly
  exact: battery temperature 100.0% within 1 C, MAE 0.07 C factual (baseline
  92.0%). But SoC collapses to 28.8% factual and 10.1% what-if (baseline
  70.8% / 18.1%), and current drops below baseline. SoC anchors are still
  absolute and 8% out of range.
- **Depth is not the issue**: the 2-layer control matches round 7.

### 13.4 Verdict

Neither skip variant beats round 7, so 42 C was not run again; round 7
remains the reported design. Skip connections carry a real but channel-
specific benefit (temperature) and a channel-specific harm (SoC, current).

Stated risk going forward: this dev split has now selected among eight
round 7 variants and three round 8 variants. Each additional variant tried
on the same 25 C set erodes how much its dev number predicts the unseen 42 C
result. A channel-selective skip design would be a reasonable next test, but
it should be validated with a fresh split (e.g. leave-one-climate-out) rather
than another pass over 25 C.

---

## 14. Round 10 — an attention-picked anchor (2026-09-16)

### 14.1 Idea

Rounds 7-9 used a fixed anchor rule: copy the latest window day of the same
drive/layup type. Round 10 lets the model choose instead, with attention over
the observed window whose weights are non-negative and sum to one, so the
anchor is a convex combination of days this battery actually had
(`experiment/attention.py`, `--attn-anchor`; spec
`docs/superpowers/specs/2026-09-16-attention-anchor-design.md`). That keeps
every anchor value inside the battery's own recent history, which is exactly
what round 8's raw-value skips violated.

Scores are a learned query/key term plus a type-match bonus (`beta`, init 20)
and a per-day-ago recency penalty (`gamma`, init 2); with zero-initialised
projections an untrained module IS the round 7 rule, so round 10 starts where
round 7 starts. Filtering attends causally (day t may copy only days before
it). 11,652 parameters against round 7's 10,914.

### 14.2 Dev result (train -10/10 C, score 25 C)

| Variant | Val MSE | Factual | What-if |
|---|---|---|---|
| round 7: fixed same-type anchor | 0.00366 | **87.5%** | 65.8% |
| round 8: 2 layers, no skips | 0.00366 | 86.6% | **67.4%** |
| round 10: attention-picked anchor | 0.00633 | 77.2% | 52.3% |
| same-day-type baseline | | 83.5% | 60.5% |

Below the fixed rule on both, and below the no-learning baseline.

### 14.3 Why

The trained attention barely moved from its initialisation -- `beta` 19.91
(from 20), `gamma` 1.96 (from 2), mean attended day 0.34 days older than the
latest same-type day -- but it did spread **21% of its weight onto wrong-type
days** (effective days used 1.33). Mixing a layup day into a driving day's
anchor blurs exactly the distinction the anchor exists to make. The
correction head then had to learn against an anchor that was itself moving,
and validation error on the TRAINING climates was worse than round 7's, so
this is not only an unseen-climate effect.

Read together with round 8: the anchor rule is load-bearing and brittle.
Injecting it deeper (round 8) or learning it (round 10) both lose to using it
plainly at the output.

### 14.4 Process note

`experiment/accuracy.py` called `rollout` without the observed window, so the
finished 100-epoch attention run could not be scored and the comparison
failed. Fixed with `checkpoints.rollout_kwargs()`, now used by accuracy,
gate2b and abduction, plus a test covering all three model shapes. Round 7's
final 42 C checkpoint still scores 88.3444% through the edited code.

### 14.5 Verdict

Round 7's design stands: RSSM, anchored at the output, absolute temperatures,
88.3% factual and 71.8% what-if on the unseen 42 C test. Rounds 8 and 10 are
two recorded negative results about where the anchor may enter the network.

The 25 C dev split has now chosen among thirteen variants; its numbers should
be treated as a ranking signal, not an estimate. Any further architecture
search needs leave-one-climate-out (four held-out climates, multiple seeds)
rather than another pass over this split.

---

## 16. Round 12 — a per-day action space (2026-09-17)

### 16.1 What changed

Every calendar day now carries its own trip count, trip length, ambient and
accessory load, drawn from a per-battery habit
(`experiment/schedule.py`, `experiment/build_planned_dataset.py`). Rounds 4-11
held all four fixed for life: within one battery `ambient_c` had exactly one
value and trip length at most two, so the only expressible counterfactual was
drive-or-park. Same single recording as every other round -- trip length
truncates it, nothing new was recorded.

The evaluation counterfactual changed accordingly, from flipping `is_layup` to
**splitting trips**: same total driving minutes, twice the cold starts.
Verified per window: 946.1 driving minutes before and after, 62 starts -> 122.

### 16.2 Results (42 C annual-mean band held out, anchored RSSM, 1 seed)

| | Model | Baseline | Gap |
|---|---|---|---|
| Factual | 47.5% | 46.9% | **+0.6** |
| Counterfactual (split trips) | 36.6% | 28.0% | **+8.6** |

### 16.3 The pre-registered expectation only half held

Recorded before running: *"accuracy will fall for everything; what matters is
whether the gap over the baseline widens."*

- **Accuracy fell**, steeply and as predicted: 89.2% -> 47.5% factual,
  72.6% -> 36.6% counterfactual.
- **The counterfactual gap did not widen** -- it narrowed, +12.5 points in
  round 11's leave-one-climate-out to +8.6 here, while staying clearly
  positive.
- **The factual gap collapsed**, +6.9 to +0.6. On ordinary forecasting the
  model is now barely distinguishable from copying the latest similar day.

That is a partial failure of the round's own hypothesis and is reported as one.

### 16.4 The numbers are NOT comparable across rounds

The tolerance bands (+/-1 C, +/-0.05 V, +/-0.5 A, +/-2 SoC points) were chosen
for a world where ambient never moved. With a seasonal swing of 5-15 C the
same +/-1 C band is a far harder test, so part of the drop is the measuring
stick, not the model. The valid comparison is **model against baseline within
round 12**, not round 12 against round 7.

### 16.5 Where it fails, and it is the channel the design doc warned about

Peak bay temperature is the dominant failure: **10.4% against the baseline's
45.0%** factually (MAE 6.2 C), 8.1% against 20.6% counterfactually (MAE 7.5 C).
That is precisely the channel `load/thermal.py` distorts -- bay heat is bounded
by the recording's coolant and oil, so it does not respond to ambient, and with
per-day trip lengths it now jumps around in a way the model cannot track while
copying a similar past day still can. The design doc barred this channel from
headline claims in advance; it turns out to be the single biggest drag on the
factual score.

### 16.6 What the round did achieve

The claim the round exists for is now expressible, and the model wins it. On
the split-trip counterfactual it beats the baseline on the channels that carry
the sulfation story:

| Channel | Model | Baseline |
|---|---|---|
| SoC | **39.8%** | 22.9% |
| Mean current | **46.3%** | 30.0% |
| Mean voltage | **36.9%** | 7.4% |
| Peak battery temperature | **28.4%** | 18.7% |

So "what happens if the same driving is split into twice as many trips" is a
question this dataset can pose and the model answers better than copying --
which no earlier round could do at all.

### 16.7 Honest verdict

Round 12 delivers the *capability* (a real action space, a real counterfactual)
and costs most of the *headline number*. One seed, one held-out band. Before
any of this is claimed, the next steps are: retune the tolerance bands for a
seasonal world, fix or exclude peak bay temperature, and repeat with seeds and
leave-one-climate-out as round 11 did.

---

## 15. Round 11 — relative-EMA skips, and the first error bars (2026-09-16)

### 15.1 What was run

The first properly validated comparison in this report: leave-one-climate-out
over the round 6 noisy dataset. Hold out -10, 10, 25 and 42 C in turn, train
on the other three, select the epoch on validation batteries drawn from the
training climates, and score the held-out climate. Two seeds, two arms,
16 training runs:

- **reference** -- round 7's design (`--anchored`)
- **EMA skip** -- stacked EMAs of the window at 2/7/30-day half-lives, each
  minus the anchor, plus the fast-minus-slow trend, projected into every
  decoder hidden layer (`--anchored --ema-skip`)

Design chosen by the range test in section 15.4 below, before any run.

### 15.2 Results, tolerance accuracy (factual / what-if)

| Fold | Seed | Reference | EMA skip | Baseline |
|---|---|---|---|---|
| -10 C | 0 | 85.7% / 70.8% | 85.6% / 68.2% | 81.9% / 62.8% |
| -10 C | 1 | 88.3% / 75.6% | 85.3% / 69.6% | 81.9% / 62.8% |
| 10 C | 0 | 91.0% / 72.9% | 88.7% / **73.2%** | 81.7% / 57.7% |
| 10 C | 1 | 89.3% / 72.7% | 88.8% / **72.8%** | 81.7% / 57.7% |
| 25 C | 0 | 92.2% / 82.4% | 89.6% / 78.3% | 82.0% / 61.7% |
| 25 C | 1 | 90.8% / 77.5% | 88.6% / 76.2% | 82.0% / 61.7% |
| 42 C | 0 | 87.8% / 67.0% | 82.7% / 58.4% | 83.7% / 58.1% |
| 42 C | 1 | 88.2% / 62.2% | 87.9% / **62.9%** | 83.7% / 58.1% |

| Arm | Factual | What-if |
|---|---|---|
| round 7 reference | **89.2%** (sd 2.1) | **72.6%** (sd 6.2) |
| round 11 EMA skip | 87.2% (sd 2.4) | 69.9% (sd 6.7) |
| same-day-type baseline | 82.3% (sd 0.9) | 60.1% (sd 2.3) |

Paired by fold and seed, the reference wins factual **8 of 8** (sign test
p = 0.008) and what-if 5 of 8. The EMA skip is therefore reliably worse
factually and not distinguishable on counterfactuals. Both arms beat the
baseline on every fold.

### 15.3 The error bars, and what they retract

This is the first design in the report run more than once, and the spread is
large: **what-if sd above 6 points**, factual sd about 2 points. Consequences,
stated rather than buried:

- Every single-seed margin in rounds 5-10 below roughly 2 points factual or
  6 points what-if was never resolvable. That includes round 7 over the depth
  control (0.9 points) and round 8's what-if edge (1.6 points).
- What survives are the large effects: round 6 to round 7 (31.9% to 88.3%),
  rounds 8 and 10's collapses (38.4%, 77.2%), and both arms over the baseline
  here (+5 to +13 points).
- **Round 7's headline replicates.** On the hold-42 C fold -- the same setup
  as its single reported run -- the reference scores 87.8% and 88.2% against
  the 88.3% reported. Three runs, tight cluster.

### 15.4 Why the design was expected to work, and why that reasoning was incomplete

The range test (dev split, share of unseen-climate values outside the
training range) said raw EMAs are as out-of-range as raw values (26.4% vs
26.6%, battery temperature 100%), while EMA-minus-anchor is 0.1-0.2% and the
trend 0.4%. Being in range is evidently **necessary but not sufficient**:
these features are in range and still cost accuracy.

Read with rounds 8 and 10, the pattern is consistent across three attempts:
the anchor works because it is added at the output, through no nonlinearity.
Anything routed into the hidden layers -- the raw anchor (round 8), a learned
anchor (round 10), or in-range relative summaries (round 11) -- has so far
cost more than it gained.

### 15.5 Verdict

Round 7 stands: RSSM, anchored at the output, absolute temperatures. Its
leave-one-climate-out average is **89.2% factual / 72.6% what-if**, against a
copy baseline of 82.3% / 60.1%, and it beats the baseline on all four
climates including the cold extrapolation.

Round 11 is the third recorded negative result on decoder inputs. Per the
protocol fixed before the run, the architecture search stops here.

Remaining gaps, unchanged by this round: abduction, calibration and the SAE
probe have still never been run on the round 7 design (the scripts became
compatible only in round 10), and nothing in the pipeline yet converts a
rollout into RUL, which is what the project's output contract actually
reports.

---

## Reproduction

```bash
python3 -m experiment.build_dataset runs/telemetry.csv --out runs/experiment   # ~22 min
python3 -m experiment.windows       --dataset-dir runs/experiment             # Gate 1
python3 -m experiment.build_counterfactual_targets --dataset-dir runs/experiment  # ~20 min (round 4 curriculum data)
python3 -m experiment.train_world_model --dataset-dir runs/experiment         # Gate 2, ~2-3 min (100 epochs, dual rollout)
python3 -m experiment.gate2b        --dataset-dir runs/experiment             # Gate 2b, ~2 s
python3 -m experiment.abduction     --dataset-dir runs/experiment             # Step 3, ~2.5 min
python3 -m experiment.sae_probe     --dataset-dir runs/experiment             # Step 4, ~3 s
python3 -m experiment.calibration   --dataset-dir runs/experiment             # Step 5 (added round 5)

# Round 5, RSSM -- writes rssm_* beside the gru_vae files
python3 -m experiment.train_rssm    --dataset-dir runs/experiment             # ~15 min (100 epochs)
python3 -m experiment.gate2b        --dataset-dir runs/experiment --arch rssm
python3 -m experiment.abduction     --dataset-dir runs/experiment --arch rssm # ~8 min
python3 -m experiment.sae_probe     --dataset-dir runs/experiment --arch rssm
python3 -m experiment.calibration   --dataset-dir runs/experiment --arch rssm
```

`experiment.build_counterfactual_targets` must run before
`experiment.train_world_model` -- the latter now requires
`windows_train_cf.npz` and fails with a clear error naming the command to
run first if it is missing. To reproduce round 3 (before the curriculum
fix) instead, checkout commit `35bce65` and skip that step; round 3's
`train_world_model.py` did not require the CF file.

Outputs (all gitignored under `runs/`): `daily_trajectory.csv`,
`scenarios.json`, `daily_dataset.json`, `windows_train.npz`,
`windows_test.npz`, `windows_meta.json`, `windows_train_cf.npz`,
`world_model.pt`, `gate2_results.json`, `gate2b_results.json`,
`abduction_results.json`, `sae_probe_results.json`. Only the `experiment/`
source package is committed.

---

## 17. Round 13 — planning, a wrong conclusion, and its correction (2026-09-17)

### 17.1 The planner

`experiment/planner.py` optimises the dials a driver controls (trips per day,
minutes per trip, accessory load) by gradient descent through the world model
onto `experiment/damage.py`'s wear terms. Ambient and parked days are held
fixed: nobody chooses the weather, a layup is a life event, and letting the
optimiser move temperature would invite it to exploit the bay-model artifact.
Every recommended schedule is then replayed through the real ODE from the
battery's true hidden state.

On 100 test batteries the model promised x0.565 damage and the simulator
delivered x0.995. 86% of schedules were genuinely gentler, all trivially so.
The verification step did its job: reported without it, this would have been a
"43% life extension" claim that is false.

### 17.2 The wrong conclusion

Hand-built extreme schedules replayed through the ODE moved 30-day damage by
~1% (one trip a day x0.986, accessories off x0.994, accessories maxed x1.001).
I concluded that driving habits are nearly irrelevant next to climate, and
that this contradicted the project's framing.

**That conclusion was wrong, and the error was the horizon.**

### 17.3 The correction

Run to a full 2,000-day life instead of 30 days, same simulator:

| Habit | Health at day 2000 | Crystal | Damage vs driver's own |
|---|---|---|---|
| 4 short trips a day | 0.7891 | 1.000 | **2.49x** |
| driver's actual pattern | 0.9154 | 0.552 | 1.00x |
| 1 long trip a day | 0.9388 | 0.191 | **0.72x** |
| accessories maxed | 0.9112 | 0.622 | 1.05x |
| +6 C hotter climate | 0.8802 | 0.528 | 1.42x |

Short trips do **3.4x** the damage of long ones, which exceeds a 6 C climate
shift. Over 30 days those differences are ~1% and invisible; over a life they
compound, because sulfation is a state whose growth feeds on itself. The
project's framing holds; the 30-day test was simply blind to it.

### 17.4 A real limitation, separated from the wrong conclusion

Driving raises battery temperature by a median of **+2.2 C** here, while peak
bay temperature rises 25.4 C. On a real car a bonnet-mounted battery runs
10-20 C above ambient after a long drive. The engine-to-battery heat path is
understated, for the reason flagged in round 12's design: bay heat is pinned
to the single recording's coolant and oil.

So the 3.4x above is carried mostly by **sulfation** (cold starts, incomplete
recharge), with driving's **thermal** contribution muted. A better thermal
coupling would likely widen the gap, not narrow it.

### 17.5 What this means for the planner

It optimised the wrong objective: 30-day damage, where habits are invisible.
It should minimise the damage RATE and project it over years, which 17.3 shows
is valid because the differences persist and compound. The world model rolls
30 days, so the planner needs to optimise rate-per-day and hand that to
`cell/life.py`'s projection rather than scoring the horizon it can see.

### 17.6 Lesson recorded

A negative result measured on the wrong horizon is not a negative result. I
reported "habits barely matter" on the strength of a 30-day test, against a
project whose own documentation said otherwise; the full-life check took
minutes and reversed it.
