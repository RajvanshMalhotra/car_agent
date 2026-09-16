# Round 12: a per-day action space, from the same recording

**Date:** 2026-09-16
**Status:** agreed in conversation, being implemented

## Why

Measured in section 14 of the report: all four action features are determined
by **one binary variable per day**. Within a single battery's 2,000-day life,
`ambient_c` has exactly 1 distinct value, `driving_minutes` and `soak_hours` at
most 2, and `is_layup` is the bit that sets them. So:

- the only counterfactual ever posed is "drive or park",
- copying the latest same-type day already scores 83%, because day type plus
  battery identity nearly determines the day,
- and "fewer short trips would have extended it by X%" -- a deliverable in
  CLAUDE.md -- is **unanswerable**, because trip length never varies within a
  battery.

## Constraint: no new data

The one BeamNG recording is all the driving data there is. Checked against the
code, four of five dials need no new recording:

| Dial | Source | Available? |
|---|---|---|
| Trip length | `truncate_driving` cuts the same recording short | yes |
| Trips per day | a loop count in `step_one_day` | yes |
| Ambient / seasons | `BatteryScenario.ambient_c`, a parameter | yes |
| Accessory + parasitic load | `accessory_base_a`, `lights`, `hvac`, `parasitic_a` | yes |
| Driving style / aggression | baked into the recording | **no -- dropped** |

`BatteryScenario` is a frozen dataclass of plain fields, so a per-day scenario
is `dataclasses.replace(...)`. `BayTemperature` is constructed inside
`run_trip`/`run_soak` rather than cached on `CellState`, so a per-day ambient
takes effect immediately instead of being silently ignored.

## The drawback, and how it is handled

`load/thermal.py` bounds bay temperature by the hot anchor (coolant, oil) taken
from the recording. A hotter simulated day does not make that engine run
hotter, so **a hot driving day is understated**: measured at 48 C, peak bay
moved only 94.2 -> 94.9 C. A hot *parked* day is fine, because battery
temperature follows ambient directly and corrosion is evaluated at battery
temperature.

The ODE carries this bias too, so validating against the simulator cannot
detect it. Therefore:

- **trip pattern is the primary axis** -- unaffected by this bias, and the one
  the short-trip claim needs;
- **seasons and accessory load are secondary** -- included so the action matrix
  has real dials (and so the planner has something to optimise), but barred
  from headline claims;
- no claim of the form "driving in summer costs X" will be made from this
  dataset.

## Action features

Appended, not replaced, so every name-based lookup and the round 7 anchor keep
working:

    is_layup, driving_minutes, soak_hours, ambient_c, trips_today, accessory_a

`trips_today` is separate from `driving_minutes` on purpose: together they
distinguish one 20-minute trip from four 5-minute trips, which is the
distinction the whole round exists to make. `is_layup` is now derived
(`trips_today == 0`).

## Per-day schedule (`experiment/schedule.py`)

Each battery keeps a *habit* (its between-battery identity); each day is drawn
from that habit, so variation now exists **within** a battery as well as across:

- `trips_today`: 0-3, drawn around the battery's habitual rate; layup blocks
  from `build_schedule` still force runs of 0.
- `trip_minutes`: 1.5-20, drawn around the battery's habitual trip length.
- `ambient_c`: seasonal sinusoid around the battery's annual mean, amplitude
  5-15 C, plus daily noise.
- `accessory_a`: drawn around the battery's habitual load. Deliberately
  independent of ambient -- correlating them would confound the two dials.

## The evaluation counterfactual changes

Rounds 4-11 flipped `is_layup`. That perturbation cannot express the claim.
The new eval counterfactual is **trip splitting**: hold total driving minutes
fixed and double the number of trips (halving each). Same distance, same
duration, more cold starts -- isolating trip *pattern* from trip *amount*,
which is exactly "fewer short trips would have extended it by X%".

The training curriculum keeps a different shape from the eval one (randomised
per-day dial perturbation), so training is never literally the test
transformation.

## Expectation, recorded before running

**Accuracy will fall for everything**, because the task stops being two
repeating templates. The number that matters is the *gap* between the model and
the same-day-type baseline, which should widen: the baseline loses its
advantage when days stop repeating, while the model can still use the actions
it is given. A narrowing gap would be evidence the model was only ever
exploiting the template structure.

## Scope

One dataset, all climates, seasonal. Train round 7's design (anchored RSSM) and
score against the per-fold baseline on a held-out annual-mean band. The full
16-run leave-one-climate-out treatment is deferred until this dataset proves
out.
