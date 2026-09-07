# Plan of action

```
════════════════════════════════════════════════════════════════════════
  PHASE 1 · COLLECT GAME DATA                          ◀── WE ARE HERE
════════════════════════════════════════════════════════════════════════

   "delivery driver, Delhi summer, aircon always on"
                    │
                    │   WHAT  plain english  →  numbers
                    │   WHY   need many driving styles, cheaply
                    │   HOW   LLM writes it ONCE, cached to disk
                    ▼
        ┌──────────────────────────┐
        │   BehaviourSpec          │    trip_duration_s = 300
        │   bounds-checked in code │    idle_fraction   = 0.45
        └──────────────────────────┘    hvac_setting    = 1.0
                    │                   cold_start      = True
                    │   WHAT  drive that style for real
                    │   WHY   real vehicle physics, not invented
                    │   HOW   pure-pursuit steer + PID speed
                    ▼
        ┌──────────────────────────┐
        │        BeamNG            │  ← Windows laptop
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐    rpm, coolant_temp, speed,
        │   RAW LOG   1 row/sec    │    throttle, idle, crank events
        └──────────────────────────┘
                    │
                    │   ⚠  game gives DRIVING only.
                    │      no battery, no current, no voltage.
                    ▼

════════════════════════════════════════════════════════════════════════
  PHASE 2 · TURN DRIVING INTO BATTERY STRESS
════════════════════════════════════════════════════════════════════════

        ┌──────────────────────────┐
        │   FEATURES  per trip     │   WHAT  squash 3600 rows → ~15 numbers
        │                          │   WHY   battery dies over MONTHS,
        │   • heat exposure        │         not seconds. raw rows are
        │     (Arrhenius-weighted) │         the wrong zoom level.
        │   • charge deficit       │   HOW   physics-shaped aggregates,
        │   • trip length / idle   │         normalised by capacity so
        │   • crank count          │         they work on any battery
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐   WHAT  stress → aging
        │   PHYSICS  (Schiffer)    │   WHY   this is what extrapolates
        │   corrosion + sulfation  │         to a life we never observed
        └──────────────────────────┘   HOW   lead-acid aging equations
                    │
                    │◀── your 21k rows check the voltage here (only job)
                    ▼
        ┌──────────────────────────┐
        │   AGING TRAJECTORIES     │   simulated batteries, full life
        └──────────────────────────┘

════════════════════════════════════════════════════════════════════════
  PHASE 3 · PREDICT
════════════════════════════════════════════════════════════════════════

        ┌──────────────────────────┐   WHAT  learn stress → lifetime
        │   MODEL (PINN)           │   WHY   the actual deliverable
        └──────────────────────────┘   HOW   physics-informed loss
                    │
                    ▼
        ┌──────────────────────────────────────────────────┐
        │  OUTPUT                                          │
        │                                                  │
        │  847 days   (90% range: 620 - 1150)              │
        │  ────────────────────────────────                │
        │  plan for 620          ← the safe number         │
        │                                                  │
        │  + "profile A is 2x worse than B"                │
        │  + "aging faster than expected"                  │
        │  + "fewer short trips → +18% life"               │
        └──────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
  PHASE 4 · LATER — WORLD MODEL  (deferred, not now)
════════════════════════════════════════════════════════════════════════

        counterfactuals:  "what if THIS battery was driven gently?"
        needs:            abduct hidden state → swap behaviour → roll
        validate on:      Severson (124 lithium cells, really died)
        why later:        project must work end-to-end first
```

## Data sources — three, doing three different jobs

```
  SOURCE                 JOB                              STATUS
  ─────────────────────  ───────────────────────────────  ──────────
  BeamNG game data       the actual input. driving.       build now
  Your 21k rows          check voltage physics. 1 job.    have it
  Severson (lithium)     prove the method works           phase 4
```

## What we build first (all on the Mac, no game needed)

```
  1. BehaviourSpec + validation      ── the driving styles
  2. Fake car backend                ── test without BeamNG
  3. Controller                      ── drives the spec
  4. CSV logger                      ── one row per second
  5. Campaign runner                 ── many trips, resumable
       │
       └─► then swap the fake car for BeamNG. Same interface.
           Nothing else changes.
```

## Known holes (not hidden)

```
  ✗  engine BAY temperature — game only gives coolant temp.
     we estimate it. it is our top feature. this is an assumption.

  ✗  battery side is 100% modelled. game has no 12V system.

  ✗  end-of-life never validated for lead-acid. no such data exists.
     that is what Phase 4 / Severson is for.
```
