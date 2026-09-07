# car_agent

Generate diverse, physics-constrained driving data in BeamNG to study how driving
and trip patterns age a **12 V lead-acid starter battery** in a combustion car.

An LLM turns plain English ("a delivery courier in Delhi in summer") into a
bounds-checked parameter set; a classical controller drives that behaviour; the
resulting engine and thermal traces feed a battery ageing layer.

---

## If you are running this on the Windows machine with BeamNG

You need **BeamNG.drive**, Python 3.10+, and about ten minutes.

### 1. Install

```
git clone <this repo>
cd car_agent
py -m pip install vgamepad
```

`vgamepad` needs the **ViGEmBus** driver. If it is not already installed, get it
from https://github.com/nefarius/ViGEmBus/releases and reboot.

There is no API key needed for anything below — three behaviours are already
generated and cached in `behaviour/cache/`.

### 2. Turn on BeamNG's telemetry

In BeamNG: **Options → Others** (some builds call it *Protocols*), enable both:

- **OutGauge** — engine channels (speed, RPM, coolant, pedals).
- **Motion Sim** — position and orientation.

Point both at `127.0.0.1`. **They may share a port** — BeamNG commonly sends
both to `4444`, and that works fine: datagrams are routed by content, not by
port. OutGauge arrives as 96 bytes, Motion Sim as 88 bytes tagged `BNG1`.

> **Both are required.** OutGauge carries no position at all, so the car cannot
> be steered along a route without Motion Sim.

### 2b. BeamNG 0.39 has a built-in MCP server — check what it offers

**Options → Advanced → General → "Enable MCP server"** (0.39+). First-party, no
BeamNG.tech licence, no third-party mod. It serves Streamable HTTP on
`http://127.0.0.1:29292/mcp`.

```
py windows_mcp_probe.py --try-state
```

It offers 86 tools, and **the MCP path is now the recommended one** — no
ViGEmBus, no UDP, no re-plug trick:

```
py windows_drive.py "Delhi Courier" --mcp --rate 20 --route town-loop.json
```

`--rate 20` because every control step is several HTTP round trips.

What MCP gives that the gamepad path cannot:

| | |
|---|---|
| `inject_input` | analog throttle/brake/steering, no virtual gamepad |
| `get_status` | position, speed and **damage** in one round trip |
| `get_electrics` | real rpm, gear, fuel, pedals, coolant and oil temperature |
| `get_vehicle_damage` | **collision detection** — the run stops when the car hits something |
| `get_vehicles` | other vehicles, i.e. traffic |
| `get_navgraph` | the road network the game's own AI drives on |
| `get_ground_at_point` | terrain drivability, 0–1 |
| `pause_physics` / `step_physics` | deterministic stepping |
| `set_simulation_speed` | faster than real time — a 10-minute run need not take 10 minutes |

Nothing in the probe drives the car.

### 3. Run the gamepad/telemetry probe

Spawn a vehicle on a road, then:

```
py windows_probe.py
```

It is safe — it never drives the car. It will:

1. create a virtual Xbox pad and sweep the steering for 3 s (watch the front
   wheels — if they do not move, BeamNG is not binding the virtual pad, check
   Options → Controls)
2. listen on 4444 and 4445 for 20 s and decode whatever arrives

**Send the whole output back.** If the packets do not match the expected layouts
it prints sizes and raw floats, which is enough to fix the decoder in one pass.

### 4. Drive a behaviour

```
py windows_drive.py --list
py windows_record.py town-loop                       # drive the road yourself first
py windows_drive.py "Delhi Courier" --route town-loop.json --ports 4444
```

**Record the road before driving it.** The default route is a synthetic straight
line with no relationship to the road, so the car will leave the tarmac and the
run will abandon itself. `windows_record.py` captures the line you drive by hand
and every behaviour then follows that.

Put the car **in gear** (not neutral) and press **Ctrl+R** in game first to reset the vehicle. The script releases throttle
and brake on exit, including on Ctrl+C. If it ever loses control: Ctrl+C, then
Ctrl+R in game.

Output lands in `runs/`:

- `<timestamp>_<spec_hash>.csv` — 1 Hz: pose, speed, RPM, coolant, under-bonnet
  temperature, oil, gear, fuel, crank count, idling flag, actual **and**
  commanded pedals
- `<timestamp>_<spec_hash>.json` — the behaviour spec, ambient temperature,
  seed, scenario, and a summary including corrosion in equivalent-hours

---

## Developing without the game

Everything except `windows_*.py` runs anywhere, against a kinematic fake backend:

```
python3 -m pytest -q        # 227 tests, ~20 s, no network, no game
./agent.py list
./agent.py drive "Delhi Courier"
```

Generating new behaviours needs a DeepSeek key:

```
export DEEPSEEK_API_KEY=...
./agent.py generate "a taxi driver in Cairo, 12 hour shifts, engine never off"
```

---

## Layout

| Path | What |
|---|---|
| `behaviour/` | `BehaviourSpec` + bounds gate, LLM generator, disk cache, DeepSeek client |
| `control/` | Path, pure-pursuit steering, PID speed, the driver that executes a spec |
| `sim/` | Backend interface, fake kinematic backend, engine/thermal model, OutGauge / MotionSim / OutSim decoding, gamepad+UDP backend |
| `battery/` | Arrhenius grid-corrosion estimator |
| `datalog/` | CSV writer + provenance sidecar |
| `campaign/` | *(empty — sampling and resumable ledger not built)* |

`CLAUDE.md` holds the design decisions and the reasoning behind them. Read it
before changing anything structural.

---

## What this does and does not do

**Does:** generate and validate behaviours offline; drive them faithfully
(acceleration, jerk and corner limits respected, ~2 cm path tracking); estimate
under-bonnet temperature; integrate grid corrosion.

**Does not, yet:**

- No electrical/alternator model, so the **recharge-deficit** ageing pathway
  produces no data.
- A run is one trip with one crank, so the **sulfation** pathway produces no
  data either.
- `idle_fraction` is recorded but not enforced during a run.
- Nothing has ever been run against the real game.

**Under-bonnet temperature is estimated, not measured.** Neither BeamNG nor
OutGauge reports engine bay air temperature. It is modelled from coolant
temperature, airflow and ambient. The structure is defensible; the coefficients
are not calibrated against a real engine bay. **Treat ratios between behaviours
as usable and absolute temperatures as indicative.**

Early result from the model (not validated against reality): with ambient held
fixed, driving style moves corrosion 1.2–1.5×, while ambient 25 °C → 42 °C moves
it ~2.9×. Ambient appears to dominate driving style roughly 2:1.
