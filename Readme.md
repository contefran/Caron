# Caron

Caron is a **real-time visualiser for 2D simulation frames** built around a simple idea: a simulation thread pushes frames into a shared buffer, and a visualisation thread consumes them at a (possibly different) rate. The `Data` layer acts as the “control tower” that keeps things stable when the producer/consumer rates drift apart.

For debugging purposes, Caron can run with a **mock simulation** stored in a NumPy file (`mock_sim.npy`) and supports two practical workflows:

1. Visualise an existing `.npy` buffer (`--no_sim`) already populated, without constant injection
2. Inject frames from a `.npy` buffer at a fixed FPS while visualising (`--fake_injection`)

---

## Features

- DearPyGui window with:
  - **Start / Stop** buttons
  - **FPS slider** (visualisation target FPS)
- Shared `Data` buffer implemented as a thread-safe `deque`
- Stability logic:
  - **Overflow protection**: pause simulation if buffer grows too large
  - **Underflow protection**: reduce visualisation FPS if buffer becomes too small
- Visualiser calibration phase to estimate maximum achievable visualisation FPS over a configurable time window

---

## Installation

Editable install (core dependencies only):

```bash
python -m pip install -e .
```

For the physics solver (JAX), pick the extra that matches your hardware:

```bash
pip install -e ".[jax-cpu]"   # CPU — Intel, AMD CPU, no GPU
pip install -e ".[jax-cuda]"  # NVIDIA GPU (CUDA 12)
pip install -e ".[jax-rocm]"  # AMD GPU (ROCm)
```

At startup the active JAX backend is printed to the console so you can confirm which device is in use.

Core dependencies (declared in `pyproject.toml`): NumPy, Matplotlib, DearPyGui, Numba.

---

## Quick start

### 1) Generate a mock simulation (`mock_sim.npy`)

```bash
python Make_mock_sim.py --num_frames 512 --img_size 512
```

This creates a 3D array with shape approximately `(num_frames, img_size, img_size)` and saves it as `mock_sim.npy`.

The details of the mock simulation can be seen by typing
```bash
python Make_mock_sim.py -h
```

---

### 2) Visualise a precomputed buffer (no simulation thread)

```bash
python Caron.py --no_sim --sim_file ./mock_sim.npy
```

In `--no_sim` mode, the visualiser loads the `.npy` file and **pre-fills the `Data` buffer**, then consumes frames from it at a FPS x, settable with the command `--viz_fps x`. Note that the maximum achievable FPS is probably limited by the monitor refresh frequency (e.g. 60 Hz)

---

### 3) Inject + visualise (producer/consumer with control loop)

```bash
python Caron.py --fake_injection --fake_sim_fps 40 --viz_fps 60 --sim_file ../mock_sim.npy
```

This starts:
- a **simulation injection thread** (`Simulation.run_mock`) pushing frames at `--fake_sim_fps`
- a **control loop thread** monitoring measured rates and buffer state
- the **visualiser** consuming frames and applying `Data` commands (FPS adjustments, waiting on underflow/overflow)

---

## Usage and CLI options

All options can be seen by typing `Caron.py -h`.

| Flag | Type | Default | Meaning |
|---|---:|---:|---|
| `--sim_size` | int | 512 | Linear grid size of the simulation grid |
| `--viz_fps` | float | 60 | Initial visualisation FPS target |
| `--calib_time` | float | 3 | Calibration duration (seconds of active “running” time) |
| `--calib_frames` | int | 50 | Minimum frames required during calibration |
| `--buffer_safe_max` | int | 300 | Buffer size above which simulation is paused (overflow) |
| `--sim_file` | str | `../mock_sim.npy` | Path to the mock `.npy` file |
| `--no_sim` | flag | off | Disable simulation thread; visualise `.npy` only |
| `--no_viz` | flag | off | Disable visualisation (run simulation only) |
| `--fake_injection` | flag | off | Inject `.npy` frames at fixed FPS in a sim thread |
| `--fake_sim_fps` | int | 60 | Injection FPS for `--fake_injection` |
| `--ctrl_dt` | float | 0.2 | Control-loop tick interval (seconds) |

---

## How it works (architecture)

Caron is split into four classes:

- **`Main`** (`Caron.py`): parses args, creates objects, starts threads, runs the controller loop.
- **`Simulation`** (`caron/simulation.py`): produces frames and pushes them into `Data`. In `--fake_injection`, it loads a `.npy` buffer and injects at a fixed rate, while reacting to pause/unpause commands from `Data`.
- **`Visualization`** (`caron/visualization.py`): consumes frames from `Data` at the required FPS and displays them via DearPyGui.
- **`Data`** (`caron/data.py`): shared buffer + control logic. It owns the `deque`, synchronisation primitives, and the rules for overflow/underflow stabilisation.

---

## Buffer control logic

### Overflow (buffer too large)
- Trigger: `len(buffer) > buffer_safe_max`
- Action: `Data` pauses the simulation (`sim_paused=True`) until the buffer is healthy again.
- Resume: when buffer falls below `(buffer_safe_max - pillow)`.

### Underflow (buffer too small)
- Trigger: `len(buffer) < buffer_safe_min` (and simulation not finished).
- Action: `Data` reduces the **visualisation target FPS** to roughly `sim_rate - viz_margin_fps`.
- Visualiser: applies the new FPS schedule.

---

## Visualisation calibration

Before normal operation, in the first `calib_time` seconds of active visualization, the visualiser runs a calibration phase where it updates as fast as possible in order to measure the maximum FPS achievable, and resizes the FPS slider max accordingly. Pausing the calibration does not affect this process.

---


