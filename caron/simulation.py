"""Simulation class producing data for the visualization."""


# imports
from caron.data import Data
import numpy as np
import time
from collections import deque
import threading
from typing import Any
import jax.numpy as jnp
from caron.physics import primitive_to_conserved, conserved_to_primitive
from caron.solver import solve_euler_2d
from jax import jit


class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""


    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, data: Data, args, init: str | None = None) -> None:
        self.args = args
        self.data = data
        self.sim_size: int = self.args.sim_size

        # Rolling window for FPS measurement
        self.timestamps = deque()
        self.window_s = 2.0 # seconds
        self.lock = threading.Lock()
        self.cond = threading.Condition()

        self._rate_lock = threading.Lock()
        self._rate_active_start: float | None = None # start of current unpaused segment
        self._rate_active_time: float = 0.0 # accumulated unpaused time
        self._rate_frames: int = 0 # frames pushed since last reset
        self._avg_fps: float = 0.0 # current average simulation fps
        self._seen_sim_cmd_version: int = 0 # in overflow, the bump changes this value and resets the avg measurement

        self.prim: jnp.ndarray = jnp.array([])  # will be set in import_init of by input file. Needed here because of initial push to buffer

        if init is None:
            self.import_init()

    def import_init(self) -> None:
        """Imports the initial conditions. Right now a 2D Sod shock tube is initialized, later more options will be added."""

        # Numerical parameters
        self.nx: int = 400
        self.ny: int = 1     # ny = 1 => effectively 1D
        self.t_end: float = 0.2
        self.dt = 0.001
        self.gamma: float = 1.4
        self.CFL: float = 0.8

        # Domain
        self.xmin: float = 0.0
        self.xmax: float = 1.0
        self.ymin: float = 0.0
        self.ymax: float = 1.0

        x0 = 0.5 * (self.xmin + self.xmax)

        x = jnp.linspace(self.xmin, self.xmax, num=self.nx)
        y = jnp.linspace(self.ymin, self.ymax, num=self.ny)
        X, _ = jnp.meshgrid(x, y, indexing="ij")

        rho = jnp.where(X < x0, 1.0, 0.125)
        prs = jnp.where(X < x0, 1.0, 0.1)

        # 3 velocity components: vx, vy, vz
        vel = jnp.zeros((3, self.nx, self.ny))

        self.coords = (x, y)
        self.prim = jnp.concatenate(
            [rho[None, ...], vel, prs[None, ...]],
            axis=0
        )
        self.data.coords = self.coords
        #

    @property
    def dx(self):
        """
        Grid spacing in the x-direction.
        """
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self):
        """
        Grid spacing in the y-direction.
        If ny = 1, this will be the full domain height (1D case).
        """
        # guard against ny = 0; normally ny >= 1
        return (self.ymax - self.ymin) / max(self.ny, 1)


    # Runs
    # ----------------------------------------------------------------------
    def run(self) -> None:
        self.cons = primitive_to_conserved(self.prim, self.gamma)

        raise NotImplementedError("Simulation.run is not implemented yet.")

    def run_no_viz(self) -> None:
        self.data.push_frame(self.prim)  # Push initial quantity frame to data buffer        
        self.cons = primitive_to_conserved(self.prim, self.gamma)
        self.cons = solve_euler_2d(
            U0=self.cons,
            t_end=self.t_end,
            dx = self.dx,
            dy = self.dy,
            gamma=self.gamma,
            coords=self.coords,
            dt = self.dt,
        )
        self.prim = conserved_to_primitive(self.cons, self.gamma)

    def run_mock(self) -> None:
        """Feed frames into the buffer at a fixed rate (sim_fps), and eventually get paused/unpaused."""
        self.sim_fps=self.args.fake_sim_fps #Hz mock simulation fps
        self.sim_file = self.args.sim_file
        sim=np.load(self.sim_file)
        self.sim_size: int = sim.shape[2]
        print(f"[Sim] Loaded mock simulation from {self.sim_file} with {sim.shape[0]} frames of linear size {sim.shape[2]}.")
        print(f"[Sim] Simulation initialized in fake_injection mode. Injecting the buffer at {self.sim_fps} FPS.")

        dt = 1.0 / self.sim_fps # seconds per frame of the mock simulation injection
        t_next = time.perf_counter() +dt # time of the next frame to push (cumulative)

        now0 = time.perf_counter() # starting time t0
        self._seen_sim_cmd_version = self._get_sim_cmd_version() # same command state at start
        self._reset_rate_measurement(now0) 

        # reporting
        report_every_s = 1.0
        last_report_t = time.perf_counter()
        last_report_pushed = 0
        pushed = 0 # number of pushed frames, for the average fps computation

        for frame in sim:
            now = time.perf_counter()
            if self.args.verbose:
                print(f"[Sim] Starting frame pushing at {now}")
            self._sync_sim_command_from_data(now) # react to Data pause/unpause command and reset avg window if changed

            if self._is_sim_paused(): # if paused, close active segment and wait until unpaused
                self._rate_on_pause(now)
                self.data.wait_if_sim_paused()
                now = time.perf_counter()
                self._rate_on_resume(now) # resume segment after pause
                t_next = now + dt # restart schedule cleanly, not immediately

            now = time.perf_counter() # start timing
            if now < t_next:
                time.sleep(t_next - now) # is this appropriate? Maybe we should take a statistical approach as in the visualizer? To be checked

            self.data.push_frame(frame) 
            t_push = time.perf_counter() # time of push completion
            if self.args.verbose:
                print(f"[Sim] Finalized frame pushing at {t_push}")

            self._rate_tick_frame_pushed(t_push) # timestamp after the frame is pushed
            pushed += 1

            t_next += dt
            if t_push > t_next + dt: # the cumulative drift correction: if we're behind, catch up
                t_next = t_push + dt

            # periodic report
            if (t_push - last_report_t) >= report_every_s:
                inst_fps = (pushed - last_report_pushed) / (t_push - last_report_t) # instantaneous fps over the last interval
                avg_fps = self.get_measured_fps() # average fps in this unpaused segment
                if self.args.verbose:
                    print(f"[Sim] pushed={pushed}/{sim.shape[0]} | buffer={len(self.data)} | inst≈{inst_fps:.2f} Hz | avg≈{avg_fps:.2f} Hz")
                last_report_t = t_push
                last_report_pushed = pushed

        self.data.sim_finished = True
        print("[Sim] Simulation finished pushing frames.")


    # Command getters (called by Data)
    # ------------------------------------------------------------------
    def get_measured_fps(self) -> float:
        """Return average FPS over active injection time."""
        with self._rate_lock:
            return self._avg_fps
        

    # Internal functions to calculate the average fps
    # ------------------------------------------------------------------
    def _reset_rate_measurement(self, now: float) -> None:
        """Reset the averaging window (required when something changes)."""
        with self._rate_lock:
            self._rate_active_time = 0.0
            self._rate_frames = 0
            self._avg_fps = 0.0
            paused = self._is_sim_paused() 
            self._rate_active_start = None if paused else now # If it has just been paused, stop the avg computing. If it's been unpaused, start it

    def _rate_on_pause(self, now: float) -> None: 
        """Close active segment (exclude paused time)."""
        with self._rate_lock:
            if self._rate_active_start is not None:
                self._rate_active_time += now - self._rate_active_start
                self._rate_active_start = None

    def _rate_on_resume(self, now: float) -> None:
        """Open active segment after pause."""
        with self._rate_lock:
            if self._rate_active_start is None:
                self._rate_active_start = now

    def _rate_tick_frame_pushed(self, now: float) -> None:
        """Called only when a frame has actually been pushed into the buffer."""
        with self._rate_lock:
            if self._rate_active_start is None:
                self._rate_active_start = now # If we push while active_start isn't set, start it now
            self._rate_frames += 1
            active_time = self._rate_active_time + (now - self._rate_active_start)
            if active_time > 0:
                self._avg_fps = self._rate_frames / active_time


    # Internal functions to handle pauses
    # ------------------------------------------------------------------
    def _is_sim_paused(self) -> bool:
        try:
            return bool(self.data.is_sim_paused())
        except Exception:
            return bool(getattr(self.data, "sim_paused", False))

    def _get_sim_cmd_version(self) -> int:
        try:
            return int(self.data.get_sim_cmd_version())
        except Exception:
            return int(getattr(self.data, "sim_cmd_version", 0))

    def _sync_sim_command_from_data(self, now: float) -> None:
        """If Data issued a new sim command (pause/unpause), reset avg-fps window."""
        ver = self._get_sim_cmd_version()
        if ver == self._seen_sim_cmd_version: # if nothing changed, do nothing
            return
        self._seen_sim_cmd_version = ver # but if something changed, reset the measurement and the timer
        self._reset_rate_measurement(now)
        if self._is_sim_paused():
            self._rate_on_pause(now)
        else:
            self._rate_on_resume(now)
