"""Simulation class producing data for the visualization."""

# imports
from caron.data import Data
import numpy as np
import time
from collections import deque
import threading
import jax.numpy as jnp
from caron.physics import primitive_to_conserved, conserved_to_primitive
from caron.solver import advance_step, solve_euler_2d
from caron.initial_conditions import load_initial_condition


class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""

    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, data: Data, args, init: str | None = None) -> None:
        self.args = args
        self.data = data
        self.sim_size: int = self.args.sim_size
        self.init_name: str = str(
            init if init is not None else getattr(self.args, "init", "riemann_2d")
        )
        self._restart_event = threading.Event()

        # Rolling window for FPS measurement
        self.timestamps = deque()
        self.window_s = 2.0  # seconds
        self.lock = threading.Lock()
        self.cond = threading.Condition()

        self._rate_lock = threading.Lock()
        self._rate_active_start: float | None = (
            None  # start of current unpaused segment
        )
        self._rate_active_time: float = 0.0  # accumulated unpaused time
        self._rate_frames: int = 0  # frames pushed since last reset
        self._avg_fps: float = 0.0  # current average simulation fps
        self._seen_sim_cmd_version: int = (
            0  # in overflow, the bump changes this value and resets the avg measurement
        )

        self.prim: jnp.ndarray = jnp.array([])  # will be set in import_init of by input file. Needed here because of initial push to buffer

        if args.fake_injection or args.no_sim:
            self.quantities = ["West", "North", "East", "South"] # these have to be set in the input
            self.data.quantities = self.quantities # and to be pushed
        if init is None and not args.no_sim and not args.fake_injection:
            self.import_init(self.init_name)

    def import_init(self, init_name: str | None = None) -> None:
        """Load one named initial-condition preset from the registry."""
        if init_name is not None:
            self.init_name = str(init_name)
        init_data = load_initial_condition(self.init_name, self.args)

        self.nx = int(init_data["nx"])
        self.ny = int(init_data["ny"])
        self.t_end = float(init_data["t_end"])
        self.dt = float(init_data["dt"])
        self.gamma = float(init_data["gamma"])
        self.CFL = float(init_data["CFL"])
        self.bc_x = str(init_data["bc_x"])
        self.bc_y = str(init_data["bc_y"])
        self.xmin = float(init_data["xmin"])
        self.xmax = float(init_data["xmax"])
        self.ymin = float(init_data["ymin"])
        self.ymax = float(init_data["ymax"])
        self.coords = init_data["coords"]
        self.initial_prim = init_data["prim"]
        self.prim = self.initial_prim
        self.data.coords = self.coords
<<<<<<< simulation
=======
        #

>>>>>>> main

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
        while True:
            self.prim = self.initial_prim
            self.data.reset_buffer([self.get_initial_frame_tuple()])

            now = time.perf_counter()
            self._reset_rate_measurement(now)
            self._rate_tick_frame_pushed(time.perf_counter())

            self.cons = primitive_to_conserved(self.prim, self.gamma)
            t, step = 0.0, 0
            restart_requested = False
            while t < self.t_end:
                if self._consume_restart_request():
                    restart_requested = True
                    break
                self.data.wait_if_sim_paused()  # respect overflow pause commands from Data
                if self._consume_restart_request():
                    restart_requested = True
                    break

                dt = float(min(self._stable_dt(), self.t_end - t))
                self.cons = advance_step(
                    self.cons, dt, self.dx, self.dy, self.gamma, self.bc_x, self.bc_y
                )
                t += dt
                step += 1
                self.prim = conserved_to_primitive(self.cons, self.gamma)
                frame = np.array(self.prim)
                if not np.isfinite(frame).all():
                    print(
                        f"[Sim] NaN/Inf detected at t={t:.4f} (step {step}): simulation unstable, stopping."
                    )
                    break
                self.data.push_frame((frame, t))  # (frame, sim_time) tuple
                self._rate_tick_frame_pushed(time.perf_counter())

            if restart_requested:
                if self.args.verbose:
                    print("[Sim] Restart requested. Resetting simulation to t=0.")
                continue

            self.data.sim_finished = True
            print("[Sim] Simulation finished pushing frames.")
            self._restart_event.wait()
            self._restart_event.clear()
            if self.args.verbose:
                print("[Sim] Restart request received after completion.")

    def run_no_viz(self) -> None:
<<<<<<< simulation
        self.data.push_frame(self.prim[0])  # Push initial density frame to data buffer
=======
        self.data.push_frame(self.prim)  # Push initial quantity frame to data buffer        
>>>>>>> main
        self.cons = primitive_to_conserved(self.prim, self.gamma)
        self.cons = solve_euler_2d(
            U0=self.cons,
            t_end=self.t_end,
            dx=self.dx,
            dy=self.dy,
            gamma=self.gamma,
            coords=self.coords,
            dt=self.dt,
        )
        self.prim = conserved_to_primitive(self.cons, self.gamma)
<<<<<<< simulation
        # raise NotImplementedError("[Sim] Simulation.run is not implemented yet.")
=======
>>>>>>> main

    def run_mock(self) -> None:
        """Feed frames into the buffer at a fixed rate (sim_fps), and eventually get paused/unpaused."""
        self.sim_fps = self.args.fake_sim_fps  # Hz mock simulation fps
        self.sim_file = self.args.sim_file
        sim = np.load(self.sim_file)
        self.sim_size: int = sim.shape[2]
<<<<<<< simulation
        print(
            f"[Sim] Loaded mock simulation from {self.sim_file} with {sim.shape[0]} frames of linear size {sim.shape[2]}."
        )
        print(
            f"[Sim] Simulation initialized in fake_injection mode. Injecting the buffer at {self.sim_fps} FPS."
        )
=======

        # set spatial coordinates
        self.xmin: float = 0.0
        self.xmax: float = 1.0
        self.ymin: float = 0.0
        self.ymax: float = 1.0
        x = jnp.linspace(self.xmin, self.xmax, num=self.args.sim_size)
        y = jnp.linspace(self.ymin, self.ymax, num=self.args.sim_size)
        self.coords = (x, y) # spatial coordinates
        self.data.coords = self.coords # push to data

        print(f"[Sim] Loaded mock simulation from {self.sim_file} with {sim.shape[0]} frames of linear size {sim.shape[2]}.")
        print(f"[Sim] Simulation initialized in fake_injection mode. Injecting the buffer at {self.sim_fps} FPS.")
>>>>>>> main

        dt = 1.0 / self.sim_fps  # seconds per frame of the mock simulation injection
        t_next = time.perf_counter() + dt  # time of the next frame to push (cumulative)

        now0 = time.perf_counter()  # starting time t0
        self._seen_sim_cmd_version = (
            self._get_sim_cmd_version()
        )  # same command state at start
        self._reset_rate_measurement(now0)

            # reporting
            report_every_s = 1.0
            last_report_t = time.perf_counter()
            last_report_pushed = 0
            pushed = 0
            restart_requested = False

            for frame in sim:
                if self._consume_restart_request():
                    restart_requested = True
                    break

                now = time.perf_counter()
                if self.args.verbose:
                    print(f"[Sim] Starting frame pushing at {now}")
                self._sync_sim_command_from_data(now)

                if self._is_sim_paused():
                    self._rate_on_pause(now)
                    self.data.wait_if_sim_paused()
                    now = time.perf_counter()
                    self._rate_on_resume(now)
                    t_next = now + dt

                now = time.perf_counter()
                if now < t_next:
                    time.sleep(t_next - now)

                self.data.push_frame(frame)
                t_push = time.perf_counter()
                if self.args.verbose:
                    print(f"[Sim] Finalized frame pushing at {t_push}")

                self._rate_tick_frame_pushed(t_push)
                pushed += 1

                t_next += dt
                if t_push > t_next + dt:
                    t_next = t_push + dt

                if (t_push - last_report_t) >= report_every_s:
                    inst_fps = (pushed - last_report_pushed) / (t_push - last_report_t)
                    avg_fps = self.get_measured_fps()
                    if self.args.verbose:
                        print(
                            f"[Sim] pushed={pushed}/{sim.shape[0]} | buffer={len(self.data)} | inst≈{inst_fps:.2f} Hz | avg≈{avg_fps:.2f} Hz"
                        )
                    last_report_t = t_push
                    last_report_pushed = pushed

            if restart_requested:
                if self.args.verbose:
                    print("[Sim] Restart requested. Restarting mock injection from frame 0.")
                continue

            self.data.sim_finished = True
            print("[Sim] Simulation finished pushing frames.")
            self._restart_event.wait()
            self._restart_event.clear()

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
            self._rate_active_start = (
                None if paused else now
            )  # If it has just been paused, stop the avg computing. If it's been unpaused, start it

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
                self._rate_active_start = (
                    now  # If we push while active_start isn't set, start it now
                )
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
        if ver == self._seen_sim_cmd_version:  # if nothing changed, do nothing
            return
        self._seen_sim_cmd_version = (
            ver  # but if something changed, reset the measurement and the timer
        )
        self._reset_rate_measurement(now)
        if self._is_sim_paused():
            self._rate_on_pause(now)
        else:
            self._rate_on_resume(now)
