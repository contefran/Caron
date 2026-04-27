from dataclasses import dataclass, field
from collections import deque
from argparse import Namespace
import numpy as np
import threading
from typing import Any, Iterable


@dataclass
class Data:
    """Shared buffer for frames, and handler for FPS control."""


    # Initialization
    # ------------------------------------------------------------------
    args: Namespace
    maxlen = None
    min_viz_fps: float = 1.0 # minimum allowed viz fps during balancing
    max_viz_fps: float =120.0 # maximum allowed viz fps during balancing, in the beginning. It is updated later by the visualizer after calibration (also setting the slider maximum)
    viz_margin:float = 5.0 # [frames] margin before pausing/resuming the simulation
    viz_margin_fps: float = 5.0  # keep viz slightly slower than sim when starving
    pillow:int = 30 # [frames] hysteresis window to avoid rapid pause/resume cycles (~10% of default buffer_safe_max)
    viz_cmd_version: int = 0 # becomes 1 when fps changes
    sim_cmd_version: int = 0
    sim_paused: bool = False
    sim_finished: bool=False
    coords: tuple | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.quantities: list[str] = []  # pushed by simulation before viz starts
        self.buffer: deque = deque(maxlen=self.maxlen)
        self.lock = threading.Lock() # to protect buffer access
        self.cond = threading.Condition(self.lock) # to notify when new frames are available
        self.buffer_safe_min=self.args.calib_frames
        self.buffer_safe_max=self.args.buffer_safe_max
        self.viz_target_fps=self.max_viz_fps


    # Functions that act on the buffer
    # ------------------------------------------------------------------
    def push_frame(self, frame: np.ndarray) -> None:
        """Add a new 2D frame to the buffer."""
        with self.cond:  # uses same lock + allows notify
            self.buffer.append(frame) # well, yeah
            self._apply_safe_control_sim() # Apply control logic based on new buffer size
            self.cond.notify_all() # notify waiting visualizer threads (just in case sim is not paused)

    def pop_frame(self) -> np.ndarray | None:
        """Remove the oldest frame from the buffer (returns None if the buffer is empty)."""
        with self.cond:
            if not self.buffer:
                if self.args.verbose:
                    print("[Data] Buffer empty.")
                return None
            frame = self.buffer.popleft()
            self._apply_safe_control_sim() # Apply control logic based on new buffer size
            self.cond.notify_all()
            return frame

    def reset_buffer(self, frames: Iterable[Any] | None = None) -> None:
        """Clear and optionally refill the shared buffer, then resume simulation."""
        with self.cond:
            self.buffer.clear()
            if frames is not None:
                for item in frames:
                    self.buffer.append(item)
            self.sim_finished = False
            self._set_sim_paused_locked(False)
            self._apply_safe_control_sim()
            self.cond.notify_all()
        

    # Control logic
    # ------------------------------------------------------------------
    def _apply_safe_control_sim(self) -> None:
        """Control based purely on buffer size and measured rates."""
        n = len(self.buffer)
        # Buffer overflow: sim too fast, pause it
        if n > self.buffer_safe_max and not self.sim_paused:
            if self.args.verbose:
                print("[Data] Buffer overflow detected: pausing simulation.")
            self._set_sim_paused_locked(True)
        elif n < (self.buffer_safe_max - self.pillow) and self.sim_paused:
            if self.args.verbose:
                print("[Data] Buffer back to safe levels: resuming simulation.")
            self._set_sim_paused_locked(False)

    def wait_if_sim_paused(self, timeout: float = 0.1) -> None:
        with self.cond:
            while self.sim_paused:
                self.cond.wait(timeout=timeout)


    # Control function
    # ------------------------------------------------------------------
    def balance_rates(self, sim_rate: float, viz_rate: float) -> None:
        """
        Controller clock.
        Inputs:
          sim_rate: measured simulation push rate (Hz)
          viz_rate: measured visualization pop/render rate (Hz)
        Outputs (commands):
          - adjusts visualization target fps (only downward; never auto-increases)
        """
        sim_rate = float(sim_rate)
        viz_rate = float(viz_rate)

        with self.cond:
            n = len(self.buffer)
            if n < self.buffer_safe_min and not self.sim_finished:
                if sim_rate <= self.viz_margin_fps:
                    # sim hasn't ramped up yet or is stalled — don't apply cap
                    if sim_rate == 0.0:
                        print("[Data] Warning: simulation was found stopped during underflow.")
                        self._set_sim_paused_locked(False)
                    return
                balance = max(self.min_viz_fps, sim_rate - self.viz_margin_fps)
                # Snap down to the nearest 5 Hz floor; if within 1 Hz of it, go one step lower
                snapped = int(balance // 5) * 5
                if balance - snapped < 1.0:
                    snapped -= 5
                new_viz = max(self.min_viz_fps, float(snapped))
                if new_viz < self.viz_target_fps:
                    print(f"[Data] Buffer underflow: balance point {balance:.1f} Hz → capping viz at {new_viz:.0f} Hz")
                    self._set_viz_target_fps_locked(new_viz)
                    self.cond.notify_all()


    # Command getters (used by sim/viz)
    # ------------------------------------------------------------------
    def get_viz_target_fps(self) -> float:
        with self.lock:
            return self.viz_target_fps

    def get_viz_cmd_version(self) -> int:
        with self.lock:
            return self.viz_cmd_version

    def is_sim_paused(self) -> bool:
        with self.lock:
            return self.sim_paused

    def get_sim_cmd_version(self) -> int:
        with self.lock:
            return self.sim_cmd_version

    def is_sim_finished(self) -> bool:
        with self.lock:
            return self.sim_finished

    def set_sim_finished(self, value: bool) -> None:
        with self.lock:
            self.sim_finished = value

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


    def update_viz_target_fps_from_user(self, fps: float) -> None:
        """Called when the user manually sets FPS via the slider (no version bump)."""
        with self.lock:
            fps = max(self.min_viz_fps, min(float(fps), self.max_viz_fps))
            self.viz_target_fps = fps

    def reset_viz_target_fps(self) -> None:
        """Reset viz FPS to calibrated max on restart (bumps version so slider updates)."""
        with self.cond:
            self._set_viz_target_fps_locked(self.max_viz_fps)
            self.cond.notify_all()

    # Command setters (data use only)
    # ------------------------------------------------------------------
    def _set_viz_target_fps_locked(self, new_fps: float) -> None:
        new_fps = float(new_fps)
        if new_fps < self.min_viz_fps: # enforce minimum
            new_fps = self.min_viz_fps
        if new_fps > self.max_viz_fps: # enforce maximum
            new_fps = self.max_viz_fps

        if abs(new_fps - self.viz_target_fps) > 1e-12: # avoid changes due to just numerical noise
            self.viz_target_fps = new_fps
            self.viz_cmd_version += 1 # notify viz of command change

    def _set_sim_paused_locked(self, paused: bool) -> None:
        if paused != self.sim_paused:
            self.sim_paused = paused
            self.sim_cmd_version += 1
