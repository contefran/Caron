import time
import numpy as np
from numba import njit
from matplotlib import colormaps as cm
import threading
import dearpygui.dearpygui as dpg
from caron.data import Data


@njit
def normalize_frame(frame):
    frame_min = np.min(frame)
    frame_max = np.max(frame)
    norm = (frame - frame_min) / (frame_max - frame_min + 1e-8)
    return np.clip(norm, 0.0, 1.0)

def build_log_map(alpha=50.0):
    """Maps linear [0..255] -> log-compressed [0..255]."""
    x = np.linspace(0.0, 1.0, 256)
    y = np.log1p(alpha * x) / np.log1p(alpha)
    return (y * 255.0).astype(np.uint8)


class Visualization:
    """Consume frames from a Data buffer and display them with DearPyGui."""


    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, data:Data, args) -> None:
        self.data = data # this is the first buffer, obtained after a few seconds of simulation
        self.args = args
        self.frames = self.data.buffer
        if self.args.no_sim:
            frames_array = np.load(self.args.sim_file)
            for frame in frames_array:
                self.data.push_frame(frame) # fill the buffer with the mock sim. It's a deque

        # Simulation size
        self.sim_size = self.args.sim_size
        if self.args.no_sim:
            self.sim_size: int = self.frames[0].shape[1]

        # Status and main vars
        self.finished: bool = False
        self.running: bool = False
        self.fps = float(self.data.max_viz_fps)
        self.max_viz_fps: float = self.fps  # after calibration will be changed to the actual max FPS
        self._frame_period = 1.0 / self.fps
        self._next_due_time = time.perf_counter()
        self._last_frame: np.ndarray | None = None

        self.frame_index: int = 0

        # Calibration state
        self.calibrated: bool = False # becomes True after calibration
        self.calibrating: bool = False # True while we are in calibration mode
        self.calib_start_time: float | None = None
        self.calib_active_start: float | None = None  # when the current running segment began
        self.calib_active_time: float = 0.0 # sum of active time during calibration
        self.calib_frame_count: int = 0 # number of frames displayed during calibration
        self.calib_duration: float = float(self.args.calib_time)
        self.calib_frames: float = float(self.args.calib_frames)

        # Rate measurement
        self._rate_lock = threading.Lock()
        self._rate_active_start: float | None = None  # start of current running segment
        self._rate_active_time: float = 0.0 # accumulated running time (excludes pauses)
        self._rate_frames: int = 0 # frames actually displayed since last reset
        self._avg_fps: float = 0.0 # current average visualization fps
        self._seen_viz_cmd_version: int = 0 # in underflow, the bump changes this value and resets the avg measurement

        # LUTs for colormaps
        self.luts = {
            "inferno": (cm["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
            "viridis": (cm["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
            "Greys":   (cm["Greys"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
            "Blues":   (cm["Blues"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
            "Greens":  (cm["Greens"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
            "Oranges": (cm["Oranges"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8),
        }
        self.cmap_names = list(self.luts.keys())
        self.current_cmap = "inferno"
        self.current_lut = self.luts[self.current_cmap]

        # Quantities
        self.quantities = self.data.quantities # for instance ['Density', 'Pressure', 'Velocity']
        self.viz_quantity_index: int = 0 # the index of the quantity to visualize
        self.viz_quantity: str = self.quantities[self.viz_quantity_index]

        # Sliders and tags
        self._programmatic_slider_update = False
        self.slider_tag = "Caron FPS Slider"
        self.cmap_list_tag = "Caron Colormaps List"
        self.quantity_tag = "Caron Quantities List"
        self.log_strength_tag = "Caron Log Strength Slider"
        self.texture_tag = "frame_tag"
        self.log_scale: bool = False
        self.log_alpha: float = 50.0
        self.log_map = build_log_map(self.log_alpha)


    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Check the buffer, set up DearPyGui and start the visualisation."""

        if not self.frames:
            raise RuntimeError("[Viz] Visualization initialised with an empty buffer.")
        if self.args.verbose:
            print(f"[Viz] Loaded a simulation of size {self.sim_size}x{self.sim_size} with {len(self.frames)} frames as initial buffering.")

        # Precompute first frame to initialise the texture
        frame = self.frames[0][self.viz_quantity_index] # still works with deque, I'm already in love
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
        frame_rgb = np.stack((frame_norm,) * 3, axis=-1).astype(np.float32)
        self.frame_flattened: np.ndarray = frame_rgb.flatten()

        # DearPyGui setup
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_raw_texture(
                int(self.sim_size),
                int(self.sim_size),
                default_value=self.frame_flattened.tolist(),
                format=dpg.mvFormat_Float_rgb,
                tag=self.texture_tag,
            )

        # Window / layout
        with dpg.window(
            label="Jet Inspector",
            width=int(self.sim_size * 2.6),
            height=int(self.sim_size* 2.3),
            no_close=True,
            no_move=True,
            no_resize=False,
        ):
            with dpg.group(label="Visualizator"): # the big visualization window
                with dpg.group(label="Slider"): # above there is the slider
                    dpg.add_slider_int(
                        label="Speed (FPS)",
                        tag=self.slider_tag,
                        width=int(self.sim_size*1.9),
                        #height=int(self.sim_size*1),
                        default_value=int(self.fps),
                        min_value=1,
                        max_value=int(self.data.max_viz_fps),
                        callback=_fps_callback,
                        user_data=self,
                        enabled=False, # don't allow changes until the calibration is finished
                    )
                with dpg.group(label="Map & Keys", horizontal=True): # below there is the image and the buttons
                    with dpg.group(label='Colormap & Color Combo'): # image and color listbox on the left
                        dpg.add_image(texture_tag=self.texture_tag, width = int(self.sim_size *1.9), height= int(self.sim_size *1.9)) 
                        with dpg.group(label="Log visualization", horizontal=True):
                            dpg.add_combo(
                                tag=self.cmap_list_tag,
                                items=self.cmap_names,
                                default_value=self.current_cmap,
                                width=int(self.sim_size * 0.2),
                                callback=_colormap_callback,
                                user_data=self,
                            )
                            dpg.add_combo(
                                tag=self.quantity_tag,
                                items=self.quantities,
                                default_value=self.viz_quantity,
                                width=int(self.sim_size * 0.2),
                                callback=_quantity_callback,
                                user_data=self,
                            )
                        with dpg.group(label="Log visualization", horizontal=True):
                            dpg.add_slider_float(
                                tag=self.log_strength_tag,
                                default_value=self.log_alpha,
                                min_value=1.0,
                                max_value=500.0,
                                format="%.1f",
                                callback=_log_strength_callback,
                                user_data=self,
                                width=int(self.sim_size * 1.8),
                                enabled=False,
                            )
                            dpg.add_checkbox(
                                label="Log",
                                default_value=self.log_scale,
                                callback=_log_checkbox_callback,
                                user_data=self,
                            )
                    with dpg.group(label="Start & Stop"): # the buttons on the right
                        dpg.add_button( # the start button above
                            label="Start",
                            callback=_start_callback,
                            user_data=self,
                            width=int(self.sim_size*0.3),
                            height=int(self.sim_size*0.15),
                        )
                        dpg.add_button( # the stop button below
                            label="Stop",
                            callback=_stop_callback,
                            user_data=self,
                            width=int(self.sim_size*0.3),
                            height=int(self.sim_size*0.15),
                        )

        dpg.create_viewport(
            title="Our lovely Caron",
            width=int(self.sim_size * 2.6),
            height=int(self.sim_size* 2.3),
        )

        dpg.setup_dearpygui()

        # Kick off the first update
        _update_frame(None, None, self)

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


    # Command getters (called by Data)
    # ------------------------------------------------------------------
    def get_measured_fps(self) -> float:
        """Return average FPS over active display time."""
        with self._rate_lock:
            return self._avg_fps

    
    # Internal functions to calculate the average fps
    # ------------------------------------------------------------------
    def _reset_rate_measurement(self, now: float) -> None:
        """Reset the averaging window (required when FPS changes forcibly)."""
        with self._rate_lock:
            self._rate_active_time = 0.0
            self._rate_frames = 0
            self._avg_fps = 0.0
            self._rate_active_start = None # If we are running, start a fresh active segment now

    def _rate_on_start(self, now: float) -> None:
        """Called when Start pressed: begin (or resume) active segment."""
        with self._rate_lock:
            if self._rate_active_start is None:
                self._rate_active_start = now

    def _rate_on_stop(self, now: float) -> None:
        """Called when Stop pressed: close active segment (exclude idle time)."""
        with self._rate_lock:
            if self._rate_active_start is not None:
                self._rate_active_time += now - self._rate_active_start
                self._rate_active_start = None

    def _rate_tick_frame_displayed(self, now: float) -> None:
        """Called only when a frame is actually displayed."""
        with self._rate_lock:
            if self._rate_active_start is None:
                self._rate_active_start = now # If we somehow display while active_start isn't set, start it.
            self._rate_frames += 1 
            active_time = self._rate_active_time + (now - self._rate_active_start) # does not include down time
            if active_time > 0 and self._rate_frames >= 2:
                self._avg_fps = (self._rate_frames-1) / active_time


    # Internal functions to apply the new average fps
    # ------------------------------------------------------------------
    def _sync_viz_command_from_data(self, now: float) -> None:
        """If Data issued a new viz command, apply it and reset the avg_fps calculation window."""
        ver = self.data.get_viz_cmd_version()
        if ver == self._seen_viz_cmd_version:
            return
        self._seen_viz_cmd_version = ver
        target = self.data.get_viz_target_fps()
        self._apply_new_fps(target, now, update_slider=True, reset_measurement=True)

    def _apply_new_fps(self, new_fps: float, now: float, *, update_slider: bool, reset_measurement: bool) -> None:
        """Apply a new FPS to the scheduler (used by both Data and slider)."""
        fps = max(1.0, float(new_fps))
        self.fps = fps
        self._frame_period = 1.0 / self.fps # our new target period
        self._next_due_time = now + self._frame_period # align schedule
        if reset_measurement:
            self._reset_rate_measurement(now)
        if update_slider and dpg.does_item_exist(self.slider_tag): 
            slider_val = int(round(self.fps))
            self._programmatic_slider_update = True
            dpg.set_value(self.slider_tag, slider_val) # change the slider value as well
            self._programmatic_slider_update = False


# Callbacks (operate via user_data)
# ----------------------------------------------------------------------
def _start_callback(sender, app_data, user_data: Visualization):
    "Start button"
    user_data.running = True
    now = time.perf_counter()

    if not user_data.calibrated: # could be the first start, or a restart after stopping during calibration
        user_data.calibrating = True # enter calibration mode
        if user_data.calib_active_start is None:
            user_data.calib_active_start = now
    else:
        user_data._rate_on_start(now) # start/resume the avg fps measurement. It should be done after calibration
        user_data._next_due_time = now # Normal mode: align schedule so next update can happen promptly

def _stop_callback(sender, app_data, user_data: Visualization):
    "Stop button"
    user_data.running = False
    now = time.perf_counter()
    user_data._rate_on_stop(now) #stop the avg fps measurement for now
    if user_data.calibrating and user_data.calib_active_start is not None: # meaning it was during calibration
        user_data.calib_active_time += now - user_data.calib_active_start # add the interval between start and stop to the active time
        user_data.calib_active_start = None

def _fps_callback(sender, app_data, user_data: Visualization):
    """For add_slider_int, app_data is the slider value"""
    if user_data._programmatic_slider_update:
        return
    now = time.perf_counter()
    user_data.fps = float(max(1, int(app_data))) # avoid negatives
    user_data._frame_period = 1.0 / user_data.fps # new frame visualization period
    user_data._next_due_time = now + user_data._frame_period  # re-align schedule
    user_data._reset_rate_measurement(now)

def _colormap_callback(sender, app_data, user_data: Visualization):
    """For add_combo, app_data is the selected string"""
    name = str(app_data)  # selected map
    if name in user_data.luts:
        user_data.current_cmap = name
        user_data.current_lut = user_data.luts[name]
    _refresh_current_frame(user_data)

def _quantity_callback(sender, app_data, user_data: Visualization):
    user_data.viz_quantity = str(app_data)
    user_data.viz_quantity_index = user_data.quantities.index(app_data)
    _refresh_current_frame(user_data)

def _log_checkbox_callback(sender, app_data, user_data: Visualization):
    user_data.log_scale = bool(app_data)
    dpg.configure_item(user_data.log_strength_tag, enabled=bool(app_data))
    _refresh_current_frame(user_data)

def _log_strength_callback(sender, app_data, user_data: Visualization):
    "Log slider"
    user_data.log_alpha = float(app_data)
    user_data.log_map = build_log_map(user_data.log_alpha) # quick recompute (256 int)
    _refresh_current_frame(user_data)


# Frame rendering and updating
# ----------------------------------------------------------------------
def _render_frame(user_data: Visualization, frame: np.ndarray) -> None:
    frame_norm = normalize_frame(frame[user_data.viz_quantity_index])
    # Apply LUT instead of colormap
    indices = (frame_norm * 255).astype(np.uint8)
    if user_data.log_scale:
        indices = user_data.log_map[indices]
    frame_rgb = user_data.current_lut[indices]
    frame_flattened = frame_rgb.reshape(-1).astype(np.float32) / 255.0
    with dpg.mutex():
        dpg.set_value(user_data.texture_tag, frame_flattened)

def _refresh_current_frame(user_data: Visualization) -> None:
    if user_data._last_frame is None:
        return
    _render_frame(user_data, user_data._last_frame)

def _update_frame(sender, app_data, user_data: Visualization):
    """
    Frame callback.
    Modes:
      1) Calibration (user_data.calibrating == True):
         - While running, display frames as fast as possible (one per callback), pop from Data buffer, and measure max visualisation rate.
         - Stop/Start pauses/resumes calibration timing (only active time counts).
      2) Normal (user_data.calibrating == False):
         - While running, display frames at requested FPS using a scheduled next_due_time (average matches requested FPS).
    """
    now = time.perf_counter()
    user_data._sync_viz_command_from_data(now)

    do_update = False

    if user_data.calibrating: # During calibration, update on every callback while running
        if user_data.running:
            do_update = True
    else: # Normal mode: scheduled updates
        if user_data.frame_index == 0:
            do_update = True
            user_data._next_due_time = now + user_data._frame_period
        elif user_data.running and now >= user_data._next_due_time: # which means we're at or past the scheduled time
            do_update = True
            user_data._next_due_time += user_data._frame_period # advance by ideal frame period rather than now-last to reduce drift
            # optional safety: if we're badly behind, catch up
            if now > user_data._next_due_time + user_data._frame_period:
                user_data._next_due_time = now + user_data._frame_period

    if do_update:
        # Act on the buffer
        frame = user_data.data.pop_frame() # frame consumed
        if frame is None:
            print("[Viz] No more frames to visualise, stopping.")
            user_data.running = False # I guess? So the GUI is not closed automatically at the end
            user_data.finished = True
            return
        else:
            user_data._last_frame = frame # let's cache the frame just in case
            _render_frame(user_data, frame)
            user_data.frame_index += 1

            t_display = time.perf_counter()

            if user_data.running and not user_data.calibrating: # running only after the calibration
                user_data._rate_tick_frame_displayed(t_display)

            # Calibration
            if user_data.calibrating:
                # Ensure active segment is marked if running
                if user_data.calib_active_start is None:
                    user_data.calib_active_start = now

                # Count frames (only frames actually displayed)
                user_data.calib_frame_count += 1

                # Compute active time (exclude idle time)
                active_time = user_data.calib_active_time
                if user_data.calib_active_start is not None:
                    active_time += now - user_data.calib_active_start

                if active_time >= user_data.calib_duration and user_data.calib_frame_count >= user_data.calib_frames:
                    measured_fps = user_data.calib_frame_count / active_time
                    user_data.max_viz_fps = measured_fps

                    user_data.calibrated = True
                    user_data.calibrating = False

                    # Finalise calibration timing state
                    user_data.calib_active_time = 0.0
                    user_data.calib_active_start = None

                    # Set normal-mode FPS to measured max
                    user_data.fps = measured_fps
                    user_data._frame_period = 1.0 / user_data.fps
                    user_data._next_due_time = now + user_data._frame_period

                    max_slider = max(1, round(measured_fps))
                    print(f"[Viz] Calibration complete: max viz FPS ≈ {max_slider} Hz")
                    user_data.data.max_viz_fps = measured_fps

                    # Resize slider to [1, measured_max], set it to max, and enable it from now on
                    dpg.configure_item(
                        user_data.slider_tag,
                        min_value=1,
                        max_value=max_slider,
                        default_value=max_slider,
                        enabled=True,
                    )
                    user_data._programmatic_slider_update = True
                    dpg.set_value(user_data.slider_tag, max_slider) # update it (need to set up an "updating" flag first)
                    user_data._programmatic_slider_update = False

    # Re-register callback one (or two) dpg frame ahead
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 1 # +2 halves the viz_fps
        dpg.set_frame_callback(target_frame, _update_frame, user_data=user_data)
