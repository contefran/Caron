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
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] < 0:
                norm[i, j] = 0.0
            elif norm[i, j] > 1.0:
                norm[i, j] = 1.0
    return norm


class Visualization:
    """
    Consume frames from a Data buffer and display them with DearPyGui.
    Phase 1:
      *[V] We assume that the simulation has already filled the buffer.
      *[V] We simply pop frames from Data at a rate set by 'fps'.
    Phase 2:
      * We feed the buffer with the mock simulation, but at a certain rate
    Phase 3:
      * We feed the buffer with a real-time simulation.
    """


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

        self.sim_size = self.args.sim_size
        if self.args.no_sim:
            self.sim_size: int = self.frames[0].shape[1]

        self.finished: bool = False
        self.running: bool = False
        self.fps = float(self.args.viz_fps)
        self.max_viz_fps: float = self.fps  # after calibration will be changed to the actual max FPS
        self._frame_period = 1.0 / self.fps
        self._next_due_time = time.perf_counter()  # better clock for intervals

        # For debugging prints only (actual time between updates)
        self.last_update_time: float = self._next_due_time
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

        # Slider
        self._programmatic_slider_update = False # to update the slider when the fps is forced down by the buffer
        self.slider_tag = "Caron FPS Slider"
        # DearPyGui tags
        #self._font_tag: str = "big_font"


    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Check the buffer, set up DearPyGui and start the visualisation."""

        # Call the buffer
        self.frames = self.data.buffer  # already a deque, as set in Data. Here it should be already populated by the full sim (no_sim) or by the injection (fake or not)
        if not self.frames: # and it would be very weird
            raise RuntimeError("[Viz] Visualization initialised with an empty buffer.")
        if self.args.verbose:
            print(f"[Viz] Loaded a simulation of size {self.sim_size}x{self.sim_size} with {len(self.frames)} frames as initial buffering.")

        # Precompute first frame to initialise the texture
        frame = self.frames[0] # still works with deque, I'm already in love
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
        frame_rgb = np.stack((frame_norm,) * 3, axis=-1)
        frame_rgb = frame_rgb.astype(np.float32) / np.max(frame_rgb)
        self.frame_flattened: np.ndarray = frame_rgb.flatten()

        # LUT for inferno colormap
        self.inferno_lut: np.ndarray = (cm['inferno'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        # DearPyGui setup
        dpg.create_context()

        # To get the font back, uncomment the registry and bind_font below
        # with dpg.font_registry():
        #     dpg.add_font("C:/Windows/Fonts/BKANT.TTF", 20, tag=self._font_tag)

        with dpg.texture_registry(show=True):
            dpg.add_raw_texture(
                int(self.sim_size),
                int(self.sim_size),
                default_value=self.frame_flattened.tolist(),
                format=dpg.mvFormat_Float_rgb,
                tag="frame_tag",
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
            # If using the font:
            # dpg.bind_font(self._font_tag)

            with dpg.group(label="Visualizator"): # the big visualization window
                with dpg.group(label="Slider"): # above there is the slider
                    dpg.add_slider_int(
                        label="Speed (FPS)",
                        tag=self.slider_tag,
                        width=int(self.sim_size*2),
                        height=int(self.sim_size*1),
                        default_value=int(self.fps),
                        min_value=1,
                        max_value=int(self.fps),
                        callback=_fps_callback,
                        user_data=self,
                    )
                with dpg.group(label="Map & Keys", horizontal=True): # below there is the image and the buttons
                    dpg.add_image("frame_tag", width = self.sim_size *2, height= self.sim_size *2) # the image on the left
                    with dpg.group(label="Start&Stop"): # the buttons on the right
                        dpg.add_button( # the start button above
                            label="Start",
                            callback=_start_callback,
                            user_data=self,
                            width=int(self.sim_size*0.1),
                            height=int(self.sim_size*0.997),
                        )
                        dpg.add_button( # the stop button below
                            label="Stop",
                            callback=_stop_callback,
                            user_data=self,
                            width=int(self.sim_size*0.1),
                            height=int(self.sim_size*0.997),
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
        try: # Just in case data wants to change something during the iteration
            ver = self.data.get_viz_cmd_version()
        except Exception:
            ver = getattr(self.data, "viz_cmd_version", 0)

        if ver == self._seen_viz_cmd_version: # no change happened
            return

        self._seen_viz_cmd_version = ver # change obviously happened
        target = self.data.get_viz_target_fps() # get the needed fps as calculated by Data
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
            cfg = dpg.get_item_configuration(self.slider_tag)
            vmin = int(cfg.get("min_value", 1))
            vmax = int(cfg.get("max_value", max(1, int(round(self.fps)))))
            slider_val = int(round(self.fps))
            #slider_val = max(vmin, min(vmax, slider_val))

            self._programmatic_slider_update = True
            dpg.set_value(self.slider_tag, slider_val) # change the slider value as well
            self._programmatic_slider_update = False


# Callbacks (operate via user_data)
# ----------------------------------------------------------------------
def _start_callback(sender, app_data, user_data: Visualization):
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
    user_data.running = False
    now = time.perf_counter()
    user_data._rate_on_stop(now) #stop the avg fps measurement for now
    if user_data.calibrating and user_data.calib_active_start is not None: # meaning it was during calibration
        user_data.calib_active_time += now - user_data.calib_active_start # add the interval between start and stop to the active time
        user_data.calib_active_start = None

def _fps_callback(sender, app_data, user_data: Visualization):
    """ app_data is the slider value """
    if user_data._programmatic_slider_update:
        return
    now = time.perf_counter()
    user_data.fps = float(max(1, int(app_data))) # avoid negatives
    user_data._frame_period = 1.0 / user_data.fps # new frame visualization period
    user_data._next_due_time = now + user_data._frame_period  # re-align schedule
    user_data._reset_rate_measurement(now)

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
        # Debug: time between displayed frames
        #print(f"[Viz] time since last update: {now - user_data.last_update_time:.3f}s") # the real time between updates
        user_data.last_update_time = now

        # Act on the buffer
        frame = user_data.data.pop_frame() # frame consumed
        if frame is None:
            print("[Viz] No more frames to visualise, stopping.")
            user_data.running = False # I guess? So the GUI is not closed automatically at the end
            user_data.finished = True
            return
        else:
            frame_norm = normalize_frame(frame)
            # Apply LUT instead of colormap
            indices = (frame_norm * 255).astype(np.uint8)
            frame_rgb = user_data.inferno_lut[indices]
            frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0
            dpg.set_value("frame_tag", frame_flattened)
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

                    max_slider = max(1, int(measured_fps))
                    print(f"[Viz] Calibration complete: max viz FPS ≈ {measured_fps:.2f}")

                    # Resize slider to [1, measured_max] and set it to max
                    dpg.configure_item(
                        "Caron FPS Slider",
                        min_value=1,
                        max_value=max_slider,
                        default_value=max_slider,
                    )

    # Re-register callback one (or two) dpg frame ahead
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 1 # +2 halves the viz_fps
        dpg.set_frame_callback(target_frame, _update_frame, user_data=user_data)
