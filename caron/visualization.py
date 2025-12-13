import time
import numpy as np
from matplotlib import colormaps as cm
import dearpygui.dearpygui as dpg
from numba import njit
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
      * We assume that the simulation has already filled the buffer.
      * We simply pop frames from Data at a rate set by 'fps'.
    Phase 2:
      * We feed the buffer with the mock simulation, but at a certain rate
    Phase 3:
      * We feed the buffer with a real-time simulation.
    """

    def __init__(self, data:Data, args) -> None:
        self.data = data # this is the first buffer, obtained after a few seconds of simulation
        self.args = args
        self.frames = self.data.buffer
        if self.args.no_sim:
            frames_array = np.load(self.args.sim_file)
            for frame in frames_array:
                self.data.push_frame(frame) # fill the buffer with the mock sim. It's a deque
        self.frames = self.data.buffer  # already a deque, as set in Data
        if not self.frames:
            raise RuntimeError("Visualization initialised with an empty buffer.")

        self.sim_size: int = self.frames[0].shape[1]
        print(f"Loaded simulation with {len(self.frames)} frames of size {self.sim_size}x{self.sim_size} as initial buffering.")

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

        # Precompute first frame to initialise the texture
        frame = self.frames[0] # still works with deque, I'm already in love
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
        frame_rgb = np.stack((frame_norm,) * 3, axis=-1)
        frame_rgb = frame_rgb.astype(np.float32) / np.max(frame_rgb)
        self.frame_flattened: np.ndarray = frame_rgb.flatten()

        # LUT for inferno colormap (same as before)
        self.inferno_lut: np.ndarray = (
            cm['inferno'](np.linspace(0, 1, 256))[:, :3] * 255
        ).astype(np.uint8)

        # DearPyGui tags
        #self._font_tag: str = "big_font"

    # Runs
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Set up DearPyGui and start the visualisation."""
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
                        tag="Caron FPS Slider",
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
        # Normal mode: align schedule so next update can happen promptly
        user_data._next_due_time = now


def _stop_callback(sender, app_data, user_data: Visualization):
    user_data.running = False
    now = time.perf_counter()
    # add this segment's duration to the total active time.
    if user_data.calibrating and user_data.calib_active_start is not None: # meaning it was during calibration
        user_data.calib_active_time += now - user_data.calib_active_start # add to active time the time between start and stop
        user_data.calib_active_start = None


def _fps_callback(sender, app_data, user_data: Visualization):
    # app_data is the slider value
    new_fps = max(0, int(app_data))  # avoid negatives
    user_data.fps = float(new_fps)
    user_data._frame_period = 1.0 / user_data.fps


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
        print(f"time since last update: {now - user_data.last_update_time:.3f}s") # the real time between updates
        user_data.last_update_time = now

        # Act on the buffer
        frame = user_data.data.pop_frame() # frame consumed
        if frame is None:
            print("No more frames to visualise, stopping.")
            user_data.running = False # I guess? So the GUI is not closed
            return
        else:
            frame_norm = normalize_frame(frame)
            # Apply LUT instead of colormap
            indices = (frame_norm * 255).astype(np.uint8)
            frame_rgb = user_data.inferno_lut[indices]
            frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0
            dpg.set_value("frame_tag", frame_flattened)
            user_data.frame_index += 1

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
                    print(f"Calibration complete: max viz FPS ≈ {measured_fps:.2f}")

                    # Resize slider to [1, measured_max] and set it to max
                    dpg.configure_item(
                        "Caron FPS Slider",
                        min_value=1,
                        max_value=max_slider,
                        default_value=max_slider,
                    )

    # Re-register callback one (or two) dpg frame ahead
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 1 # +2
        dpg.set_frame_callback(target_frame, _update_frame, user_data=user_data)
