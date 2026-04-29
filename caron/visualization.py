import math
import time
import numpy as np
from numba import njit
from matplotlib import colormaps as cm
import threading
import dearpygui.dearpygui as dpg
from caron.data import Data


def build_log_map(alpha=50.0):
    """Maps linear [0..255] -> log-compressed [0..255]."""
    x = np.linspace(0.0, 1.0, 256)
    y = np.log1p(alpha * x) / np.log1p(alpha)
    return (y * 255.0).astype(np.uint8)

@njit(cache=True)
def apply_colormap_to_buffer(field, vmin, vmax, lut, log_map, use_log, out):
    """Normalize field, apply LUT, write float32 RGB into pre-allocated out (flat nx*ny*3)."""
    nx, ny = field.shape
    scale = 1.0 / (vmax - vmin + 1e-8)
    k = 0
    for i in range(nx):
        for j in range(ny):
            v = (field[i, j] - vmin) * scale
            if v != v:   # NaN → clamp to 0
                v = 0.0
            elif v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            idx = int(v * 255.0)
            if idx > 255:
                idx = 255
            if use_log:
                idx = int(log_map[idx])
            out[k]     = lut[idx, 0] / 255.0
            out[k + 1] = lut[idx, 1] / 255.0
            out[k + 2] = lut[idx, 2] / 255.0
            k += 3


class Visualization:
    """Consume frames from a Data buffer and display them with DearPyGui."""


    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, data:Data, args, sim=None) -> None:
        self.data = data # this is the first buffer, obtained after a few seconds of simulation
        self.args = args
        self.sim = sim
        self.frames = self.data.buffer
        self.no_sim_source_frames: list[np.ndarray] = []
        if self.args.no_sim:
            frames_array = np.load(self.args.sim_file)
            self.no_sim_source_frames = [np.array(frame) for frame in frames_array]
            for frame in frames_array:
                self.data.push_frame(frame)
            if not self.data.quantities:
                n_ch = int(frames_array.shape[1])
                self.data.quantities = [f"Channel {i}" for i in range(n_ch)]

        # Simulation shape (supports rectangular grids nx != ny)
        self.sim_nx = int(self.args.sim_size)
        self.sim_ny = int(self.args.sim_size)

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
        self.quantities = self.data.quantities
        self.viz_quantity_index: int = 0
        self.viz_quantity: str = self.quantities[0] if self.quantities else ""

        # Sliders and tags
        self._programmatic_slider_update = False
        self.slider_tag = "Caron FPS Slider"
        self.cmap_list_tag = "Caron Colormaps List"
        self.quantity_tag = "Caron Quantities List"
        self.log_strength_tag = "Caron Log Strength Slider"
        self.restart_tag = "Caron Restart Button"
        self.texture_tag = "frame_tag"
        self.main_window_tag = "Caron Main Window"
        self.log_scale: bool = False
        self.log_alpha: float = 50.0 # compression log strength
        self.log_map = build_log_map(self.log_alpha) # basically mapping the log scale once on a 1D array

        # Simulation time display
        self.sim_time_tag = "Caron Sim Time"
        self.exit_modal_tag = "Caron Exit Modal"

        # Pre-allocated render buffers, one per quantity (set in run() once resolution is known)
        self._render_bufs: list[np.ndarray] = []

        # All-quantities view
        self.show_all: bool = False
        self._all_mode_initialized: bool = False  # textures+widgets created on first toggle
        self.all_texture_tags: list[str] = []
        self._grid_img_w: int = 0   # set in run(), used by lazy init
        self._grid_img_h: int = 0
        self._n_cols_grid: int = 0
        self._n_rows_grid: int = 0
        self.single_view_group_tag = "Caron Single View Group"
        self.all_view_group_tag = "Caron All View Group"
        self.show_all_tag = "Caron Show All Checkbox"

        # Colorscale limits
        self.clim_auto: bool = True
        self.clim_vmin: float = 0.0
        self.clim_vmax: float = 1.0
        self.clim_auto_tag = "Caron Clim Auto"
        self.clim_min_tag  = "Caron Clim Min"
        self.clim_max_tag  = "Caron Clim Max"

    def _compute_layout_size(self, sim_nx: int, sim_ny: int) -> tuple[int, int, int, int]:
        """
        Return (image_w, image_h, window_width, window_height) fitted to max dimensions.
        Keeps aspect ratio for rectangular fields.
        """
        max_w = int(getattr(self.args, "max_window_width", 1400))
        max_h = int(getattr(self.args, "max_window_height", 1000))
        cap_img = int(self.args.max_display_px)

        # Initial estimate only; final size is tightened to real content later.
        side_panel_w = 170.0
        horiz_pad = 70.0
        vert_pad = 120.0

        max_img_w = max(120.0, min(float(cap_img), float(max_w) - side_panel_w - horiz_pad))
        max_img_h = max(120.0, min(float(cap_img), float(max_h) - vert_pad))
        scale = min(max_img_w / max(float(sim_nx), 1.0), max_img_h / max(float(sim_ny), 1.0))

        image_w = max(120, int(round(sim_nx * scale)))
        image_h = max(120, int(round(sim_ny * scale)))
        win_w = int(round(image_w + side_panel_w + horiz_pad))
        win_h = int(round(image_h + vert_pad))
        return image_w, image_h, win_w, win_h


    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Check the buffer, set up DearPyGui and start the visualisation."""

        if not self.frames:
            raise RuntimeError("[Viz] Visualization initialised with an empty buffer.")
        if not self.quantities:
            raise RuntimeError("[Viz] No quantities defined. Simulation must set data.quantities before run().")
        self.viz_quantity = self.quantities[self.viz_quantity_index]
        if self.args.verbose:
            print(f"[Viz] Loaded a simulation with {len(self.frames)} frames as initial buffering.")

        # Precompute first frame to initialise the texture
        first = self.frames[0]
        first_frame = first[0] if isinstance(first, tuple) else first  # unwrap (frame, t) if needed
        frame = first_frame[self.viz_quantity_index]
        self.sim_nx = int(frame.shape[0])
        self.sim_ny = int(frame.shape[1])
        self._render_bufs = [
            np.zeros(self.sim_nx * self.sim_ny * 3, dtype=np.float32)
            for _ in range(len(self.quantities))
        ]
        image_w, image_h, window_width, window_height = self._compute_layout_size(self.sim_nx, self.sim_ny)
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
        frame_rgb = np.stack((frame_norm,) * 3, axis=-1).astype(np.float32)
        self.frame_flattened: np.ndarray = frame_rgb.flatten()

        # Store grid layout params for lazy all-mode initialisation
        n_qty = len(self.quantities)
        self._n_cols_grid = max(1, math.ceil(math.sqrt(n_qty)))
        self._n_rows_grid = math.ceil(n_qty / self._n_cols_grid)
        _grid_gap = 4
        self._grid_img_w = max(60, (image_w - _grid_gap * (self._n_cols_grid - 1)) // self._n_cols_grid)
        _aspect = self.sim_ny / max(self.sim_nx, 1)
        self._grid_img_h = max(60, int(round(self._grid_img_w * _aspect)))

        # DearPyGui setup
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_raw_texture(
                int(self.sim_nx),    # texture must match actual frame resolution
                int(self.sim_ny),
                default_value=self.frame_flattened.tolist(),
                format=dpg.mvFormat_Float_rgb,
                tag=self.texture_tag,
            )

        # Window / layout
        with dpg.window(
            tag=self.main_window_tag,
            label="Jet Inspector",
            width=window_width,
            height=window_height,
            no_close=True,
            no_move=True,
            no_resize=False,
        ):
            with dpg.group(label="Visualizator"): # the big visualization window
                with dpg.group(label="Slider"): # above there is the slider
                    dpg.add_slider_int(
                        label="Speed (FPS)",
                        tag=self.slider_tag,
                        width=int(image_w),
                        default_value=int(self.fps),
                        min_value=1,
                        max_value=int(self.data.max_viz_fps),
                        callback=_fps_callback,
                        user_data=self,
                        enabled=False, # don't allow changes until the calibration is finished
                    )
                with dpg.group(label="Map & Keys", horizontal=True): # below there is the image and the buttons
                    with dpg.group(label='Colormap & Color Combo'): # image and color listbox on the left
                        # Single-quantity view (default)
                        with dpg.group(tag=self.single_view_group_tag, show=True):
                            dpg.add_image(texture_tag=self.texture_tag, width=int(image_w), height=int(image_h))
                        # All-quantities grid view — populated lazily on first toggle
                        with dpg.group(tag=self.all_view_group_tag, show=False):
                            pass
                        with dpg.group(label="Log visualization", horizontal=True):
                            dpg.add_combo(
                                tag=self.cmap_list_tag,
                                items=self.cmap_names,
                                default_value=self.current_cmap,
                                width=max(100, int(image_w * 0.35)),
                                callback=_colormap_callback,
                                user_data=self,
                            )
                            dpg.add_combo(
                                tag=self.quantity_tag,
                                items=self.quantities,
                                default_value=self.viz_quantity,
                                width=max(100, int(image_w * 0.35)),
                                callback=_quantity_callback,
                                user_data=self,
                            )
                            dpg.add_checkbox(
                                label="All",
                                tag=self.show_all_tag,
                                default_value=self.show_all,
                                callback=_toggle_all_callback,
                                user_data=self,
                            )
                        with dpg.group(label="Colorscale", horizontal=True):
                            dpg.add_checkbox(
                                label="Auto",
                                tag=self.clim_auto_tag,
                                default_value=self.clim_auto,
                                callback=_clim_auto_callback,
                                user_data=self,
                            )
                            dpg.add_input_float(
                                label="Min",
                                tag=self.clim_min_tag,
                                default_value=self.clim_vmin,
                                width=max(120, int(image_w * 0.4)),
                                enabled=False,
                                callback=_clim_min_callback,
                                user_data=self,
                                step=0,
                            )
                            dpg.add_input_float(
                                label="Max",
                                tag=self.clim_max_tag,
                                default_value=self.clim_vmax,
                                width=max(120, int(image_w * 0.4)),
                                enabled=False,
                                callback=_clim_max_callback,
                                user_data=self,
                                step=0,
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
                                width=max(220, int(image_w * 0.95)),
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
                            width=140,
                            height=44,
                        )
                        dpg.add_button( # the stop button below
                            label="Stop",
                            callback=_stop_callback,
                            user_data=self,
                            width=140,
                            height=44,
                        )
                        dpg.add_button(
                            label="Restart",
                            tag=self.restart_tag,
                            callback=_restart_callback,
                            user_data=self,
                            width=140,
                            height=44,
                        )
                        dpg.add_text("t = -.----", tag=self.sim_time_tag)

            _btn_w, _btn_h = 55, 20
            dpg.add_button(
                label="Quit",
                callback=_exit_callback,
                width=_btn_w,
                height=_btn_h,
                pos=[window_width - _btn_w - 8, window_height - _btn_h - 8],
            )

        _modal_w, _modal_h = 280, 100
        with dpg.window(
            label="Confirm Close",
            modal=True,
            no_close=True,
            show=False,
            tag=self.exit_modal_tag,
            width=_modal_w,
            height=_modal_h,
            no_resize=True,
        ):
            dpg.add_text("Close the visualizer?")
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Yes, close", callback=lambda: dpg.stop_dearpygui(), width=120)
                dpg.add_spacer(width=6)
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.configure_item(self.exit_modal_tag, show=False),
                    width=120,
                )

        dpg.create_viewport(
            title="Our lovely Caron",
            width=window_width,
            height=window_height,
        )

        dpg.setup_dearpygui()
        if hasattr(dpg, 'set_exit_callback'):
            dpg.set_exit_callback(_exit_callback)
        else:
            print("[Viz] DPG version does not support set_exit_callback; use the Quit button to exit safely.")
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            _update_frame(None, None, self)
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


    # Command getters (called by Data)
    # ------------------------------------------------------------------
    def get_measured_fps(self) -> float:
        """Return average FPS over active display time."""
        with self._rate_lock:
            return self._avg_fps

    
    # Internal functions to calculate the average fps
    # ------------------------------------------------------------------
    def _reset_rate_measurement(self, _now: float) -> None:
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
def _exit_callback(_sender=None, _app_data=None, _user_data=None):
    """Show the close-confirmation modal (used by both X button and Quit button)."""
    try:
        dpg.configure_item("Caron Exit Modal", show=True)
    except Exception as e:
        print(f"[Viz] Exit callback error: {e}")

def _start_callback(_sender, _app_data, user_data: Visualization):
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

def _stop_callback(_sender, _app_data, user_data: Visualization):
    "Stop button"
    user_data.running = False
    now = time.perf_counter()
    user_data._rate_on_stop(now) #stop the avg fps measurement for now
    if user_data.calibrating and user_data.calib_active_start is not None: # meaning it was during calibration
        user_data.calib_active_time += now - user_data.calib_active_start # add the interval between start and stop to the active time
        user_data.calib_active_start = None

def _restart_callback(_sender, _app_data, user_data: Visualization):
    """Reset visualization and simulation streams to t=0."""
    now = time.perf_counter()
    user_data.running = False
    user_data.finished = False
    user_data._rate_on_stop(now)
    user_data._reset_rate_measurement(now)
    user_data.frame_index = 0

    first_frame = None
    if user_data.args.no_sim:
        if user_data.no_sim_source_frames:
            user_data.data.reset_buffer(user_data.no_sim_source_frames)
            first_frame = user_data.no_sim_source_frames[0]
    elif user_data.args.fake_injection:
        # mock mode has no initial_prim — just clear the buffer and let the sim refill from frame 0
        user_data.data.reset_buffer()
        if user_data.sim is not None:
            user_data.sim.request_restart()
    else:
        if user_data.sim is not None:
            initial_item = user_data.sim.get_initial_frame_tuple()
            user_data.data.reset_buffer([initial_item])
            first_frame = initial_item[0]
            user_data.sim.request_restart()

    if first_frame is not None:
        user_data._last_frame = first_frame
        _render_frame(user_data, first_frame)
    dpg.set_value(user_data.sim_time_tag, "t = 0.0000")

    user_data.data.reset_viz_target_fps()
    user_data.running = True
    user_data._rate_on_start(time.perf_counter())
    user_data._next_due_time = time.perf_counter()

def _fps_callback(_sender, app_data, user_data: Visualization):
    """For add_slider_int, app_data is the slider value"""
    if user_data._programmatic_slider_update:
        return
    now = time.perf_counter()
    new_fps = float(max(1, int(app_data)))
    user_data.fps = new_fps
    user_data._frame_period = 1.0 / user_data.fps
    user_data._next_due_time = now + user_data._frame_period
    user_data._reset_rate_measurement(now)
    user_data.data.update_viz_target_fps_from_user(new_fps)  # keep Data in sync so underflow can re-cap if needed

def _colormap_callback(_sender, app_data, user_data: Visualization):
    """For add_combo, app_data is the selected string"""
    name = str(app_data)  # selected map
    if name in user_data.luts:
        user_data.current_cmap = name
        user_data.current_lut = user_data.luts[name]
    _refresh_current_frame(user_data)

def _quantity_callback(_sender, app_data, user_data: Visualization):
    user_data.viz_quantity = str(app_data)
    user_data.viz_quantity_index = user_data.quantities.index(app_data)
    _refresh_current_frame(user_data)

def _toggle_all_callback(_sender, app_data, user_data: Visualization):
    user_data.show_all = bool(app_data)
    if user_data.show_all and not user_data._all_mode_initialized:
        _init_all_mode(user_data)
    dpg.configure_item(user_data.single_view_group_tag, show=not user_data.show_all)
    dpg.configure_item(user_data.all_view_group_tag, show=user_data.show_all)
    dpg.configure_item(user_data.quantity_tag, enabled=not user_data.show_all)
    _refresh_current_frame(user_data)

def _init_all_mode(user_data: Visualization) -> None:
    """Create all-mode textures and image widgets on first toggle (lazy init)."""
    n_qty = len(user_data.quantities)
    with dpg.texture_registry():
        for i in range(n_qty):
            tag = f"frame_tag_all_{i}"
            user_data.all_texture_tags.append(tag)
            dpg.add_raw_texture(
                user_data.sim_nx, user_data.sim_ny,
                default_value=user_data._render_bufs[i].tolist(),
                format=dpg.mvFormat_Float_rgb,
                tag=tag,
            )
    for row in range(user_data._n_rows_grid):
        row_group = dpg.add_group(horizontal=True, parent=user_data.all_view_group_tag)
        for col in range(user_data._n_cols_grid):
            i = row * user_data._n_cols_grid + col
            if i >= n_qty:
                break
            cell = dpg.add_group(parent=row_group)
            dpg.add_text(user_data.quantities[i], parent=cell)
            dpg.add_image(
                texture_tag=user_data.all_texture_tags[i],
                width=user_data._grid_img_w,
                height=user_data._grid_img_h,
                parent=cell,
            )
    user_data._all_mode_initialized = True

def _log_checkbox_callback(_sender, app_data, user_data: Visualization):
    user_data.log_scale = bool(app_data)
    dpg.configure_item(user_data.log_strength_tag, enabled=bool(app_data))
    _refresh_current_frame(user_data)

def _log_strength_callback(_sender, app_data, user_data: Visualization):
    "Log slider"
    user_data.log_alpha = float(app_data)
    user_data.log_map = build_log_map(user_data.log_alpha) # quick recompute (256 int)
    _refresh_current_frame(user_data)


def _clim_auto_callback(_sender, app_data, user_data: Visualization):
    user_data.clim_auto = bool(app_data)
    enabled = not user_data.clim_auto
    dpg.configure_item(user_data.clim_min_tag, enabled=enabled)
    dpg.configure_item(user_data.clim_max_tag, enabled=enabled)
    _refresh_current_frame(user_data)

def _clim_min_callback(_sender, app_data, user_data: Visualization):
    if user_data.clim_auto:
        return
    user_data.clim_vmin = float(app_data)
    _refresh_current_frame(user_data)

def _clim_max_callback(_sender, app_data, user_data: Visualization):
    if user_data.clim_auto:
        return
    user_data.clim_vmax = float(app_data)
    _refresh_current_frame(user_data)


# Frame rendering and updating
# ----------------------------------------------------------------------
def _render_quantity_to_texture(
    user_data: Visualization,
    frame: np.ndarray,
    qty_idx: int,
    tex_tag: str,
    *,
    update_clim_display: bool = False,
    force_auto_clim: bool = False,
) -> None:
    """Apply colormap pipeline to one quantity channel and upload to a DPG texture."""
    field = np.asarray(frame[qty_idx], dtype=np.float32)
    auto = user_data.clim_auto or force_auto_clim
    if auto:
        vmin = float(field.min())
        vmax = float(field.max())
        if update_clim_display:
            user_data._programmatic_slider_update = True
            dpg.set_value(user_data.clim_min_tag, vmin)
            dpg.set_value(user_data.clim_max_tag, vmax)
            user_data._programmatic_slider_update = False
    else:
        vmin = user_data.clim_vmin
        vmax = user_data.clim_vmax
    buf = user_data._render_bufs[qty_idx]
    apply_colormap_to_buffer(
        field, vmin, vmax,
        user_data.current_lut, user_data.log_map, user_data.log_scale,
        buf,
    )
    dpg.set_value(tex_tag, buf)

def _render_frame(user_data: Visualization, frame: np.ndarray) -> None:
    if user_data.show_all:
        for i, tag in enumerate(user_data.all_texture_tags):
            _render_quantity_to_texture(user_data, frame, i, tag, force_auto_clim=True)
    else:
        _render_quantity_to_texture(
            user_data, frame, user_data.viz_quantity_index, user_data.texture_tag,
            update_clim_display=True,
        )

def _refresh_current_frame(user_data: Visualization) -> None:
    if user_data._last_frame is None:
        return
    _render_frame(user_data, user_data._last_frame)

def _update_frame(_sender, _app_data, user_data: Visualization):
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
        item = user_data.data.pop_frame() # frame consumed
        if item is None:
            if user_data.data.sim_finished:
                print("[Viz] No more frames to visualise, stopping.")
                user_data.running = False
                user_data.finished = True
            # else: sim still running but buffer temporarily empty — hold last frame
        else:
            frame, sim_t = item if isinstance(item, tuple) else (item, None)
            user_data._last_frame = frame # let's cache the frame just in case
            _render_frame(user_data, frame)
            if sim_t is not None:
                dpg.set_value(user_data.sim_time_tag, f"t = {sim_t:.4f}")
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
                    max_slider = max(1, round(measured_fps))
                    print(f"[Viz] Calibration complete: max viz FPS ≈ {max_slider} Hz")
                    user_data.data.max_viz_fps = measured_fps

                    # Start at user-requested FPS (clamped to measured max)
                    start_fps = min(float(user_data.args.viz_fps), measured_fps)
                    user_data.fps = start_fps
                    user_data._frame_period = 1.0 / user_data.fps
                    user_data._next_due_time = now + user_data._frame_period

                    # Resize slider to [1, measured_max], set it to start_fps, and enable it from now on
                    start_slider = max(1, round(start_fps))
                    dpg.configure_item(
                        user_data.slider_tag,
                        min_value=1,
                        max_value=max_slider,
                        default_value=start_slider,
                        enabled=True,
                    )
                    user_data._programmatic_slider_update = True
                    dpg.set_value(user_data.slider_tag, start_slider)
                    user_data._programmatic_slider_update = False

