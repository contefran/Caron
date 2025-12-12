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
        self.fps: int = self.args.viz_fps
        self.last_update_time: float = time.time()
        self.frame_index: int = 0

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
            width=int(self.sim_size * 1.1),
            height=int(self.sim_size),
            no_close=True,
            no_move=True,
            no_resize=False,
        ):
            # If using the font:
            # dpg.bind_font(self._font_tag)

            with dpg.group(label="Visualizator", horizontal=True):
                with dpg.group(label="Map and slider"):
                    dpg.add_slider_int(
                        label="Speed (FPS)",
                        height=40,
                        default_value=self.fps,
                        min_value=1,
                        max_value=self.fps,
                        callback=_speed_callback,
                        user_data=self,
                    )
                    dpg.add_image("frame_tag")
                with dpg.group(label="Start&Stop"):
                    dpg.add_button(
                        label="Start",
                        callback=_start_callback,
                        user_data=self,
                        width=80,
                        height=200,
                    )
                    dpg.add_button(
                        label="Stop",
                        callback=_stop_callback,
                        user_data=self,
                        width=80,
                        height=200,
                    )

        dpg.create_viewport(
            title="Our lovely Caron",
            width=int(self.sim_size * 1.1),
            height=int(self.sim_size),
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


def _stop_callback(sender, app_data, user_data: Visualization):
    user_data.running = False


def _speed_callback(sender, app_data, user_data: Visualization):
    # app_data is the slider value
    user_data.fps = int(app_data)


def _update_frame(sender, app_data, user_data: Visualization):
    """ Everything lives on user_data instead of self or globals."""
    current_time = time.time()

    # Check timing
    if (user_data.running and (current_time - user_data.last_update_time) >= 1 / user_data.fps) or user_data.frame_index == 0:
        print(f"time since last update: {current_time - user_data.last_update_time:.3f}s")
        user_data.last_update_time = current_time

        # Act on the buffer
        frame = user_data.data.pop_frame()
        if frame is None:
            print("No more frames to visualise, stopping.")
            user_data.running = False # I guess?
            return
        else:
            frame_norm = normalize_frame(frame)

            # Apply LUT instead of colormap
            indices = (frame_norm * 255).astype(np.uint8)
            frame_rgb = user_data.inferno_lut[indices]
            frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0
            dpg.set_value("frame_tag", frame_flattened)
            user_data.frame_index += 1 # just increment, stop when no more frames
        
    # Re-register callback a couple of frames ahead
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 1
        dpg.set_frame_callback(target_frame, _update_frame, user_data=user_data)
