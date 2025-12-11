import time
import numpy as np
from matplotlib import colormaps as cm 
import dearpygui.dearpygui as dpg
from numba import njit


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
    A thin OO wrapper around the existing Caron visualiser.

    For now this class:
      * Loads 'Simulation.npy'
      * Sets up DearPyGui with the same controls as before
      * Uses the same update_frame scheduling (set_frame_callback)
    """

    def __init__(self, sim_file: str = "Simulation.npy") -> None:
        # Load frames exactly as before
        self.frames: np.ndarray = np.load(sim_file)
        self.sim_size: int = self.frames.shape[1]
        print(
            f"Loaded simulation with {self.frames.shape[0]} "
            f"frames of size {self.sim_size}x{self.sim_size}"
        )

        # Playback state (were globals before)
        self.running: bool = False
        self.speed: int = 10  # interpreted as FPS (slider label says FPS)
        self.last_update_time: float = time.time()
        self.frame_index: int = 0

        # Precompute first frame to initialise the texture
        frame = self.frames[0]
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

        # DearPyGui bookkeeping
        self._texture_tag: str = "frame_tag"
        self._font_tag: str = "big_font"

    # Runs
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Set up DearPyGui and start the visualisation.
        """
        dpg.create_context()

        # If you want the font back, uncomment the registry and bind_font below
        # with dpg.font_registry():
        #     dpg.add_font("C:/Windows/Fonts/BKANT.TTF", 20, tag=self._font_tag)

        with dpg.texture_registry(show=True):
            # Same call as before, just expanded the ellipsis:
            dpg.add_raw_texture(
                int(self.sim_size),
                int(self.sim_size),
                default_value=self.frame_flattened,
                format=dpg.mvFormat_Float_rgb,
                tag=self._texture_tag,
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
                        default_value=self.speed,
                        min_value=1,
                        max_value=60,
                        callback=_speed_callback,
                        user_data=self,
                    )
                    dpg.add_image(self._texture_tag)
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

        # Kick off the first update, as in the original script
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
    user_data.speed = int(app_data)


def _update_frame(sender, app_data, user_data: Visualization):
    """
    This is almost identical to the original update_frame(),
    but everything lives on 'user_data' instead of globals.
    """
    current_time = time.time()

    # Same timing condition as before
    if (user_data.running and (current_time - user_data.last_update_time) >= 1 / user_data.speed) or user_data.frame_index == 0:
        user_data.last_update_time = current_time

        frame = user_data.frames[user_data.frame_index]
        frame_norm = normalize_frame(frame)

        # Apply LUT instead of colormap (same optimisation as before)
        indices = (frame_norm * 255).astype(np.uint8)
        frame_rgb = user_data.inferno_lut[indices]
        frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0

        dpg.set_value(user_data._texture_tag, frame_flattened)

        user_data.frame_index = (user_data.frame_index + 1) % len(user_data.frames)

    # Same scheduling logic: re-register callback a couple of frames ahead
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 2
        dpg.set_frame_callback(target_frame, _update_frame, user_data=user_data)
