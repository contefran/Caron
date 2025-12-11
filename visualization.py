# imports
import numpy as np
from matplotlib import colormaps as cm
import time
import dearpygui.dearpygui as dpg
from numba import njit
from data import Data


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

    # Inits
    # ------------------------------------------------------------------
    def __init__(self, fps, sim_size, data: Data, **_) -> None:
        self.data = data

        # Playback control
        self.running: bool = True
        self.fps = float(fps)

        # Frame / texture state
        self.current_frame  = None
        self.sim_size = sim_size
        self.last_update_time = time.time()
        self.frame_index = 0

        # Colormap
        self.cmap = cm["inferno"]


    # Run methods
    # ------------------------------------------------------------------
    def run_no_sim(self) -> None:
        raise NotImplementedError("Visualization.run_no_sim is not implemented yet.")
    

    def run(self) -> None:
        """
        Start the DearPyGui visualisation.

        This call blocks until the GUI window is closed.
        """


    # Callbacks
    # ------------------------------------------------------------------
    def start_callback(self):
        self.running=True


    def stop_callback(self):
        self.running = False


    def speed_callback(self, sender, app_data):
        new_fps = float(app_data)
        if new_fps <= 0:
            return
        self.fps = new_fps


    # Frame handler
    # ------------------------------------------------------------------
    @njit
    def normalize_frame(self,frame):
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        norm = (frame - frame_min) / (frame_max - frame_min + 1e-8)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                if norm[i, j] < 0:
                    norm[i, j] = 0.0
                elif norm[i, j] > 1:
                    norm[i, j] = 1.0
        return norm


    def update_frame(self):
        global frame_index
        current_time = time.time()
        if (self.running and (current_time - self.last_update_time) >= 1 / self.fps) or self.frame_index == 0:
            last_update_time = current_time
            frame = frames[frame_index]
            frame_norm = self.normalize_frame(frame)
            indices = (frame_norm * 255).astype(np.uint8)
            frame_rgb = inferno_lut[indices] # Apply LUT, faster than colormap
            frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0
            dpg.set_value("frame_tag", frame_flattened)
            frame_index = (frame_index + 1) % len(frames)
        with dpg.mutex():
            target_frame = dpg.get_frame_count() + 2
            dpg.set_frame_callback(target_frame, update_frame)