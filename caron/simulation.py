"""Simulation class producing data for the visualization."""
from caron.data import Data
import numpy as np
import time

class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""


    def __init__(self, data: Data, args, **kwargs) -> None:
        self.args = args
        self.data = data
        self.kwargs = kwargs
        self.sim_size: int = self.args.sim_size


    def run(self) -> None:
        raise NotImplementedError("[Sim] Simulation.run is not implemented yet.")


    def run_mock(self) -> None:
        """Feed frames into the buffer at a fixed rate (sim_fps)."""
        self.sim_fps=40 #Hz mock simulation fps
        self.sim_file = self.args.sim_file
        self.sim=np.load(self.sim_file)
        self.sim_size: int = self.sim.shape[1]
        print(f"[Sim] Loaded mock simulation from {self.sim_file} with {self.sim[0].shape[1]} linear size with {self.sim.shape[2]} frames.")
        print(f"[Sim] Simulation initialized in fake_injection mode. Filling the buffer at {self.sim_fps} FPS.")
        dt = 1.0 / self.sim_fps # seconds per frame of the mock simulation injection
        t_next = time.perf_counter()

        for i, frame in enumerate(self.sim):
            now = time.perf_counter() # Wait until next frame time
            if now < t_next:
                time.sleep(t_next - now) # is this appropriate? Maybe we should take a statistical approach as in the visualizer? To be checked

            self.data.push_frame(frame)
            t_next += dt # this assumes that pushing the frame is instantaneous. Needs to be checked
            print(f"[Sim] added frame {i+1}/{len(self.sim)} in {time.perf_counter()-now:.4f}s | buffer size={len(self.data)}")

        print("[Sim] Simulation finished pushing frames.")


    def run_no_viz(self) -> None:
        raise NotImplementedError("[Sim] Simulation.run_no_viz is not implemented yet.")