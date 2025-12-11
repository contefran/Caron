# imports

import numpy as np
from data import Data


class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""

    def __init__(self, data: Data, n_frames: int, size: int, **kwargs):
        self.data = data
        self.n_frames = n_frames
        self.size = size
        self.kwargs = kwargs

    def run(self) -> None:
        raise NotImplementedError("Simulation.run is not implemented yet.")

    def run_no_viz(self) -> None:
        raise NotImplementedError("Simulation.run_no_viz is not implemented yet.")