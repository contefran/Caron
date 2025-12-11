# imports
import numpy as np
from data import Data


class Simulation:
    """
    Produces 2D frames and pushes them into the Data buffer.

    Phase 1: use precomputed arrays / mock simulation.
    Phase 2: stream frames in real time.
    """

    def __init__(self, data: Data, n_frames: int, size: int, **kwargs: Any):
        self.data = data
        self.n_frames = n_frames
        self.size = size
        self.kwargs = kwargs

    def run(self) -> None:
        """
        Run the simulation.

        Phase 1 placeholder: fill buffer with dummy frames.
        Replace with real logic or loading from Simulation.npy.
        """
        raise NotImplementedError("Simulation.run is not implemented yet.")
