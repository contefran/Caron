# imports
from collections import deque # ma perche' no
import numpy as np


class Data:
    """Shared buffer for frames, and handler for FPS control."""

    def __init__(self, maxlen=None):
        self.buffer = deque(maxlen=maxlen)

    def push_frame(self, frame: np.ndarray) -> None:
        """Add a new 2D frame to the buffer."""
        self.buffer.append(frame)

    def pop_frame(self) -> np.ndarray | None:
        """Remove the oldest frame from the buffer (returns None if the buffer is empty)."""
        if not self.buffer:
            print("Data buffer is empty!") # Need to handle this case properly because it'll always be the case after the simulation is over
            return None
        return self.buffer.popleft()

    def __len__(self) -> int:
        return len(self.buffer)