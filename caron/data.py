# imports
from collections import deque
import numpy as np


class Data:
    """Shared buffer for frames, and handler for FPS control."""

    def __init__(self, buffer_safe_min, maxlen=None, buffer_safe_max=500):
        self.buffer = deque(maxlen=maxlen)
        self.buffer_safe_min = buffer_safe_min
        self.buffer_safe_max = buffer_safe_max

    def push_frame(self, frame: np.ndarray) -> None:
        """Add a new 2D frame to the buffer."""
        self.buffer.append(frame)

    def pop_frame(self) -> np.ndarray | None:
        """Remove the oldest frame from the buffer (returns None if the buffer is empty)."""
        if not self.buffer:
            print("Data buffer is empty!")
            return None
        return self.buffer.popleft()

    def __len__(self) -> int:
        return len(self.buffer)
