from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Data:
    """Shared buffer for frames, and handler for FPS control."""

    buffer_safe_min: int
    buffer_safe_max: int = 500
    maxlen = None


    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.maxlen)


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
