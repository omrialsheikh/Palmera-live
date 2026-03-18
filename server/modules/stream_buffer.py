"""
Sliding Window Buffer for temporal consistency.
Maintains a rolling buffer of pose frames for MimicMotion inference.
"""

import numpy as np
from collections import deque
from PIL import Image


class StreamBuffer:
    def __init__(self, window_size: int = 8, overlap: int = 2):
        self.window_size = window_size
        self.overlap = overlap
        self.pose_buffer: deque[dict] = deque(maxlen=window_size)
        self.frame_count = 0

    def add_frame(self, pose_data: dict):
        """Add a new pose frame to the buffer."""
        self.pose_buffer.append(pose_data)
        self.frame_count += 1

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.pose_buffer) >= self.window_size

    def get_window(self) -> list[dict]:
        """Get the current window of frames for inference."""
        return list(self.pose_buffer)

    def get_pose_images(self) -> list[Image.Image]:
        """Get pose images from the current window."""
        return [frame["pose_image"] for frame in self.pose_buffer]

    def get_overlap_frames(self) -> list[dict]:
        """Get the overlap frames from the end of the current window."""
        if len(self.pose_buffer) < self.overlap:
            return list(self.pose_buffer)
        return list(self.pose_buffer)[-self.overlap:]

    def should_run_inference(self) -> bool:
        """
        Determine if we should run inference.
        Runs when buffer is full and we've accumulated enough new frames.
        """
        new_frames = self.window_size - self.overlap
        return self.is_ready() and (self.frame_count % new_frames == 0)

    def reset(self):
        self.pose_buffer.clear()
        self.frame_count = 0
