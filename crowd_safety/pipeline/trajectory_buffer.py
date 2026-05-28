"""
Maintains a rolling 8-frame history of world coordinates per track ID.
When a track has 8 frames buffered, it's ready for LSTM inference.
"""

from collections import defaultdict, deque
import numpy as np

OBS_LEN = 8

class TrajectoryBuffer:
    def __init__(self, obs_len=OBS_LEN):
        self.obs_len = obs_len
        self.buffers = defaultdict(lambda: deque(maxlen=obs_len))

    def update(self, track_id, world_x, world_y):
        self.buffers[track_id].append((world_x, world_y))

    def get_ready_tracks(self):
        """
        Returns dict of track_id -> np.array(8, 2)
        Only includes tracks with full obs_len history.
        """
        ready = {}
        for tid, buf in self.buffers.items():
            if len(buf) == self.obs_len:
                coords = np.array(list(buf), dtype=np.float32)
                # Normalize to relative coords (same as training)
                origin = coords[0].copy()
                ready[tid] = coords - origin
        return ready

    def clear_lost_tracks(self, active_ids):
        """Remove buffers for tracks no longer in the scene."""
        lost = [tid for tid in self.buffers if tid not in active_ids]
        for tid in lost:
            del self.buffers[tid]