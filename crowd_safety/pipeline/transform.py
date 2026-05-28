"""
Converts pixel coordinates to real-world metres.
Requires 4 calibration points — pairs of (pixel, metres).

For a demo video where you don't know exact measurements,
use the ETH camera parameters which are already known.
"""

import numpy as np
import cv2

class PerspectiveTransform:
    def __init__(self):
        self.H = None   # homography matrix

    def calibrate(self, pixel_points, world_points):
        """
        pixel_points : list of 4 (px, py) points in the image
        world_points : corresponding (wx, wy) in metres
        Call this once before processing video.
        """
        src = np.float32(pixel_points)
        dst = np.float32(world_points)
        self.H, _ = cv2.findHomography(src, dst)

    def calibrate_eth(self, frame_width=640, frame_height=480):
        """
        Pre-calibrated for ETH dataset camera.
        Use this for testing with ETH scene videos.
        Approximate — good enough for demo.
        """
        pixel_points = [
            [0,           0          ],
            [frame_width, 0          ],
            [frame_width, frame_height],
            [0,           frame_height],
        ]
        # ETH scene is roughly 12m x 9m
        world_points = [
            [0,  0 ],
            [12, 0 ],
            [12, 9 ],
            [0,  9 ],
        ]
        self.calibrate(pixel_points, world_points)

    def pixel_to_world(self, cx, cy):
        """
        Convert single pixel centre to world (x, y) in metres.
        """
        if self.H is None:
            raise ValueError("Call calibrate() first")
        pt  = np.float32([[[cx, cy]]])
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0][0][0]), float(out[0][0][1])

    def batch_transform(self, centres):
        """
        centres: list of (cx, cy) pixel coords
        Returns: list of (wx, wy) world coords in metres
        """
        if not centres:
            return []
        pts = np.float32([[[c[0], c[1]]] for c in centres])
        out = cv2.perspectiveTransform(pts, self.H)
        return [(float(p[0][0]), float(p[0][1])) for p in out]