"""Motion-based frame skipping shared by the detection modes."""
from dataclasses import dataclass
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from core.tracker import Track


@dataclass
class FrameSkipConfig:
    enabled: bool = True
    motion_threshold: float = 0.01
    follow_every_n: int = 3
    idle_heartbeat_seconds: float = 2.0


class MotionGate:
    def __init__(self, config: FrameSkipConfig):
        self.config = config
        self._previous: Optional[np.ndarray] = None
        self._last_run: Optional[float] = None
        self._follow_frames = 0

    def should_run(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        now: float,
        region: Optional[Tuple[int, int, int, int]] = None,
        always_run: bool = False,
    ) -> bool:
        motion = self._has_motion(frame, region)
        active = [track for track in tracks if getattr(track, "confirmed", False)]

        if always_run or not self.config.enabled:
            run = True
        elif any(not track.data.finalized for track in tracks):
            run = True
        elif active:
            self._follow_frames += 1
            run = self._follow_frames >= max(1, self.config.follow_every_n)
        else:
            self._follow_frames = 0
            heartbeat = (self._last_run is None or
                         now - self._last_run >= self.config.idle_heartbeat_seconds)
            run = motion or heartbeat

        if run:
            self._last_run = now
            self._follow_frames = 0
        return run

    def _has_motion(self, frame: np.ndarray, region: Optional[Tuple[int, int, int, int]]) -> bool:
        image = frame
        if region is not None:
            x1, y1, x2, y2 = region
            if x2 > x1 and y2 > y1:
                image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return False

        height, width = image.shape[:2]
        target_width = min(160, width)
        target_height = max(1, round(height * target_width / width))
        thumbnail = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return False

        difference = cv2.absdiff(previous, gray)
        changed = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        return float(np.count_nonzero(changed)) / changed.size > self.config.motion_threshold
