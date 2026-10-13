"""Recent plate-thumbnail matching for suppressing repeated vehicle reports."""
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np


class RecentPlates:
    def __init__(self, window_seconds: float, similarity: float, min_box_iou: float = 0.4):
        self.window_seconds = window_seconds
        self.similarity = similarity
        self.min_box_iou = min_box_iou
        # (time, plate thumbnail, vehicle class, vehicle box x1y1x2y2)
        self._entries: Deque[Tuple[float, np.ndarray, str, Tuple[float, float, float, float]]] = deque(maxlen=20)

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def make_thumb(plate_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    def best_match(self, thumb: np.ndarray, now: float, vehicle_class: Optional[str] = None,
                   vehicle_box: Optional[Tuple[float, float, float, float]] = None) -> float:
        """Highest plate similarity against recent events of the same vehicle class at about the same spot.

        A plate thumbnail alone cannot tell plates apart that differ by one character, so an earlier event
        only counts when its vehicle box overlaps this one (min_box_iou) and the class matches."""
        while self._entries and now - self._entries[0][0] > self.window_seconds:
            self._entries.popleft()

        best_score = 0.0
        for _timestamp, previous, previous_class, previous_box in self._entries:
            if vehicle_class is not None and previous_class != vehicle_class:
                continue
            if vehicle_box is not None and self._iou(vehicle_box, previous_box) < self.min_box_iou:
                continue
            score = float(cv2.matchTemplate(thumb, previous, cv2.TM_CCOEFF_NORMED)[0, 0])
            if np.isfinite(score) and score > best_score:
                best_score = score
        return best_score

    def add(self, thumb: np.ndarray, now: float, vehicle_class: str = "",
            vehicle_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._entries.append((now, thumb, vehicle_class, vehicle_box))
