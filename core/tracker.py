"""Lightweight ByteTrack-style tracker: one stable id per physical object across frames.

Why it exists: without it every frame of a car standing in the zone became a brand-new event. With it a
vehicle / person gets ONE track, the modes try their second-stage model (plate / face) a few times on it,
keep the best result, and report it once.

How it works (the ByteTrack recipe, in plain numpy):
  1. Every track carries a constant-velocity Kalman filter, so each frame we know where it should be.
  2. Detections are split into HIGH confidence (>= the mode's normal threshold) and LOW confidence.
  3. HIGH detections are matched to the predicted track boxes by IoU (Hungarian assignment).
  4. Tracks still unmatched get a second chance against the LOW detections - a car half hidden by a pole
     or a person turning away often dips in confidence for a few frames; this keeps the id alive.
  5. HIGH detections nobody claimed start new tracks. A track must be seen `min_hits` times before it is
     "confirmed" (filters one-frame ghosts).
  6. A track that is not seen for `track_ttl_seconds` is dropped. Confirmed tracks that drop are reported
     in `TrackUpdate.expired`, so the mode can finalize whatever it has for them.

Boxes use the detector's centre format: {"x", "y", "width", "height", "class", "confidence"}.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # The greedy assignment fallback keeps SciPy optional.
    linear_sum_assignment = None


@dataclass
class TrackingConfig:
    """Shared tracking + 'how long do we keep trying' settings. Lives under `tracking:` in a mode's yaml."""
    min_hits: int = 2                   # detections needed before a track counts (1 = trust every detection)
    track_ttl_seconds: float = 2.0      # unseen this long -> the track ends and is finalized
    match_iou: float = 0.30             # min IoU to match a HIGH detection to a track
    low_conf: float = 0.15              # model runs this low so weak boxes can keep tracks alive (never start one)
    low_match_iou: float = 0.50         # min IoU to match a LOW detection to a track
    max_attempts: int = 3               # second-stage (plate / face) tries per track before giving up
    retry_interval_seconds: float = 0.3 # minimum gap between two tries on the same track
    accept_score: float = 0.60          # stop trying early once a result scores at least this
    no_plate_retry_interval_seconds: float = 0.5
    no_plate_max_attempts: int = 12
    no_face_retry_interval_seconds: float = 0.5
    no_face_max_attempts: int = 12


@dataclass
class TrackState:
    """Per-track bookkeeping the modes keep in `track.data`."""
    attempts: int = 0
    last_attempt: float = float("-inf")
    finalized: bool = False
    best: Optional[Dict[str, Any]] = None
    best_score: float = float("-inf")
    rejections: Dict[str, int] = field(default_factory=dict)


class Track:
    __slots__ = ("id", "mean", "cov", "det", "conf", "cls", "hits", "misses", "confirmed",
                 "last_seen", "created", "data")

    def __init__(self, track_id: int, det: Dict[str, Any], now: float):
        self.id = track_id
        self.mean = np.array([det["x"], det["y"], det["width"], det["height"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        h = max(float(det["height"]), 1.0)
        self.cov = np.diag([(2 * _STD_POS * h) ** 2] * 4 + [(10 * _STD_VEL * h) ** 2] * 4).astype(np.float64)
        self.det: Optional[Dict[str, Any]] = det       # detection matched THIS frame (None when not seen)
        self.conf = float(det.get("confidence", 0.0))
        self.cls = det.get("class")
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.last_seen = now
        self.created = now
        self.data = TrackState()

    @property
    def box_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# Kalman noise scale, relative to box height (the DeepSORT / ByteTrack convention).
_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.zeros((4, 8))
_H[:4, :4] = np.eye(4)


def _predict(track: Track) -> None:
    h = max(float(track.mean[3]), 1.0)
    q = np.diag([(_STD_POS * h) ** 2] * 4 + [(_STD_VEL * h) ** 2] * 4)
    track.mean = _F @ track.mean
    track.cov = _F @ track.cov @ _F.T + q
    track.mean[2:4] = np.maximum(track.mean[2:4], 1.0)       # a box never collapses to nothing


def _correct(track: Track, det: Dict[str, Any]) -> None:
    z = np.array([det["x"], det["y"], det["width"], det["height"]], dtype=np.float64)
    h = max(float(track.mean[3]), 1.0)
    r = np.diag([(_STD_POS * h) ** 2] * 4)
    s = _H @ track.cov @ _H.T + r
    gain = np.linalg.solve(s, _H @ track.cov).T               # = P H^T S^-1 (S is symmetric)
    track.mean = track.mean + gain @ (z - _H @ track.mean)
    track.cov = (np.eye(8) - gain @ _H) @ track.cov
    track.mean[2:4] = np.maximum(track.mean[2:4], 1.0)


def _det_xyxy(dets: List[Dict[str, Any]]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 4))
    arr = np.array([[d["x"], d["y"], d["width"], d["height"]] for d in dets], dtype=np.float64)
    return np.stack([arr[:, 0] - arr[:, 2] / 2, arr[:, 1] - arr[:, 3] / 2,
                     arr[:, 0] + arr[:, 2] / 2, arr[:, 1] + arr[:, 3] / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in `a` (N,4 xyxy) and every box in `b` (M,4 xyxy) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _assign(iou: np.ndarray, min_iou: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Best one-to-one matching with IoU >= min_iou. Returns (pairs, unmatched rows, unmatched cols)."""
    rows, cols = iou.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(1.0 - iou)
        candidates = list(zip(row_idx.tolist(), col_idx.tolist()))
    else:                                                      # greedy fallback
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        used_r, used_c, candidates = set(), set(), []
        for r, c in order.tolist():
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            candidates.append((r, c))

    pairs = [(r, c) for r, c in candidates if iou[r, c] >= min_iou]
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (pairs,
            [r for r in range(rows) if r not in matched_r],
            [c for c in range(cols) if c not in matched_c])


@dataclass
class TrackUpdate:
    active: List[Track] = field(default_factory=list)    # confirmed tracks seen in THIS frame (track.det is set)
    expired: List[Track] = field(default_factory=list)   # confirmed tracks that just ended


class ByteTracker:
    def __init__(self, config: TrackingConfig, high_conf: float):
        self.cfg = config
        self.high_conf = high_conf
        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> Tuple[Track, ...]:
        """All current tracks, including tentative and briefly unmatched tracks."""
        return tuple(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def flush(self) -> List[Track]:
        """End every confirmed track now (shutdown / mode switch / single image). Returns them."""
        ended = [t for t in self._tracks if t.confirmed]
        self._tracks.clear()
        return ended

    def update(self, detections: List[Dict[str, Any]], now: float, confirm_all: bool = False) -> TrackUpdate:
        cfg = self.cfg
        min_hits = 1 if confirm_all else max(1, cfg.min_hits)

        for track in self._tracks:
            _predict(track)
            track.det = None

        high = [d for d in detections if d.get("confidence", 0.0) >= self.high_conf]
        low = [d for d in detections if d.get("confidence", 0.0) < self.high_conf]

        tracks = self._tracks
        track_boxes = np.array([t.box_xyxy for t in tracks]) if tracks else np.zeros((0, 4))

        # 1st association: all tracks vs HIGH detections
        pairs, free_tracks, free_high = _assign(iou_matrix(track_boxes, _det_xyxy(high)), cfg.match_iou)
        matched = [(tracks[r], high[c]) for r, c in pairs]

        # 2nd association: tracks that were seen last frame but not claimed vs LOW detections
        retry = [r for r in free_tracks if tracks[r].misses == 0]
        if retry and low:
            sub = iou_matrix(track_boxes[retry], _det_xyxy(low))
            pairs2, _, _ = _assign(sub, cfg.low_match_iou)
            claimed = set()
            for r_sub, c in pairs2:
                matched.append((tracks[retry[r_sub]], low[c]))
                claimed.add(retry[r_sub])
            free_tracks = [r for r in free_tracks if r not in claimed]

        for track, det in matched:
            _correct(track, det)
            track.det = det
            track.hits += 1
            track.misses = 0
            track.last_seen = now
            track.conf = float(det.get("confidence", 0.0))
            track.cls = det.get("class", track.cls)
            if not track.confirmed and track.hits >= min_hits:
                track.confirmed = True

        survivors = [t for t in tracks if t.det is not None]
        expired: List[Track] = []
        for r in free_tracks:
            track = tracks[r]
            track.misses += 1
            if not track.confirmed:
                continue                                       # a ghost that vanished: drop silently
            if now - track.last_seen > cfg.track_ttl_seconds:
                expired.append(track)
            else:
                survivors.append(track)                        # briefly lost, keep predicting

        for c in free_high:                                    # new objects (HIGH detections only)
            track = Track(self._next_id, high[c], now)
            self._next_id += 1
            if track.hits >= min_hits:
                track.confirmed = True
            survivors.append(track)

        survivors.sort(key=lambda t: t.id)
        self._tracks = survivors
        return TrackUpdate(active=[t for t in survivors if t.confirmed and t.det is not None], expired=expired)
