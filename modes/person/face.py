"""Face detection for the person mode.

YuNet (OpenCV's `cv2.FaceDetectorYN`) looks for a face in the upper part of a person crop.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

from modes.person.config import FaceValidationConfig
from modes.person import quality

log = logging.getLogger("pipeline")


@dataclass
class FaceHit:
    box: Tuple[float, float, float, float]            # x1, y1, x2, y2 in PERSON-CROP pixels
    landmarks: List[Tuple[float, float]]              # 5 points, PERSON-CROP pixels
    score: float                                      # detector confidence
    sharpness: float                                  # Laplacian variance of the face region
    quality: float                                    # combined score, see quality.quality_score
    passed_gate: bool                                 # big enough and sharp enough to keep


class YuNetFaceDetector:
    """Wrapper that turns a person crop into the best FaceHit (or None)."""

    def __init__(self, weights: str, score_threshold: float, search_region: List[float],
                 min_face_px: int, min_sharpness: float, ref_px: float = 64.0, ref_sharpness: float = 80.0,
                 max_side: int = 480, validation: Optional[FaceValidationConfig] = None):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Face model not found at: {weights}\n"
                f"Download face_detection_yunet_2023mar.onnx into models/, "
                f"or set person.face_enabled: false.")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ImportError("this OpenCV build has no FaceDetectorYN - install opencv-python >= 4.8")
        self._detector = cv2.FaceDetectorYN.create(weights, "", (320, 320), score_threshold)
        self.search_region = search_region
        self.min_face_px = min_face_px
        self.min_sharpness = min_sharpness
        self.ref_px = ref_px
        self.ref_sharpness = ref_sharpness
        self.max_side = max(1, max_side)
        self.validation = validation or FaceValidationConfig(enabled=False)
        self.score_threshold = score_threshold
        self.last_rejections = {}

    def detect(self, person_crop: np.ndarray) -> Optional[FaceHit]:
        self.last_rejections = {}
        if person_crop is None or person_crop.size == 0:
            return None
        height, width = person_crop.shape[:2]
        fx0, fy0, fx1, fy1 = self.search_region
        x0, y0 = max(0, int(width * fx0)), max(0, int(height * fy0))
        x1, y1 = min(width, int(width * fx1)), min(height, int(height * fy1))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None

        region = np.ascontiguousarray(person_crop[y0:y1, x0:x1])
        region_height, region_width = region.shape[:2]
        longest_side = max(region_width, region_height)
        scale = min(1.0, self.max_side / longest_side)
        if scale < 1.0:
            input_width = max(1, int(round(region_width * scale)))
            input_height = max(1, int(round(region_height * scale)))
            input_region = cv2.resize(region, (input_width, input_height), interpolation=cv2.INTER_AREA)
        else:
            input_region = region
        scale_x = region_width / input_region.shape[1]
        scale_y = region_height / input_region.shape[0]
        self._detector.setInputSize((input_region.shape[1], input_region.shape[0]))
        _, faces = self._detector.detect(input_region)
        if faces is None or len(faces) == 0:
            return None

        best: Optional[FaceHit] = None
        for row in faces:
            score = float(row[14])
            if score < self.score_threshold:
                self._reject("confidence")
                continue

            fx, fy, fw, fh = (float(v) for v in row[:4])
            fx, fy, fw, fh = fx * scale_x, fy * scale_y, fw * scale_x, fh * scale_y
            box = (max(0.0, fx + x0), max(0.0, fy + y0),
                   min(float(width), fx + fw + x0), min(float(height), fy + fh + y0))
            if box[2] <= box[0] or box[3] <= box[1]:
                self._reject("size")
                continue
            landmarks = [
                (float(row[4 + 2 * i]) * scale_x + x0, float(row[5 + 2 * i]) * scale_y + y0)
                for i in range(5)
            ]
            if self.validation.enabled:
                reason = self._validation_rejection(box, landmarks, width, height)
                if reason is not None:
                    self._reject(reason)
                    continue

            face_region = person_crop[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            face_sharpness = quality.sharpness(face_region)
            min_side = min(box[2] - box[0], box[3] - box[1])
            passed_gate = quality.passes_gate(min_side, face_sharpness, self.min_face_px, self.min_sharpness)
            if self.validation.enabled and not passed_gate:
                if min_side < self.min_face_px:
                    self._reject("small")
                if face_sharpness < self.min_sharpness:
                    self._reject("blurry")
                continue

            hit = FaceHit(
                box=box, landmarks=landmarks, score=score, sharpness=face_sharpness,
                quality=quality.quality_score(score, min_side, face_sharpness, self.ref_px, self.ref_sharpness),
                passed_gate=passed_gate,
            )
            if best is None or hit.score > best.score:
                best = hit
        return best

    def _validation_rejection(
        self, box: Tuple[float, float, float, float], landmarks: List[Tuple[float, float]],
        crop_width: int, crop_height: int,
    ) -> Optional[str]:
        cfg = self.validation
        x1, y1, x2, y2 = box
        face_width, face_height = x2 - x1, y2 - y1
        aspect = face_width / face_height
        if not cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio:
            return "aspect"
        if not cfg.min_height_ratio <= face_height / crop_height <= cfg.max_height_ratio:
            return "size"
        center_x = (x1 + x2) / (2.0 * crop_width)
        center_y = (y1 + y2) / (2.0 * crop_height)
        if not cfg.min_center_x_ratio <= center_x <= cfg.max_center_x_ratio or center_y > cfg.max_center_y_ratio:
            return "position"

        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
        if (right_eye[1] >= nose[1] or left_eye[1] >= nose[1]
                or nose[1] >= right_mouth[1] or nose[1] >= left_mouth[1]
                or abs(right_eye[0] - left_eye[0]) < cfg.min_eye_distance_ratio * face_width):
            return "landmarks"

        expand_x, expand_y = face_width * cfg.landmark_expand_ratio, face_height * cfg.landmark_expand_ratio
        if any(not (x1 - expand_x <= x <= x2 + expand_x and y1 - expand_y <= y <= y2 + expand_y)
               for x, y in landmarks):
            return "landmarks"
        return None

    def _reject(self, reason: str) -> None:
        self.last_rejections[reason] = self.last_rejections.get(reason, 0) + 1
