"""Lean person detection flow optimized for Raspberry Pi:

    person detect → track → crop → face detect (best of a few tries) → crop → send metadata once per person
"""
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from core import detector
from core.geometry import crop_box
from core.motion_gate import MotionGate
from core.tracker import ByteTracker, Track
from core.timing import timer
from modes.base import DetectionMode, RuntimeContext
from modes.person.face import YuNetFaceDetector
from modes.person.geometry import is_person_inside_roi
from storage.buffer import PersonRecord, encode_jpeg

log = logging.getLogger("pipeline")


class PersonMode(DetectionMode):
    """Lean person -> face detection pipeline."""

    name = "person"

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = ctx.config
        self.cfg = ctx.config.person
        self.buffer = ctx.buffer
        self.roi = ctx.roi
        self.camera_source = ctx.camera_source

        self.person_model = None
        self.face_detector: Optional[YuNetFaceDetector] = None
        self.tracker = ByteTracker(self.cfg.tracking, high_conf=self.cfg.conf_threshold)
        self.motion_gate = MotionGate(self.cfg.frame_skip)
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stopped = False

        try:
            self._load()
        except Exception:
            self.stop()
            raise

    def _load(self) -> None:
        cfg, config = self.cfg, self.config
        runtime = config.runtime
        capped = detector.limit_onnx_threads(runtime.inference_threads)
        if capped:
            print(f"[person] ONNX Runtime limited to {capped} CPU thread(s)")

        print("[person] loading person model...")
        self.person_model = detector.load_model(cfg.weights, runtime.device)
        self._sync_imgsz()

        if cfg.face_enabled:
            print("[person] loading face model...")
            self.face_detector = YuNetFaceDetector(
                cfg.face_weights, cfg.face_score_threshold, cfg.face_search_region,
                cfg.face_min_px, cfg.face_min_sharpness, cfg.face_ref_px, cfg.face_ref_sharpness,
                cfg.face_detect_max_side, cfg.face_validation,
            )
        else:
            log.warning("[person] face_enabled is OFF - people will be stored without faces")

        classes = cfg.classes
        self.classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def _sync_imgsz(self) -> None:
        actual = self.person_model.input_hw
        if actual is None:
            return
        configured = detector.imgsz_hw(self.cfg.imgsz)
        if actual != configured:
            log.warning("[person] %s static shape %dx%d != configured %dx%d; using %dx%d",
                        self.cfg.weights, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
            self.cfg.imgsz = [actual[0], actual[1]]

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[person] warming up models...")
        region = self._detect_region(frame_shape) if frame_shape else None
        shape = (region[3] - region[1], region[2] - region[0]) if region else frame_shape
        detector.warmup_model(self.person_model, self.cfg.imgsz, shape)
        if self.face_detector is not None:
            dummy = np.zeros((300, 120, 3), dtype=np.uint8)
            try:
                self.face_detector.detect(dummy)
                log.info("[person] face model self-test OK")
            except Exception as error:
                log.error("[person] face model self-test failed: %s", error)

    def _detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        cfg = self.cfg
        if not (cfg.roi_crop_detect and cfg.roi_filter):
            return None
        return self.roi.detect_region(
            frame_shape, detector.imgsz_hw(cfg.imgsz), cfg.roi_crop_margin,
            top_extra=cfg.roi_top_extra, label="person model",
        )

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        """Returns how many person events were stored by this call (not how many boxes were seen)."""
        cfg = self.cfg
        tcfg = cfg.tracking
        frame_shape = frame.shape[:2]
        now = time.monotonic()

        # 1. Person detect. The model runs at the lower tracking.low_conf so weak boxes can keep an existing
        #    track alive; the tracker only lets boxes >= conf_threshold START a track.
        region = self._detect_region(frame.shape)
        if not self.motion_gate.should_run(
            frame, self.tracker.tracks, now, region, always_run=always_finalize,
        ):
            update = self.tracker.update([], now)
            return sum(self._finalize(track) for track in update.expired)

        with timer("person_detect").measure():
            predictions = detector.detect(
                self.person_model, frame, min(cfg.conf_threshold, tcfg.low_conf), cfg.iou_threshold, cfg.imgsz,
                self.classes, region=region, agnostic_nms=cfg.agnostic_nms,
            )

        # Filter to ROI (feet check)
        if cfg.roi_filter:
            polygon = self.roi.get_polygon()
            sizes = (cfg.min_width_ratio, cfg.min_height_ratio, cfg.max_width_ratio, cfg.max_height_ratio)
            predictions = [
                p for p in predictions
                if is_person_inside_roi(p, frame_shape, polygon, *sizes, edge_margin_ratio=cfg.edge_margin_ratio)
            ]

        # 2. Track: one id per physical person across frames
        update = self.tracker.update(predictions, now, confirm_all=always_finalize)

        stored = 0
        for track in update.active:
            state = track.data
            if state.finalized:
                continue                                    # already reported; just keep following them
            has_face = bool(state.best and state.best["face_detected"])
            retry_interval = (tcfg.retry_interval_seconds if has_face or self.face_detector is None
                              else tcfg.no_face_retry_interval_seconds)
            if state.attempts and now - state.last_attempt < retry_interval:
                continue
            self._attempt(track, frame, now)
            has_face = bool(state.best and state.best["face_detected"])
            done = (
                always_finalize or self.face_detector is None
                or (has_face and state.best_score >= tcfg.accept_score)
                or (has_face and state.attempts >= tcfg.max_attempts)
                or (not has_face and state.attempts >= tcfg.no_face_max_attempts)
            )
            if done:
                stored += self._finalize(track)

        for track in update.expired:                        # person left: store the best we got
            stored += self._finalize(track)
        if always_finalize:                                 # single image / folder: nothing carries over
            for track in self.tracker.flush():
                stored += self._finalize(track)
        return stored

    def _attempt(self, track: Track, frame: np.ndarray, now: float) -> None:
        """One crop + face-detect try on this person. Keeps the result only if it beats the best so far."""
        cfg = self.cfg
        state = track.data
        state.attempts += 1
        state.last_attempt = now
        prediction = track.det

        # 3. Crop person
        original_person_crop = crop_box(frame, prediction, cfg.padding_ratio, 0)
        person_crop = crop_box(frame, prediction, cfg.padding_ratio, cfg.min_crop_height)
        if original_person_crop is None or person_crop is None:
            return

        # 4. Face detect + crop
        face_crop: Optional[np.ndarray] = None
        face_landmarks: Optional[str] = None
        face_quality: Optional[float] = None
        if self.face_detector is not None:
            face_source_crop = original_person_crop if cfg.face_validation.enabled else person_crop
            with timer("face_detect").measure():
                hit = self.face_detector.detect(face_source_crop)
            for reason, count in self.face_detector.last_rejections.items():
                state.rejections[reason] = state.rejections.get(reason, 0) + count
            if hit is not None and hit.passed_gate:
                fx1, fy1, fx2, fy2 = hit.box
                margin_x, margin_y = (fx2 - fx1) * cfg.face_margin, (fy2 - fy1) * cfg.face_margin
                crop_h, crop_w = face_source_crop.shape[:2]
                x1, y1 = max(0, int(fx1 - margin_x)), max(0, int(fy1 - margin_y))
                x2, y2 = min(crop_w, int(fx2 + margin_x)), min(crop_h, int(fy2 + margin_y))
                candidate = face_source_crop[y1:y2, x1:x2]
                if candidate.size > 0:
                    face_crop = candidate
                    face_landmarks = json.dumps([[round(x, 1), round(y, 1)] for x, y in hit.landmarks])
                    face_quality = round(hit.quality, 4)
                else:
                    state.rejections["size"] = state.rejections.get("size", 0) + 1
            elif hit is not None and not hit.passed_gate:
                if min(hit.box[2] - hit.box[0], hit.box[3] - hit.box[1]) < cfg.face_min_px:
                    state.rejections["small"] = state.rejections.get("small", 0) + 1
                if hit.sharpness < cfg.face_min_sharpness:
                    state.rejections["blurry"] = state.rejections.get("blurry", 0) + 1

        # A usable face always beats no face; confidence ranks person-only candidates when allowed.
        person_confidence = float(prediction.get("confidence", 0.0))
        score = face_quality if face_crop is not None else person_confidence - 1.0
        if score <= state.best_score:
            return

        # Only the winning try is JPEG-encoded (keeps the CPU cost of retries low).
        person_jpeg = encode_jpeg(person_crop, cfg.jpeg_quality)
        face_jpeg = encode_jpeg(face_crop, cfg.face_jpeg_quality) if face_crop is not None else None
        state.best_score = score
        state.best = {
            "person_confidence": person_confidence,
            "person_jpeg": person_jpeg,
            "person_size": (person_crop.shape[1], person_crop.shape[0]),
            "face_detected": face_crop is not None,
            "face_jpeg": face_jpeg,
            "face_size": (face_crop.shape[1], face_crop.shape[0]) if face_crop is not None else None,
            "face_landmarks": face_landmarks,
            "face_quality": face_quality,
            "frame_size": (frame.shape[1], frame.shape[0]),
            "at": datetime.now(),
        }

    def _finalize(self, track: Track) -> int:
        """5. Store the best result for this person (once) - enqueue to the in-memory send buffer."""
        state = track.data
        if state.finalized:
            return 0
        state.finalized = True
        best, state.best = state.best, None                # drop the JPEG bytes either way
        if best is None:
            return 0
        if self.face_detector is not None and not best["face_detected"] and not self.cfg.store_without_face:
            return 0
        track_id = f"p{track.id}"
        stem = f"{self._session_tag}_{track_id}"
        disk_files: Optional[dict] = {} if self.cfg.save_images_to_disk else None
        if disk_files is not None:
            disk_files[f"person_detection/{stem}_person.jpg"] = best["person_jpeg"]
            if best["face_jpeg"] is not None:
                disk_files[f"face_detection/{stem}_face.jpg"] = best["face_jpeg"]

        self.buffer.enqueue(PersonRecord(
            event_uuid=str(uuid.uuid4()),
            device_id=self.config.runtime.device_id or "unknown",
            session_id=self.ctx.session_id,
            track_id=stem,     # unique across restarts / mode switches: <start time>_p<n>
            person_confidence=best["person_confidence"],
            person_image_jpeg=best["person_jpeg"],
            face_detected=best["face_detected"],
            detected_at=best["at"],
            face_image_jpeg=best["face_jpeg"],
            face_landmarks=best["face_landmarks"],
            face_quality_score=best["face_quality"],
            disk_files=disk_files,
        ))

        status = f"face found (q={best['face_quality']:.2f})" if best["face_detected"] else "no face"
        if not best["face_detected"]:
            rejected = self._format_rejections(state.rejections)
            if rejected:
                status += f" (rejected: {rejected})"
        frame_width, frame_height = best["frame_size"]
        person_width, person_height = best["person_size"]
        details = (f"frame {frame_width}x{frame_height}, person {person_width}x{person_height} "
                   f"({len(best['person_jpeg']) / 1024:.0f} KB)")
        if best["face_jpeg"] is not None:
            face_width, face_height = best["face_size"]
            details += f", face {face_width}x{face_height} ({len(best['face_jpeg']) / 1024:.0f} KB)"
        print(f"[person] {track_id} ({best['person_confidence']:.2f}) - {status} "
              f"[{details}] [{state.attempts} tries]")
        return 1

    @staticmethod
    def _format_rejections(rejections: dict) -> str:
        order = ("confidence", "aspect", "position", "size", "landmarks", "small", "blurry")
        return ", ".join(f"{reason} {rejections[reason]}" for reason in order if rejections.get(reason))

    def _flush_pending(self) -> None:
        """Hand every person still being followed to storage (mode switch / shutdown): nothing is lost."""
        try:
            for track in self.tracker.flush():
                self._finalize(track)
        except Exception as error:
            log.warning("[person] could not flush in-flight tracks: %s", error)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._flush_pending()
        models = [self.person_model]
        self.person_model = None
        self.face_detector = None
        detector.release_models(*models)
        log.info("[person] stopped, models released")
