"""Lean vehicle detection flow optimized for Raspberry Pi:

    vehicle detect → track → crop → plate detect (best of a few tries) → crop → send metadata once per vehicle
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import detector
from core.geometry import box_edges, crop_box
from core.tracker import ByteTracker, Track
from modes.base import DetectionMode, RuntimeContext
from modes.vehicle.geometry import crop_plate, is_inside_roi
from storage.buffer import VehicleRecord, encode_jpeg

log = logging.getLogger("pipeline")


class VehicleMode(DetectionMode):
    """Lean vehicle -> plate detection pipeline."""

    name = "vehicle"

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = ctx.config.vehicle
        self.buffer = ctx.buffer
        self.roi = ctx.roi
        self.camera_source = ctx.camera_source

        self.vehicle_model = None
        self.plate_model = None
        self.tracker = ByteTracker(self.config.tracking, high_conf=self.config.model.vehicle_conf_threshold)
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stopped = False

        try:
            self._load()
        except Exception:
            self.stop()
            raise

    def _load(self) -> None:
        runtime = self.ctx.config.runtime
        config = self.config
        capped_threads = detector.limit_onnx_threads(runtime.inference_threads)
        if capped_threads:
            print(f"[vehicle] ONNX Runtime limited to {capped_threads} CPU thread(s)")

        print("[vehicle] loading vehicle model...")
        self.vehicle_model = detector.load_model(config.model.vehicle_weights, runtime.device)
        print("[vehicle] loading plate model...")
        self.plate_model = detector.load_model(config.model.plate_weights, runtime.device)
        self._sync_imgsz_with_model("vehicle", config.model.vehicle_weights, "vehicle_imgsz")
        self._sync_imgsz_with_model("plate", config.model.plate_weights, "plate_imgsz")

        classes = config.model.vehicle_classes
        self.vehicle_classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[vehicle] warming up models...")
        region = self._get_detect_region(frame_shape) if frame_shape else None
        if region is not None:
            frame_shape = (region[3] - region[1], region[2] - region[0])
        detector.warmup_model(self.vehicle_model, self.config.model.vehicle_imgsz, frame_shape)
        detector.warmup_model(self.plate_model, self.config.model.plate_imgsz)

    def _sync_imgsz_with_model(self, label: str, weights: str, attribute: str) -> None:
        actual = detector.onnx_static_input_hw(weights)
        if actual is None:
            return
        configured = detector.imgsz_hw(getattr(self.config.model, attribute))
        if actual != configured:
            log.warning("[vehicle] %s model %s static shape %dx%d != configured %dx%d; using %dx%d",
                        label, weights, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
            setattr(self.config.model, attribute, [actual[0], actual[1]])

    def _get_detect_region(self, frame_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        features = self.config.features
        if not (features.roi_crop_detect and features.roi_filter):
            return None
        return self.roi.detect_region(
            frame_shape, detector.imgsz_hw(self.config.model.vehicle_imgsz),
            self.config.roi.crop_margin, label="vehicle model",
        )

    def process_frame(self, frame: np.ndarray, always_finalize: bool = False,
                      captured_at: Optional[float] = None) -> int:
        """Returns how many vehicle events were stored by this call (not how many boxes were seen)."""
        frame_shape = frame.shape[:2]
        cfg = self.config
        features = cfg.features
        tcfg = cfg.tracking
        now = time.monotonic()

        # 1. Vehicle detect. The model runs at the lower tracking.low_conf so weak boxes can keep an existing
        #    track alive; the tracker only lets boxes >= vehicle_conf_threshold START a track.
        region = self._get_detect_region(frame_shape)
        predictions = detector.detect(
            self.vehicle_model, frame,
            min(cfg.model.vehicle_conf_threshold, tcfg.low_conf), cfg.model.vehicle_iou_threshold,
            cfg.model.vehicle_imgsz, self.vehicle_classes, region=region,
            agnostic_nms=cfg.model.agnostic_nms,
        )
        predictions = [p for p in predictions if not self._looks_like_plate(p)]

        # Filter to ROI
        if features.roi_filter:
            polygon = self.roi.get_polygon()
            roi_cfg = cfg.roi
            predictions = [
                p for p in predictions
                if is_inside_roi(
                    p, frame_shape, polygon,
                    roi_cfg.min_width_ratio, roi_cfg.min_height_ratio,
                    roi_cfg.max_width_ratio, roi_cfg.max_height_ratio,
                    roi_cfg.edge_margin_ratio,
                )
            ]

        # 2. Track: one id per physical vehicle across frames
        update = self.tracker.update(predictions, now, confirm_all=always_finalize)

        stored = 0
        for track in update.active:
            state = track.data
            if state.finalized:
                continue                                    # already reported; just keep following it
            if state.attempts and now - state.last_attempt < tcfg.retry_interval_seconds:
                continue
            self._attempt(track, frame, now)
            done = (always_finalize or not features.plate_detection
                    or state.best_score >= tcfg.accept_score or state.attempts >= tcfg.max_attempts)
            if done:
                stored += self._finalize(track)

        for track in update.expired:                        # vehicle left: store the best we got
            stored += self._finalize(track)
        if always_finalize:                                 # single image / folder: nothing carries over
            for track in self.tracker.flush():
                stored += self._finalize(track)
        return stored

    def _attempt(self, track: Track, frame: np.ndarray, now: float) -> None:
        """One crop + plate-detect try on this vehicle. Keeps the result only if it beats the best so far."""
        cfg = self.config
        crop_cfg = cfg.crop
        state = track.data
        state.attempts += 1
        state.last_attempt = now
        prediction = track.det

        # 3. Crop vehicle
        vehicle_crop = crop_box(frame, prediction, crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height)
        if vehicle_crop is None:
            return

        # 4. Plate detect + crop
        plate_confidence: Optional[float] = None
        plate_crop: Optional[np.ndarray] = None
        plate_box = (None, None, None, None)
        if cfg.features.plate_detection:
            plate_predictions = detector.detect(
                self.plate_model, vehicle_crop,
                cfg.model.plate_conf_threshold, cfg.model.vehicle_iou_threshold,
                cfg.model.plate_imgsz,
            )
            if plate_predictions:
                best_plate = max(plate_predictions, key=lambda p: p.get("confidence", 0.0))
                crop = crop_plate(
                    vehicle_crop, best_plate,
                    crop_cfg.plate_padding_pixels, crop_cfg.plate_min_crop_height, crop_cfg.plate_padding_ratio,
                )
                if crop is not None:
                    plate_crop = crop
                    plate_confidence = float(best_plate.get("confidence", 0.0))
                    plate_box = box_edges(best_plate)

        # A plate always beats no plate; among plates the more confident wins; among no-plates the surer vehicle.
        vehicle_confidence = float(prediction.get("confidence", 0.0))
        score = plate_confidence if plate_crop is not None else vehicle_confidence - 1.0
        if score <= state.best_score:
            return

        # Only the winning try is JPEG-encoded (keeps the CPU cost of retries low).
        vx1, vy1, vx2, vy2 = box_edges(prediction)
        state.best_score = score
        state.best = {
            "class": str(prediction.get("class") or "vehicle"),
            "vehicle_confidence": vehicle_confidence,
            "vehicle_jpeg": encode_jpeg(vehicle_crop, cfg.images.jpeg_quality),
            "vehicle_box": (vx1, vy1, vx2, vy2),
            "plate_detected": plate_crop is not None,
            "plate_confidence": plate_confidence,
            "plate_jpeg": encode_jpeg(plate_crop, cfg.images.plate_jpeg_quality) if plate_crop is not None else None,
            "plate_box": plate_box,
            "at": datetime.now(),
        }

    def _finalize(self, track: Track) -> int:
        """5. Store the best result for this vehicle (once) - enqueue to the shared SQLite buffer."""
        state = track.data
        if state.finalized:
            return 0
        state.finalized = True
        best, state.best = state.best, None                # drop the JPEG bytes either way
        if best is None:
            return 0

        features = self.config.features
        track_id = f"v{track.id}"
        stem = f"{self._session_tag}_{track_id}"
        disk_files: Optional[Dict[str, bytes]] = {} if features.save_images_to_disk else None
        if disk_files is not None:
            disk_files[f"vehicle_detection/{stem}_{self._safe_name(best['class'])}.jpg"] = best["vehicle_jpeg"]
            if best["plate_jpeg"] is not None:
                disk_files[f"plate_detection/{stem}_plate.jpg"] = best["plate_jpeg"]

        vx1, vy1, vx2, vy2 = best["vehicle_box"]
        px1, py1, px2, py2 = best["plate_box"]
        self.buffer.enqueue(VehicleRecord(
            track_id=track_id,
            camera_source=self.camera_source,
            vehicle_class=best["class"],
            vehicle_confidence=best["vehicle_confidence"],
            vehicle_image_jpeg=best["vehicle_jpeg"],
            vehicle_box_x1=vx1, vehicle_box_y1=vy1, vehicle_box_x2=vx2, vehicle_box_y2=vy2,
            plate_detected=best["plate_detected"],
            plate_confidence=best["plate_confidence"],
            plate_image_jpeg=best["plate_jpeg"],
            plate_box_x1=px1, plate_box_y1=py1, plate_box_x2=px2, plate_box_y2=py2,
            detected_at=best["at"],
            disk_files=disk_files,
        ))

        if features.print_console:
            status = f"plate found ({best['plate_confidence']:.2f})" if best["plate_detected"] else "no plate"
            print(f"[vehicle] {track_id} {best['class']} ({best['vehicle_confidence']:.2f}) - {status} "
                  f"[{state.attempts} try/tries]")
        return 1

    def _flush_pending(self) -> None:
        """Hand every vehicle still being followed to storage (mode switch / shutdown): nothing is lost."""
        try:
            for track in self.tracker.flush():
                self._finalize(track)
        except Exception as error:
            log.warning("[vehicle] could not flush in-flight tracks: %s", error)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._flush_pending()
        models = [self.vehicle_model, self.plate_model]
        self.vehicle_model = self.plate_model = None
        detector.release_models(*models)
        log.info("[vehicle] stopped, models released")

    @staticmethod
    def _looks_like_plate(prediction: Dict[str, Any]) -> bool:
        label = "".join(ch for ch in str(prediction.get("class") or "").lower() if ch.isalnum())
        return any(marker in label for marker in ("plate", "license", "licence"))

    @staticmethod
    def _safe_name(value: Optional[str]) -> str:
        value = value or "vehicle"
        return "".join(ch if ch.isalnum() else "_" for ch in value)
