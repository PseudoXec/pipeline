"""Lean vehicle detection flow optimized for Raspberry Pi:

    vehicle detect → track → crop → plate detect (best of a few tries) → crop → send metadata once per vehicle
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core import detector
from core.dedupe import RecentPlates
from core.geometry import box_edges, crop_box
from core.motion_gate import MotionGate
from core.tracker import ByteTracker, Track
from core.timing import timer
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
        self.motion_gate = MotionGate(self.config.frame_skip)
        self.recent_plates = RecentPlates(
            self.config.features.dedupe_window_seconds,
            self.config.features.dedupe_similarity,
            self.config.features.dedupe_min_box_iou,
        )
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
        self._sync_imgsz_with_model("vehicle", self.vehicle_model, "vehicle_imgsz")
        self._sync_imgsz_with_model("plate", self.plate_model, "plate_imgsz")

        classes = config.model.vehicle_classes
        self.vehicle_classes = {c.strip() for c in classes.split(",") if c.strip()} if classes else None

    def warmup(self, frame_shape: Optional[tuple] = None) -> None:
        print("[vehicle] warming up models...")
        region = self._get_detect_region(frame_shape) if frame_shape else None
        if region is not None:
            frame_shape = (region[3] - region[1], region[2] - region[0])
        detector.warmup_model(self.vehicle_model, self.config.model.vehicle_imgsz, frame_shape)
        detector.warmup_model(self.plate_model, self.config.model.plate_imgsz)

    def _sync_imgsz_with_model(self, label: str, model, attribute: str) -> None:
        actual = model.input_hw
        if actual is None:
            return
        configured = detector.imgsz_hw(getattr(self.config.model, attribute))
        if actual != configured:
            log.warning("[vehicle] %s model static shape %dx%d != configured %dx%d; using %dx%d",
                        label, actual[0], actual[1], configured[0], configured[1], actual[0], actual[1])
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
        if not self.motion_gate.should_run(
            frame, self.tracker.tracks, now, region, always_run=always_finalize,
        ):
            update = self.tracker.update([], now)
            return sum(self._finalize(track) for track in update.expired)

        with timer("vehicle_detect").measure():
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
            has_plate = bool(state.best and state.best["plate_detected"])
            retry_interval = (tcfg.retry_interval_seconds if has_plate or not features.plate_detection
                              else tcfg.no_plate_retry_interval_seconds)
            if state.attempts and now - state.last_attempt < retry_interval:
                continue
            self._attempt(track, frame, now)
            has_plate = bool(state.best and state.best["plate_detected"])
            done = (
                always_finalize or not features.plate_detection
                or (has_plate and state.best_score >= tcfg.accept_score)
                or (has_plate and state.attempts >= tcfg.max_attempts)
                or (not has_plate and state.attempts >= tcfg.no_plate_max_attempts)
            )
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
        original_vehicle_crop = crop_box(frame, prediction, crop_cfg.vehicle_padding_ratio, 0)
        vehicle_crop = crop_box(frame, prediction, crop_cfg.vehicle_padding_ratio, crop_cfg.vehicle_min_crop_height)
        if vehicle_crop is None:
            return

        # 4. Plate detect + crop
        plate_confidence: Optional[float] = None
        plate_crop: Optional[np.ndarray] = None
        plate_box = (None, None, None, None)
        if cfg.features.plate_detection:
            with timer("plate_detect").measure():
                plate_predictions = detector.detect(
                    self.plate_model, vehicle_crop,
                    # Keep weak candidates so rejections can be counted, but not ~every anchor (conf 0.0 made NMS
                    # run on thousands of boxes: about +200 ms per try on a desktop CPU).
                    min(0.25, cfg.model.plate_conf_threshold) if cfg.plate_validation.enabled
                    else cfg.model.plate_conf_threshold,
                    cfg.model.vehicle_iou_threshold,
                    cfg.model.plate_imgsz,
                )
            if cfg.plate_validation.enabled:
                valid_plates = []
                if original_vehicle_crop is not None:
                    scale_x = original_vehicle_crop.shape[1] / vehicle_crop.shape[1]
                    scale_y = original_vehicle_crop.shape[0] / vehicle_crop.shape[0]
                    for candidate in plate_predictions:
                        mapped = dict(candidate)
                        mapped["x"] = float(candidate["x"]) * scale_x
                        mapped["y"] = float(candidate["y"]) * scale_y
                        mapped["width"] = float(candidate["width"]) * scale_x
                        mapped["height"] = float(candidate["height"]) * scale_y
                        rejections = self._plate_rejections(original_vehicle_crop, mapped)
                        if rejections:
                            for rejection in rejections:
                                state.rejections[rejection] = state.rejections.get(rejection, 0) + 1
                            continue
                        crop = crop_plate(
                            original_vehicle_crop, mapped,
                            crop_cfg.plate_padding_pixels, crop_cfg.plate_min_crop_height,
                            crop_cfg.plate_padding_ratio,
                        )
                        if crop is not None:
                            valid_plates.append((float(candidate.get("confidence", 0.0)), mapped, crop))
                if valid_plates:
                    plate_confidence, best_plate, plate_crop = max(valid_plates, key=lambda item: item[0])
                    plate_box = box_edges(best_plate)
            elif plate_predictions:
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
        vehicle_jpeg = encode_jpeg(vehicle_crop, cfg.images.jpeg_quality)
        plate_jpeg = encode_jpeg(plate_crop, cfg.images.plate_jpeg_quality) if plate_crop is not None else None
        state.best_score = score
        state.best = {
            "class": str(prediction.get("class") or "vehicle"),
            "vehicle_confidence": vehicle_confidence,
            "vehicle_jpeg": vehicle_jpeg,
            "vehicle_size": (vehicle_crop.shape[1], vehicle_crop.shape[0]),
            "vehicle_box": (vx1, vy1, vx2, vy2),
            "plate_detected": plate_crop is not None,
            "plate_thumb": RecentPlates.make_thumb(plate_crop) if plate_crop is not None else None,
            "plate_confidence": plate_confidence,
            "plate_jpeg": plate_jpeg,
            "plate_size": (plate_crop.shape[1], plate_crop.shape[0]) if plate_crop is not None else None,
            "plate_box": plate_box,
            "frame_size": (frame.shape[1], frame.shape[0]),
            "at": datetime.now(),
        }

    def _plate_rejections(self, vehicle_crop: np.ndarray, plate: Dict[str, Any]) -> List[str]:
        cfg = self.config.plate_validation
        confidence = float(plate.get("confidence", 0.0))
        if confidence < self.config.model.plate_conf_threshold:
            return ["confidence"]
        plate_width, plate_height = float(plate["width"]), float(plate["height"])
        if plate_width <= 0 or plate_height <= 0:
            return ["size"]
        aspect = plate_width / plate_height
        if not cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio:
            return ["aspect"]
        crop_height, crop_width = vehicle_crop.shape[:2]
        center_x, center_y = float(plate["x"]) / crop_width, float(plate["y"]) / crop_height
        if not cfg.min_center_x_ratio <= center_x <= cfg.max_center_x_ratio or center_y < cfg.min_center_y_ratio:
            return ["position"]
        width_ratio = plate_width / crop_width
        if not cfg.min_width_ratio <= width_ratio <= cfg.max_width_ratio:
            return ["size"]

        raw_crop = crop_plate(
            vehicle_crop, plate, self.config.crop.plate_padding_pixels,
            min_crop_height=0, padding_ratio=self.config.crop.plate_padding_ratio,
        )
        if raw_crop is None:
            return ["size"]
        small = raw_crop.shape[0] < cfg.min_height_px
        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        blurry = float(cv2.Laplacian(gray, cv2.CV_64F).var()) < cfg.min_sharpness
        reasons = []
        if small:
            reasons.append("small")
        if blurry:
            reasons.append("blurry")
        return reasons

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
        if features.plate_detection and not best["plate_detected"] and not features.store_without_plate:
            return 0
        track_id = f"v{track.id}"
        if best["plate_detected"] and features.dedupe_enabled:
            now = time.monotonic()
            similarity = self.recent_plates.best_match(
                best["plate_thumb"], now, best["class"], best["vehicle_box"],
            )
            if similarity >= features.dedupe_similarity:
                log.info("[vehicle] %s dropped as duplicate (similarity %.2f)", track_id, similarity)
                return 0
            self.recent_plates.add(best["plate_thumb"], now, best["class"], best["vehicle_box"])
            log.debug("[vehicle] %s plate dedupe similarity %.2f", track_id, similarity)
        stem = f"{self._session_tag}_{track_id}"
        disk_files: Optional[Dict[str, bytes]] = {} if features.save_images_to_disk else None
        if disk_files is not None:
            disk_files[f"vehicle_detection/{stem}_{self._safe_name(best['class'])}.jpg"] = best["vehicle_jpeg"]
            if best["plate_jpeg"] is not None:
                disk_files[f"plate_detection/{stem}_plate.jpg"] = best["plate_jpeg"]

        vx1, vy1, vx2, vy2 = best["vehicle_box"]
        px1, py1, px2, py2 = best["plate_box"]
        self.buffer.enqueue(VehicleRecord(
            event_uuid=str(uuid.uuid4()),
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
            if not best["plate_detected"]:
                rejected = self._format_rejections(state.rejections)
                if rejected:
                    status += f" (rejected: {rejected})"
            frame_width, frame_height = best["frame_size"]
            vehicle_width, vehicle_height = best["vehicle_size"]
            details = (f"frame {frame_width}x{frame_height}, vehicle {vehicle_width}x{vehicle_height} "
                       f"({len(best['vehicle_jpeg']) / 1024:.0f} KB)")
            if best["plate_jpeg"] is not None:
                plate_width, plate_height = best["plate_size"]
                details += f", plate {plate_width}x{plate_height} ({len(best['plate_jpeg']) / 1024:.0f} KB)"
            print(f"[vehicle] {track_id} {best['class']} ({best['vehicle_confidence']:.2f}) - {status} "
                  f"[{details}] [{state.attempts} tries]")
        return 1

    @staticmethod
    def _format_rejections(rejections: Dict[str, int]) -> str:
        order = ("confidence", "aspect", "position", "size", "small", "blurry")
        return ", ".join(f"{reason} {rejections[reason]}" for reason in order if rejections.get(reason))

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
