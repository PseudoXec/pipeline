"""In-memory detection buffer for both vehicle and person flows.

Records are enqueued by the frame loop (never blocks) and sent via WebSocket
by a background thread. If the WebSocket is temporarily disconnected, records
wait in the queue until it reconnects. Records are not persisted across restarts.

Optional disk image saving is retained for spot-checking purposes.
"""
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

log = logging.getLogger("pipeline")

_DEFAULT_MAX_QUEUE = 500
_RETRY_START_SECONDS = 1.0
_RETRY_MAX_SECONDS = 15.0
_KEEPALIVE_SECONDS = 15.0


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode an OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    return buffer.tobytes()


@dataclass
class VehicleRecord:
    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    event_uuid: Optional[str] = None              # kept on the record; not sent
    vehicle_image_jpeg: Optional[bytes] = None
    vehicle_box_x1: Optional[float] = None
    vehicle_box_y1: Optional[float] = None
    vehicle_box_x2: Optional[float] = None
    vehicle_box_y2: Optional[float] = None
    plate_detected: bool = False
    plate_confidence: Optional[float] = None
    plate_image_jpeg: Optional[bytes] = None
    plate_box_x1: Optional[float] = None
    plate_box_y1: Optional[float] = None
    plate_box_x2: Optional[float] = None
    plate_box_y2: Optional[float] = None
    detected_at: datetime = field(default_factory=datetime.now)
    disk_files: Optional[Dict[str, bytes]] = None


@dataclass
class PersonRecord:
    event_uuid: str
    device_id: str
    session_id: str
    track_id: str
    person_confidence: float
    person_image_jpeg: bytes
    face_detected: bool
    detected_at: datetime = field(default_factory=datetime.now)
    face_image_jpeg: Optional[bytes] = None
    face_landmarks: Optional[str] = None          # JSON: [[x, y] x 5]
    face_quality_score: Optional[float] = None
    disk_files: Optional[Dict[str, bytes]] = None


class DetectionBuffer:
    """In-memory queue buffer used by both vehicle and person flows."""

    def __init__(self, config, camera_info=None, api_sender=None):
        self.config = config
        self.output_dir = config.storage.output_dir
        self.api_sender = api_sender

        self._queue: "queue.Queue[Union[VehicleRecord, PersonRecord]]" = queue.Queue(
            maxsize=_DEFAULT_MAX_QUEUE
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_keepalive = time.monotonic()

    def start(self) -> "DetectionBuffer":
        self._thread = threading.Thread(target=self._sender_loop, name="detection-buffer", daemon=True)
        self._thread.start()
        log.info("[storage] in-memory detection buffer started (max queue: %d)", _DEFAULT_MAX_QUEUE)
        return self

    def enqueue(self, record: Union[VehicleRecord, PersonRecord]) -> None:
        """Never blocks. When the queue is full the OLDEST waiting record is dropped to make room, so after a
        long outage the newest detections are the ones kept."""
        while True:
            try:
                self._queue.put_nowait(record)
                return
            except queue.Full:
                try:
                    old = self._queue.get_nowait()
                except queue.Empty:
                    continue                    # the sender just freed a slot - try again
                log.warning("[storage] queue full (%d), dropped oldest %s %s to keep newest %s",
                            _DEFAULT_MAX_QUEUE, old.__class__.__name__, old.track_id, record.track_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self.api_sender is not None:
            self.api_sender.ws_client.close()
        log.info("[storage] detection buffer stopped (dropped %d unsent record(s))", self._queue.qsize())

    def _sender_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self.api_sender is not None and self.config.switches.api:
                    self._keepalive_if_due()
                continue

            self._write_disk_files(record)

            if self.api_sender is None or not self.config.switches.api:
                continue
            if not self._wants_send(record):
                continue

            # Hold this record until the WebSocket takes it. Everything behind it waits in the queue
            # (max _DEFAULT_MAX_QUEUE); the frame loop drops new detections with a warning when it is full.
            delay = _RETRY_START_SECONDS
            while True:
                try:
                    result, detail = self._send(record)
                except Exception as error:      # a bug/odd record must never kill the sender thread or block the queue
                    log.exception("[storage] dropping %s %s after unexpected error: %s",
                                  record.__class__.__name__, record.track_id, error)
                    break
                if result.name == "REJECTED":
                    log.warning("[storage] %s %s dropped: %s", record.__class__.__name__, record.track_id, detail)
                    break
                if result.name != "FAILED":
                    break
                if self._stop_event.is_set():
                    log.warning("[storage] shutting down - %s %s not sent", record.__class__.__name__, record.track_id)
                    break
                log.warning("[storage] %s %s not sent (%s) - retrying in %.0fs",
                            record.__class__.__name__, record.track_id, detail, delay)
                self._stop_event.wait(delay)
                delay = min(delay * 2, _RETRY_MAX_SECONDS)

    def _wants_send(self, record: Any) -> bool:
        if isinstance(record, VehicleRecord):
            return bool(self.config.vehicle.features.send_via_api)
        if isinstance(record, PersonRecord):
            return bool(self.config.person.send_via_api)
        return False

    def _send(self, record: Any):
        if isinstance(record, VehicleRecord):
            return self.api_sender.send_vehicle(record), "websocket"
        result, detail = self.api_sender.send_person(record)
        return result, detail

    def _keepalive_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_keepalive >= _KEEPALIVE_SECONDS:
            self._last_keepalive = now
            self.api_sender.ws_client.keepalive()

    def _write_disk_files(self, record: Any) -> None:
        files: Optional[Dict[str, bytes]] = getattr(record, "disk_files", None)
        if not self.output_dir or not files:
            return
        for relative_path, data in files.items():
            try:
                path = os.path.join(self.output_dir, relative_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as error:
                log.warning("[storage] could not write %s: %s", relative_path, error)

    def status(self) -> Dict[str, Any]:
        return {
            "queue_depth": self._queue.qsize(),
            "queue_max": _DEFAULT_MAX_QUEUE,
            "sender_alive": bool(self._thread and self._thread.is_alive()),
        }
