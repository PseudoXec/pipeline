"""Unified WebSocket sender for vehicle and person detection metadata."""
import base64
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from api.result import SendResult
from api.ws_client import PiWebSocketClient

if TYPE_CHECKING:
    from storage.buffer import PersonRecord, VehicleRecord

log = logging.getLogger("pipeline")


def _number(value: Optional[float], digits: int) -> Optional[str]:
    return None if value is None else f"{round(float(value), digits):.{digits}f}"


class ApiSender:
    """Sends detection metadata for both vehicle and person flows via WebSocket."""

    def __init__(self, api_config, camera_info=None):
        self.config = api_config
        self.camera_info = camera_info
        if api_config.camera_id is None:
            raise ValueError("api.camera_id is required")
        self.ws_client = PiWebSocketClient(api_config.websocket_host, str(api_config.camera_id))

    @property
    def camera_name(self) -> Optional[str]:
        return getattr(self.camera_info, "name", None) if self.camera_info else None

    @property
    def camera_ip(self) -> Optional[str]:
        return getattr(self.camera_info, "ip_address", None) if self.camera_info else None

    @property
    def camera_location(self) -> Optional[str]:
        return getattr(self.camera_info, "location", None) if self.camera_info else None

    def send(self, record: Any) -> Tuple[SendResult, Optional[str]]:
        """Dispatch to send_vehicle or send_person based on record type."""
        cls_name = record.__class__.__name__
        if cls_name == "VehicleRecord" or hasattr(record, "vehicle_confidence"):
            result = self.send_vehicle(record)
            return (result, None) if result == SendResult.SENT else (result, str(result.value))
        elif cls_name == "PersonRecord" or hasattr(record, "event_uuid"):
            return self.send_person(record)
        else:
            log.warning("[api] unknown record type: %s", type(record))
            return SendResult.FAILED, f"unknown record type {type(record)}"

    def send_vehicle(self, record: "VehicleRecord") -> SendResult:
        payload: Dict[str, Any] = {
            "TrackId": record.track_id,
            "DetectionConfidence": round(record.vehicle_confidence or 0.0, 2),
            "DetectedVehicle": record.vehicle_class or "Unknown",
            "TimeStamp": record.detected_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if self.config.camera_id is not None:
            payload["CameraTargetID"] = self.config.camera_id
        if self.camera_name:
            payload["CameraName"] = self.camera_name
        if self.camera_ip:
            payload["CameraIpAddress"] = self.camera_ip
        if self.camera_location:
            payload["CameraLocation"] = self.camera_location
        image = record.plate_image_jpeg or record.vehicle_image_jpeg   # no plate -> send the vehicle crop instead
        if image:
            payload["Image"] = base64.b64encode(image).decode()

        try:
            self.ws_client.send(payload)
            return SendResult.SENT
        except ConnectionError as error:
            log.warning("[api-ws] failed to send vehicle %s: %s", record.track_id, error)
            return SendResult.FAILED
        except (TypeError, ValueError) as error:
            log.error("[api-ws] vehicle %s can never be sent, dropping it: %s", record.track_id, error)
            return SendResult.REJECTED

    def send_person(self, record: "PersonRecord") -> Tuple[SendResult, Optional[str]]:
        fields: Dict[str, Optional[str]] = {
            "TrackId": record.track_id,
            "CameraId": None if self.config.camera_id is None else str(self.config.camera_id),
            "CameraName": self.camera_name,
            "CameraIpAddress": self.camera_ip,
            "CameraLocation": self.camera_location,
            "TimeStamp": record.detected_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "PersonConfidence": _number(record.person_confidence, 4),
            "FaceDetected": "true" if record.face_detected else "false",
            "FaceLandmarks": record.face_landmarks,
            "FaceQualityScore": _number(record.face_quality_score, 4),
        }
        payload = {k: v for k, v in fields.items() if v not in (None, "")}
        payload["PersonImage"] = base64.b64encode(record.person_image_jpeg).decode()
        if record.face_image_jpeg:
            payload["FaceImage"] = base64.b64encode(record.face_image_jpeg).decode()

        try:
            self.ws_client.send(payload)
            return SendResult.SENT, None
        except ConnectionError as error:
            log.warning("[api-ws] failed to send person %s: %s", record.track_id, error)
            return SendResult.FAILED, f"websocket: {error}"
        except (TypeError, ValueError) as error:
            log.error("[api-ws] person %s can never be sent, dropping it: %s", record.track_id, error)
            return SendResult.REJECTED, f"not sendable: {error}"
