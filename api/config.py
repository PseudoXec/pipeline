"""Command-center API settings."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ApiConfig:
    websocket_host: str = ""                     # WebSocket metadata endpoint host (required)

    roi_endpoint_url: Optional[str] = None       # also used to look up the camera source
    camera_id: Optional[int] = None
    roi_fetch_timeout_seconds: float = 5.0
    roi_coordinates_are_pixels: bool = False
    roi_reference_width: Optional[int] = None
    roi_reference_height: Optional[int] = None
    roi_poll_interval_seconds: float = 30.0
