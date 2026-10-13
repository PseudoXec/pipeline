"""Every setting that only person mode uses: models, thresholds, ROI rules, face search, storage options."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

from core.tracker import TrackingConfig
from core.motion_gate import FrameSkipConfig

HERE = Path(__file__).resolve().parent


@dataclass
class FaceValidationConfig:
    enabled: bool = True
    min_aspect_ratio: float = 0.55
    max_aspect_ratio: float = 1.25
    min_height_ratio: float = 0.05
    max_height_ratio: float = 0.40
    max_center_y_ratio: float = 0.50
    min_center_x_ratio: float = 0.15
    max_center_x_ratio: float = 0.85
    min_eye_distance_ratio: float = 0.25
    landmark_expand_ratio: float = 0.15


@dataclass
class PersonConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {
        "weights": "here", "face_weights": "here",
    }

    weights: str = str(HERE / "models" / "person.onnx")
    face_weights: str = str(HERE / "models" / "face_detection_yunet_2023mar.onnx")

    imgsz: Union[int, List[int]] = 416
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    classes: Optional[str] = None
    agnostic_nms: bool = True

    roi_filter: bool = True
    roi_crop_detect: bool = False
    roi_crop_margin: float = 0.05
    roi_top_extra: float = 0.30
    min_width_ratio: float = 0.01
    max_width_ratio: float = 0.60
    min_height_ratio: float = 0.05
    max_height_ratio: float = 0.98
    edge_margin_ratio: float = 0.0

    padding_ratio: float = 0.08
    min_crop_height: int = 0
    jpeg_quality: int = 90
    face_jpeg_quality: int = 95

    face_enabled: bool = True
    face_score_threshold: float = 0.80
    face_validation: FaceValidationConfig = field(default_factory=FaceValidationConfig)
    face_search_region: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.6])
    face_margin: float = 0.2
    face_min_px: int = 40
    face_min_sharpness: float = 40.0
    face_ref_px: int = 64
    face_ref_sharpness: float = 100.0
    face_detect_max_side: int = 480

    save_images_to_disk: bool = False
    send_via_api: bool = True
    store_without_face: bool = True
    delete_row_after_api_send: bool = True

    tracking: TrackingConfig = field(default_factory=lambda: TrackingConfig(max_attempts=5, retry_interval_seconds=0.3, accept_score=0.60))
    frame_skip: FrameSkipConfig = field(default_factory=FrameSkipConfig)
