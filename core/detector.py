"""CPU-only ONNX Runtime inference for YOLOv8/YOLO11 detection exports."""
import ast
import gc
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

log = logging.getLogger("pipeline")

_inference_threads = 0


class InferenceError(RuntimeError):
    pass


class OnnxDetector:
    def __init__(self, session: Any):
        self.session = session
        self.input = session.get_inputs()[0]
        self.input_name = self.input.name
        self.output_name = session.get_outputs()[0].name
        self.input_hw = _shape_hw(self.input.shape)
        metadata = session.get_modelmeta().custom_metadata_map
        raw_names = metadata.get("names")
        if raw_names is None:
            raise InferenceError("ONNX model metadata does not contain class names ('names')")
        try:
            parsed_names = ast.literal_eval(raw_names)
        except (SyntaxError, ValueError) as error:
            raise InferenceError(f"Invalid ONNX class-name metadata: {error}") from error
        if isinstance(parsed_names, dict):
            self.names = {int(index): str(name) for index, name in parsed_names.items()}
        elif isinstance(parsed_names, (list, tuple)):
            self.names = {index: str(name) for index, name in enumerate(parsed_names)}
        else:
            raise InferenceError("ONNX class-name metadata must be a mapping or sequence")
        if not self.names:
            raise InferenceError("ONNX class-name metadata is empty")


def _runtime():
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("onnxruntime is required; install it with: pip install onnxruntime") from error
    return ort


def weights_exist(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".onnx")


def resolve_inference_threads(configured: int) -> int:
    if configured > 0:
        return configured
    return max(2, (os.cpu_count() or 4) - 1)


def normalize_imgsz(imgsz: "int | list | tuple") -> "int | List[int]":
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0])
        return [int(imgsz[0]), int(imgsz[1])]
    return int(imgsz)


def imgsz_hw(imgsz: "int | list | tuple") -> Tuple[int, int]:
    size = normalize_imgsz(imgsz)
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def limit_onnx_threads(configured: int) -> Optional[int]:
    """Set the thread count that subsequent sessions use; no runtime monkeypatching is performed."""
    global _inference_threads
    if configured < 0:
        _inference_threads = 0
        return None
    _inference_threads = resolve_inference_threads(configured)
    return _inference_threads


def _session_options():
    ort = _runtime()
    options = ort.SessionOptions()
    options.intra_op_num_threads = _inference_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _shape_hw(shape: list) -> Optional[Tuple[int, int]]:
    if len(shape) != 4:
        raise InferenceError(f"Expected ONNX input shape [1, 3, H, W], got {shape}")
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return height, width
    return None


def onnx_static_input_hw(weights_path: str) -> Optional[Tuple[int, int]]:
    if not weights_exist(weights_path):
        return None
    try:
        ort = _runtime()
        session = ort.InferenceSession(
            weights_path, sess_options=_session_options(), providers=["CPUExecutionProvider"],
        )
        return _shape_hw(session.get_inputs()[0].shape)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        log.warning("[detector] could not read ONNX input shape for %s: %s", weights_path, error)
        return None


def load_model(weights_path: str, device: Optional[str] = None) -> OnnxDetector:
    del device  # CPUExecutionProvider is intentionally the only supported execution provider.
    if not weights_exist(weights_path):
        raise FileNotFoundError(f"ONNX model weights not found at: {weights_path}")
    try:
        ort = _runtime()
        session = ort.InferenceSession(
            weights_path, sess_options=_session_options(), providers=["CPUExecutionProvider"],
        )
        return OnnxDetector(session)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise InferenceError(f"Could not load ONNX model {weights_path}: {error}") from error


def _input_hw(model: OnnxDetector, imgsz: "int | list | tuple") -> Tuple[int, int]:
    return model.input_hw or imgsz_hw(imgsz)


def _letterbox(image: np.ndarray, target_hw: Tuple[int, int]):
    target_height, target_width = target_hw
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = target_width - resized_width, target_height - resized_height
    left, top = int(round(pad_x / 2 - 0.1)), int(round(pad_y / 2 - 0.1))
    right, bottom = pad_x - left, pad_y - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return (
        tensor[None, ...],
        resized_width / width,
        resized_height / height,
        left,
        top,
    )


def warmup_model(
    model: OnnxDetector, imgsz: "int | list | tuple", frame_shape: Optional[Tuple[int, int]] = None,
) -> None:
    height, width = frame_shape if frame_shape else _input_hw(model, imgsz)
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        tensor, _, _, _, _ = _letterbox(dummy_frame, _input_hw(model, imgsz))
        model.session.run([model.output_name], {model.input_name: tensor})
    except Exception as error:
        log.warning("[detector] model warm-up failed (continuing anyway): %s", error)


def _class_name(model: OnnxDetector, class_index: int) -> str:
    return model.names.get(class_index, str(class_index))


def _postprocess(
    model: OnnxDetector,
    output: np.ndarray,
    original_hw: Tuple[int, int],
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    iou_threshold: float,
    classes: Optional[Set[str]],
    agnostic_nms: bool,
    offset_x: int,
    offset_y: int,
) -> List[Dict[str, Any]]:
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise InferenceError(f"Unsupported YOLO output shape: {predictions.shape}")
    if predictions.shape[0] == 4 + len(model.names):
        predictions = predictions.T
    elif predictions.shape[1] != 4 + len(model.names):
        raise InferenceError(
            f"Expected YOLO output with 4+{len(model.names)} channels, got {predictions.shape}"
        )

    if predictions.shape[1] == 0:
        return []
    class_scores = predictions[:, 4:]
    class_indexes = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_indexes]
    keep = confidences >= conf_threshold
    if classes:
        keep &= np.array([_class_name(model, int(index)) in classes for index in class_indexes])
    selected = np.flatnonzero(keep)
    if selected.size == 0:
        return []

    boxes = predictions[selected, :4].astype(np.float32, copy=True)
    selected_scores = confidences[selected].astype(float)
    selected_classes = class_indexes[selected].astype(int)
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y

    height, width = original_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid]
    selected_scores = selected_scores[valid]
    selected_classes = selected_classes[valid]
    if not len(boxes):
        return []

    nms_groups = [np.arange(len(boxes))] if agnostic_nms else [
        np.flatnonzero(selected_classes == class_id) for class_id in np.unique(selected_classes)
    ]
    retained = []
    for group in nms_groups:
        rects = [
            [float(boxes[index, 0]), float(boxes[index, 1]),
             float(boxes[index, 2] - boxes[index, 0]), float(boxes[index, 3] - boxes[index, 1])]
            for index in group
        ]
        indexes = cv2.dnn.NMSBoxes(
            rects, [float(selected_scores[index]) for index in group], conf_threshold, iou_threshold,
        )
        if len(indexes):
            retained.extend(int(group[int(index)]) for index in np.asarray(indexes).reshape(-1))

    retained.sort(key=lambda index: selected_scores[index], reverse=True)
    results = []
    for index in retained:
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        results.append({
            "x": (x1 + x2) / 2 + offset_x,
            "y": (y1 + y2) / 2 + offset_y,
            "width": x2 - x1,
            "height": y2 - y1,
            "class": _class_name(model, int(selected_classes[index])),
            "confidence": float(selected_scores[index]),
        })
    return results


def detect(
    model: OnnxDetector,
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: "int | list | tuple",
    classes: Optional[Set[str]] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
    agnostic_nms: bool = False,
) -> List[Dict[str, Any]]:
    if frame is None or frame.size == 0:
        return []
    source = frame
    offset_x = offset_y = 0
    if region is not None:
        region_x1, region_y1, region_x2, region_y2 = region
        region_x1, region_y1 = max(0, region_x1), max(0, region_y1)
        region_x2, region_y2 = min(frame.shape[1], region_x2), min(frame.shape[0], region_y2)
        if region_x2 <= region_x1 or region_y2 <= region_y1:
            return []
        source = np.ascontiguousarray(frame[region_y1:region_y2, region_x1:region_x2])
        offset_x, offset_y = region_x1, region_y1

    try:
        tensor, scale_x, scale_y, pad_x, pad_y = _letterbox(source, _input_hw(model, imgsz))
        output = model.session.run([model.output_name], {model.input_name: tensor})[0]
        return _postprocess(
            model, output, source.shape[:2], scale_x, scale_y, pad_x, pad_y,
            conf_threshold, iou_threshold, classes, agnostic_nms, offset_x, offset_y,
        )
    except Exception as error:
        if isinstance(error, InferenceError):
            raise
        raise InferenceError(f"Detection failed: {error}") from error


def release_models(*models: Any) -> None:
    for model in models:
        if isinstance(model, OnnxDetector):
            model.session = None
    gc.collect()
