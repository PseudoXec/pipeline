# Lean Detection Pipeline for Raspberry Pi 5

Optimized detection pipeline for Raspberry Pi supporting two lean detection flows:

---

### VEHICLE DETECTION FLOW
`vehicle detect → crop → plate detect → crop → send metadata`

### PERSON DETECTION FLOW
`person detect → crop → face detect → crop → send metadata`

---

## 1. Project Structure

```
run.py                       Entry point: reads settings, initializes buffer, runs camera loop
config/    config.yaml       Global switches and mode settings
           config.py         Combines settings into PipelineConfig
           loader.py         Reads yaml, resolves relative paths, merges config.local.yaml
core/                        Shared engine: detector.py, geometry.py, tracker.py (ByteTrack-style), urls.py
camera/    config.py/.yaml   Camera + ROI polygon
           camera.py  isapi_client.py  roi_manager.py  resolve.py  frame_loop.py
api/       config.py/.yaml   Command-center WebSocket host, camera id, ROI endpoint
           sender.py  ws_client.py  roi_client.py  camera_source_client.py  result.py
storage/   config.py/.yaml   Send buffer settings (disk-copy folder)
           buffer.py  reset_buffer.py
control/   mode_manager.py   Detection mode manager
modes/     base.py  registry.py  idle.py
  vehicle/ config.py/.yaml   Vehicle mode settings
           mode.py  geometry.py  models/
  person/  config.py/.yaml   Person mode settings
           mode.py  face.py  quality.py  geometry.py  models/
deploy/    detection-pipeline.service
docs/      CHANGES.md  person-api-contract.md
```

## 2. Shared Services

Both flows share:
- **In-memory buffer (`storage/buffer.py`)**: The frame loop enqueues records (never blocks); a background thread dequeues them and sends over WebSocket. Max 500 queued records: while the socket is down they wait and are sent as soon as it reconnects; if the queue fills, new detections are dropped with a warning. Unsent records are lost on restart/power loss. Optional image-file writer kept for spot checks.
- **Unified API Sender (`api/sender.py`, `api/ws_client.py`)**: Sends vehicle and person metadata as JSON over one WebSocket (`api.websocket_host`, `api.camera_id`); images are base64-encoded inside the JSON. Reconnects automatically with backoff.
- **ROI Manager (`camera/roi_manager.py`)**: Synchronizes detection polygon with the command center.

## 3. Installation

Install the CPU inference runtime and pipeline dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements-export.txt` contains the additional Ultralytics, NCNN, and PNNX tools
used only to export models; they are not imported by the runtime pipeline.

## 4. Configuration

- `runtime.inference_threads` defaults to `3` for the Pi 5; set it back to `2` if FPS drops.
- Both modes use `frame_skip.enabled`, `motion_threshold`, `follow_every_n`, and
  `idle_heartbeat_seconds` to gate idle inference. Single-image and folder runs always infer.
- `save_images_to_disk: false` avoids optional JPEG copies; the in-memory send queue is always on.
- Delivery only needs `api.websocket_host` and `api.camera_id`. `roi_endpoint_url` is still used for the HTTP ROI / camera-source fetch.
- Vehicle plate thumbnails are compared locally for 30 seconds to suppress repeated reports;
  this does not deduplicate an event whose API reply was lost. Configure
  `features.dedupe_enabled`, `dedupe_window_seconds`, `dedupe_similarity`, and `dedupe_min_box_iou`
  in vehicle mode (same class, overlapping vehicle box, and similar plate crop are all required).

| Config key | Default | Behavior |
|---|---:|---|
| `vehicle.features.store_without_plate` | `true` | Store and send the vehicle event even when no valid plate crop is found. |
| `person.store_without_face` | `true` | Store and send the person event even when no valid face crop is found. |
| `vehicle.plate_validation.enabled` | `true` | Require confidence, geometry, minimum pixel height, and sharpness checks before making a plate crop. |
| `person.face_validation.enabled` | `true` | Require face geometry and sane YuNet landmarks before making a face crop; the original-resolution size/sharpness gate also applies. |
| `vehicle.tracking.no_plate_retry_interval_seconds` / `no_plate_max_attempts` | `0.5` / `12` | Keep looking for a valid plate before storing a plate-less vehicle event. |
| `person.tracking.no_face_retry_interval_seconds` / `no_face_max_attempts` | `0.5` / `12` | Keep looking for a valid face before storing a face-less person event. |
- Person mode uses `face_detect_max_side: 480` to cap YuNet input while its validation
  quality gates operate on the original-resolution crop.
- Put the camera URL with credentials in the ignored `camera/config.local.yaml`; the committed
  camera config contains only a placeholder.

## 5. Running the Pipeline

```bash
# Run with camera using default mode (vehicle)
python run.py

# Run specifically in vehicle mode
python run.py --mode vehicle

# Run specifically in person mode
python run.py --mode person

# Run on a single image or folder of images
python run.py --mode vehicle --image test.jpg
python run.py --mode person --folder ./imgs
```