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
api/       config.py/.yaml   Command-center endpoint URLs
           sender.py  roi_client.py  camera_source_client.py  result.py
storage/   config.py/.yaml   Shared SQLite buffer settings
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
- **SQLite buffer (`storage/buffer.py`)**: A SQLite-only batching writer, optional image-file writer, and separate sender thread; WAL mode, retry backoff (5 seconds up to 60 seconds, a pass stops after 3 straight failures), rejection cleanup, and three-day retention for unsent rows.
- **Unified API Sender (`api/sender.py`)**: Sends metadata to the existing endpoints for vehicle detections (`api.vehicle_endpoint_url`) and person events (`api.person_endpoint_url`).
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
- `save_images_to_disk: false` avoids optional JPEG copies; SQLite buffering remains enabled.
- `storage.max_reject_attempts: 3` drops a record only after three permanent API rejections;
  transient failures use exponential backoff and are not counted against that limit.
- Vehicle plate thumbnails are compared locally for 30 seconds to suppress repeated reports;
  this does not deduplicate an event whose API reply was lost. Configure
  `features.dedupe_enabled`, `dedupe_window_seconds`, `dedupe_similarity`, and `dedupe_min_box_iou`
  in vehicle mode (same class, overlapping vehicle box, and similar plate crop are all required).
- `api.send_vehicle_event_id: false` is off by default; enable it only if the command center API
  ignores repeated `EventId` values.

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