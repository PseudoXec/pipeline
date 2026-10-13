# Person mode models

`person.onnx` is the runtime person detector. Use the FP32 model exported at 416 px;
the Pi runtime loads it with ONNX Runtime on the CPU.

To export models, install the tools from the repository root with
`python -m pip install -r requirements-export.txt`. Export scripts and their dependencies
are separate from the runtime requirements.

`face_detection_yunet_2023mar.onnx` is the YuNet face detector from the OpenCV Zoo
(`models/face_detection_yunet`); check the license in that folder.

Paths in `modes/person/config.yaml` are relative to the `modes/person/` folder.
