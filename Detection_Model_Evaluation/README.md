# Object Detection Evaluation

This directory contains Python scripts for running inference and benchmarking YOLO Object Detection models using three different inference backends:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The objective is to compare inference performance while maintaining consistent detection results across all supported model formats.

> **Note**
>
> YOLO object detection models expect an input image size of **640 × 640**. Images are automatically preprocessed before inference.

---

# Directory Structure

```text
Detection_Model_Evaluation/
│
├── detection_pt_model.py
├── detection_onnx_model.py
├── detection_engine_model.py
├── config.txt
├── Detection_Models/
├── detection-yaml/
└── README.md
```

---

# Available Scripts

| Script                      | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| `detection_pt_model.py`     | Runs inference using the Ultralytics YOLO (`.pt`) model. |
| `detection_onnx_model.py`   | Runs inference using the ONNX Runtime (`.onnx`) model.   |
| `detection_engine_model.py` | Runs inference using the TensorRT (`.engine`) model.     |

---

# Supported Model Formats

| Model Format | Inference Backend          |
| ------------ | -------------------------- |
| `.pt`        | Ultralytics YOLO (PyTorch) |
| `.onnx`      | ONNX Runtime               |
| `.engine`    | TensorRT                   |

---

# Workflow

```text
YOLO (.pt)
      │
      ▼
PyTorch Inference
      │
      ▼
ONNX Runtime (.onnx)
      │
      ▼
TensorRT (.engine)
```

---

# Configuration

The inference scripts use the settings defined in `config.txt`, including model paths, input images, output directories, confidence thresholds, and other runtime parameters.

Update this file as needed before running the scripts.

---

# Notes

* NVIDIA GPU with CUDA support is required.
* The required detection model files (`.pt`, `.onnx`, and `.engine`) should be placed inside the `Detection_Models/` directory.
* The label file should be available inside the `detection-yaml/` directory.
* Refer to the main repository `README.md` for Docker setup and execution instructions.
