# Image Classification Evaluation

This directory contains Python scripts for running inference and benchmarking YOLO Image Classification models using three different inference backends:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The goal is to compare inference performance while maintaining consistent prediction outputs across all supported model formats.

> **Note**
>
> YOLO classification models expect an input image size of **224 × 224**. Images are automatically preprocessed before inference.

---

# Directory Structure

```text
Classification_Model_Evaluation/
│
├── classification_pt_model.py
├── classification_onnx_model.py
├── classification_engine_model.py
├── config.txt
├── Classification_Models/
├── cls-yaml/
└── README.md
```

---

# Available Scripts

| Script                           | Description                                              |
| -------------------------------- | -------------------------------------------------------- |
| `classification_pt_model.py`     | Runs inference using the Ultralytics YOLO (`.pt`) model. |
| `classification_onnx_model.py`   | Runs inference using the ONNX Runtime (`.onnx`) model.   |
| `classification_engine_model.py` | Runs inference using the TensorRT (`.engine`) model.     |

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

The inference scripts use the settings defined in `config.txt`, including model paths, input images, output directories, and other runtime parameters.

Update this file as needed before running the scripts.

---

# Notes

* NVIDIA GPU with CUDA support is required.
* The required classification model files (`.pt`, `.onnx`, and `.engine`) should be placed inside the `Classification_Models/` directory.
* The label file should be available inside the `cls-yaml/` directory.
* Refer to the main repository `README.md` for Docker setup and execution instructions.
