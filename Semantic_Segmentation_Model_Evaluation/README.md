# Semantic Segmentation Evaluation

This directory contains Python scripts for running inference and benchmarking YOLO Semantic Segmentation models using three different inference backends:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The objective is to compare inference performance while maintaining consistent segmentation results across all supported model formats.

> **Note**
>
> YOLO semantic segmentation models expect an input image size of **1024 × 2048**. Images are automatically preprocessed before inference.

---

# Directory Structure

```text
Semantic_Segmentation_Model_Evaluation/
│
├── segmentation_pt_model.py
├── segmentation_onnx_model.py
├── segmentation_engine_model.py
├── config.txt
├── Sem_Models/
└── README.md
```

---

# Available Scripts

| Script                         | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `segmentation_pt_model.py`     | Runs inference using the Ultralytics YOLO (`.pt`) model. |
| `segmentation_onnx_model.py`   | Runs inference using the ONNX Runtime (`.onnx`) model.   |
| `segmentation_engine_model.py` | Runs inference using the TensorRT (`.engine`) model.     |

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
* The required segmentation model files (`.pt`, `.onnx`, and `.engine`) should be placed inside the `Sem_Models/` directory.
* Refer to the main repository `README.md` for Docker setup and execution instructions.
