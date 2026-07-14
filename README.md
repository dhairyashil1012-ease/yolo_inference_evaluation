# YOLO Inference Benchmark & Evaluation Pipeline

This repository provides a Docker-based inference and benchmarking pipeline for pretrained **YOLO** models across three computer vision tasks:

* Image Classification
* Object Detection
* Semantic Segmentation

Each task supports inference using the following model formats:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The primary objective of this repository is to benchmark and compare inference performance across different deployment backends while maintaining consistent prediction results.

---

# Repository Structure

```text
yolo_inference_evaluation/
│
├── Classification_Model_Evaluation/
├── Detection_Model_Evaluation/
├── Semantic_Segmentation_Model_Evaluation/
│
└── README.md
```

Each task directory is a self-contained project that includes:

* Dockerfile
* Source code
* Configuration file
* Model directory
* Sample input images
* Output directories
* Task-specific README

---

# Supported Tasks

| Task                  | Supported Model Formats   |
| --------------------- | ------------------------- |
| Image Classification  | `.pt`, `.onnx`, `.engine` |
| Object Detection      | `.pt`, `.onnx`, `.engine` |
| Semantic Segmentation | `.pt`, `.onnx`, `.engine` |

---

# Supported Model Formats

| Model Format | Inference Backend          |
| ------------ | -------------------------- |
| `.pt`        | Ultralytics YOLO (PyTorch) |
| `.onnx`      | ONNX Runtime               |
| `.engine`    | TensorRT                   |

---

# Inference Workflow

```text
YOLO (.pt)
      │
      ▼
ONNX (.onnx)
      │
      ▼
TensorRT (.engine)
```

---

# Getting Started

## Step 1: Clone the Repository

Clone the **Development_Branch_V3** branch and navigate to the project directory.

```bash
git clone -b Development_Branch_V3 https://github.com/dhairyashil1012-ease/yolo_inference_evaluation.git

cd yolo_inference_evaluation
```

## Step 2: Build and Run

Each task directory contains its own Dockerfile and is designed to be built independently.

Navigate to the task you want to execute:

```text
Classification_Model_Evaluation/
```

or

```text
Detection_Model_Evaluation/
```

or

```text
Semantic_Segmentation_Model_Evaluation/
```

Build the Docker image by following the instructions provided in the corresponding task-specific **README.md**.

> **Note**
>
> The Dockerfiles use the NVIDIA TensorRT base image (`nvcr.io/nvidia/tensorrt`). If the base image is not available locally, Docker will automatically download it during the build process.

---

# Project Organization

Each task directory contains:

* Dockerfile
* `main.py`
* Source code (`src/`)
* Configuration (`config.txt`)
* Model directory
* Input images
* Output directories
* Task-specific documentation

The task-specific README explains how to:

1. Build the Docker image.
2. Run inference using:

   * Ultralytics YOLO (`.pt`)
   * ONNX Runtime (`.onnx`)
   * TensorRT (`.engine`)
3. Configure runtime parameters.
4. View the generated outputs.

---

# Model Dependency

The supported model formats follow the workflow below:

```text
.pt
 │
 ▼
.onnx
 │
 ▼
.engine
```

* The **ONNX (`.onnx`)** model is exported from the corresponding **PyTorch (`.pt`)** model.
* The **TensorRT (`.engine`)** model is generated from the corresponding **ONNX (`.onnx`)** model.

Ensure that the required model files are available before running inference.

---

# Requirements

* Ubuntu
* Docker
* NVIDIA GPU
* NVIDIA Driver
* NVIDIA Container Toolkit

---

# Repository Goal

This repository offers a modular and reproducible environment for evaluating YOLO inference performance across multiple deployment backends.

It enables users to:

* Run inference using PyTorch, ONNX Runtime, and TensorRT.
* Benchmark GPU inference performance.
* Compare prediction outputs across different model formats.
* Execute the complete inference pipeline using Docker.

Refer to the **README.md** inside each task directory for detailed setup, configuration, and execution instructions.
