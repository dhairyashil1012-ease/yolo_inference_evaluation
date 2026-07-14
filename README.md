# YOLO Inference Benchmark & Evaluation Pipeline

This repository provides a Docker-based inference and benchmarking pipeline for pretrained **YOLO** models across three computer vision tasks:

* Image Classification
* Object Detection
* Semantic Segmentation

Each task supports inference using three different model formats:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The project is designed to help users compare inference performance across different inference backends while maintaining consistent prediction results.

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

Each task directory is a self-contained project containing:

* Dockerfile
* Source code
* Model directory
* Configuration file
* Sample images
* Output directories
* Task-specific README

---

# Supported Tasks

| Task                  | Supported Models          |
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

# Workflow

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

---

## Step 2: Download the TensorRT Docker Image

The Dockerfiles used in this repository are based on NVIDIA TensorRT Docker images.

TensorRT Docker images are available from the NVIDIA NGC Catalog:

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags

Pull one of the supported TensorRT images before building the project Docker images.

```bash
docker pull nvcr.io/nvidia/tensorrt:26.06-py3
```

or

```bash
docker pull nvcr.io/nvidia/tensorrt:26.04-py3
```

> **Note**
>
> Ensure that Docker, the NVIDIA GPU Driver, and the NVIDIA Container Toolkit are installed and configured correctly before pulling or running the TensorRT Docker images.

---

# Project Organization

Each task is an independent project with its own:

* Dockerfile
* Docker image
* Source code
* Configuration
* Model directory
* Sample images
* Output directories
* Task-specific README

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

Each task-specific README explains how to:

1. Build the Docker image.
2. Run inference using:

   * Ultralytics YOLO (`.pt`)
   * ONNX Runtime (`.onnx`)
   * TensorRT (`.engine`)
3. Configure runtime parameters.
4. View the generated output.

---

# Model Dependency

The supported model formats follow the workflow shown below:

```text
.pt
 │
 ▼
.onnx
 │
 ▼
.engine
```

* **ONNX (`.onnx`)** models are exported from the corresponding **PyTorch (`.pt`)** models.
* **TensorRT (`.engine`)** models are generated from the corresponding **ONNX (`.onnx`)** models.

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

This repository provides a simple, modular, and reproducible environment for evaluating YOLO inference performance across multiple deployment backends.

It enables users to:

* Run inference using PyTorch, ONNX Runtime, and TensorRT.
* Benchmark GPU inference performance.
* Compare outputs across different model formats.
* Execute the complete pipeline using Docker.

For task-specific setup and execution instructions, refer to the corresponding **README.md** inside each task directory.
