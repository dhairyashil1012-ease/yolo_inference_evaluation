# YOLO Inference Benchmark & Evaluation Pipeline

This repository demonstrates inference and benchmarking of pretrained **YOLO** models for the following computer vision tasks:

* Image Classification
* Object Detection
* Semantic Segmentation

Each task provides separate Python scripts for running inference using three different model formats:

* **YOLO (.pt)** – Ultralytics (PyTorch)
* **ONNX (.onnx)** – ONNX Runtime
* **TensorRT (.engine)** – TensorRT

The objective is to compare inference performance while maintaining consistent prediction outputs across all supported model formats.

> **Note**
>
> This project is designed for **GPU-enabled systems** and requires an NVIDIA GPU with CUDA support.

---

# Project Structure

```text
yolo_inference_evaluation/
│
├── Dockerfile
├── README.md
├── requirements.txt
├── Images/
│
├── Classification_Model_Evaluation/
│   ├── cls-yaml
│   ├── classification_pt_model.py
│   ├── classification_onnx_model.py
│   ├── classification_engine_model.py
│   ├── Classification_Models/
│   ├── config.txt
│   └── README.md
│
├── Detection_Model_Evaluation/
│   ├── detection-yaml
│   ├── detection_pt_model.py
│   ├── detection_onnx_model.py
│   ├── detection_engine_model.py
│   ├── Detection_Models/
│   ├── config.txt
│   └── README.md
│
└── Semantic_Segmentation_Model_Evaluation/
    ├── segmentation_pt_model.py
    ├── segmentation_onnx_model.py
    ├── segmentation_engine_model.py
    ├── config.txt
    └── README.md
```

Each task directory contains:

* Python scripts for `.pt`, `.onnx`, and `.engine` inference
* Model files
* Configuration file
* Task-specific README

---

# Prerequisites

* Docker
* NVIDIA GPU
* NVIDIA Driver
* NVIDIA Container Toolkit

---

# Getting Started

## 1. Clone the Repository

Clone the **Development_Branch_V3** branch.

```bash
git clone -b Development_Branch_V3 https://github.com/dhairyashil1012-ease/yolo_inference_evaluation.git

cd yolo_inference_evaluation
```

---

## 2. Build the Docker Image

```bash
docker build -t yolo_inference_evaluation .
```

---

## 3. Start the Docker Container

```bash
docker run -it \
    --gpus all \
    -p 8888:8888 \
    -v `pwd`:/workspace \
    -w /workspace \
    yolo_inference_evaluation:latest \
    bash
```

---

# Running the Code

Navigate to the required task directory.

## Image Classification

```bash
cd Classification_Model_Evaluation
```

Run any of the following:

```bash
python3 classification_pt_model.py
```

```bash
python3 classification_onnx_model.py
```

```bash
python3 classification_engine_model.py
```

---

## Object Detection

```bash
cd Detection_Model_Evaluation
```

Run:

```bash
python3 detection_pt_model.py
```

```bash
python3 detection_onnx_model.py
```

```bash
python3 detection_engine_model.py
```

---

## Semantic Segmentation

```bash
cd Semantic_Segmentation_Model_Evaluation
```

Run:

```bash
python3 segmentation_pt_model.py
```

```bash
python3 segmentation_onnx_model.py
```

```bash
python3 segmentation_engine_model.py
```

---

# Supported Tasks

| Task                  | Supported Model Formats   |
| --------------------- | ------------------------- |
| Image Classification  | `.pt`, `.onnx`, `.engine` |
| Object Detection      | `.pt`, `.onnx`, `.engine` |
| Semantic Segmentation | `.pt`, `.onnx`, `.engine` |

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

# Repository Goal

This repository provides a simple and modular implementation for benchmarking YOLO models across multiple inference backends.

It enables users to:

* Run inference using PyTorch, ONNX Runtime, and TensorRT
* Compare inference performance
* Understand the inference pipeline for each backend
* Use a Docker-based environment for reproducible execution

Refer to the **README.md** inside each task directory for task-specific details and configuration.
