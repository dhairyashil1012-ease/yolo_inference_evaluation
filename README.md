# YOLO Inference Benchmark & Evaluation Pipeline

This repository demonstrates the complete inference and benchmarking pipeline for pretrained YOLO models across **Image Classification**, **Object Detection**, and **Semantic Segmentation** tasks.

Each notebook walks through the entire workflow, including:

- Running inference using the original **Ultralytics YOLO (.pt)** model
- Converting the model to **ONNX (.onnx)**
- Converting the ONNX model to **TensorRT (.engine)**
- Running inference on all three model formats
- Comparing performance and evaluating inference speed on a GPU

The notebooks separate the inference pipeline into its individual stages to provide a clear understanding of what happens internally during model execution.

The workflow covers:

- **Preprocessing** – Preparing input images for inference
- **Model Inference** – Executing inference using PyTorch, ONNX Runtime, and TensorRT
- **Postprocessing** – Processing model outputs into final predictions

The primary objective of this repository is to understand both the **deployment workflow** and the **performance differences** between `.pt`, `.onnx`, and `.engine` models while maintaining identical prediction outputs.

> **Note**
> This repository is designed and tested for **GPU-enabled systems**. CUDA-compatible NVIDIA GPUs are required to run the notebooks and benchmark `.pt`, `.onnx`, and `.engine` models.

---
# Project Structure

The repository is organized as follows:

```text
YOLO_Inference_Benchmark_Evaluation/
│
├── README.md                     # Project overview
├── requirements.txt
├── INSTALLATION.md               # Common installation guide
│
├── Images-20260702T141014Z-3-001/
│   └── Images/
│
├── Classification_Model_Evaluation/
│   ├── README.md                 # Classification-specific guide
│   ├── Classification_Model_Evaluation.ipynb
│   ├── Classification_Models/
│   └── cls-yaml/
│
├── Detection_Model_Evaluation/
│   ├── README.md                 # Detection-specific guide
│   ├── Detection_Model_Evaluation.ipynb
│   ├── Detection_Models/
│   └── detection-yaml/
│
└── Semantic_Segmentation_Model_Evaluation/
    ├── README.md                 # Segmentation-specific guide
    ├── Semantic_Segmentation_Model_Evaluation.ipynb
    ├── Sem_Models/
    └── sem-yaml/
```

### Directory Overview

- **Classification_Model_Evaluation/** – Notebook, models, and configuration files for image classification.
- **Detection_Model_Evaluation/** – Notebook, models, and configuration files for object detection.
- **Semantic_Segmentation_Model_Evaluation/** – Notebook, models, and configuration files for semantic segmentation.
- **Images-20260702T141014Z-3-001/Images/** – Sample input images used for inference.
- **requirements.txt** – Python package dependencies.
- **README.md** – Project documentation.
# Prerequisites

- Python 3.10 or above
- NVIDIA GPU
- CUDA installed
- Compatible PyTorch version
- Internet connection (only required for downloading the model)
- Ubuntu system

---

# Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/dhairyashil1012-ease/yolo_inference_evaluation

cd yolo_inference_evaluation
```

---

## Step 2: Install Dependencies

Install all required packages.

```bash
pip install -r requirements.txt
```

run this command on you cli
```bash
jupyter-notebook
```

---
## Supported Tasks

| Task                  | Notebook                                       |
|---------------------- |------------------------------------------------|
| Image Classification  | `Classification_Model_Evaluation.ipynb`        |
| Object Detection      | `Detection_Model_Evaluation.ipynb`             |
| Semantic Segmentation | `Semantic_Segmentation_Model_Evaluation.ipynb` |


## Supported Model Formats

| Model Format | Framework                  |
|--------------|----------------------------|
| `.pt`        | Ultralytics YOLO (PyTorch) |
| `.onnx`      | ONNX Runtime               |
| `.engine`    | TensorRT                   |




## Workflow

```text
YOLO (.pt)
    │
    ▼
PyTorch Inference
    │
    ▼
Export to ONNX
    │
    ▼
ONNX Runtime Inference
    │
    ▼
Export to TensorRT
    │
    ▼
TensorRT Inference
```



# Summary

This notebook is intended to help users understand how a YOLO Classification model performs inference internally.

Instead of relying entirely on Ultralytics' built-in pipeline, each stage is implemented separately to provide a clear understanding of:

- Image preprocessing
- Batch creation
- Model inference
- Prediction decoding

The same workflow can later be extended to ONNX and TensorRT models while maintaining a similar inference pipeline.