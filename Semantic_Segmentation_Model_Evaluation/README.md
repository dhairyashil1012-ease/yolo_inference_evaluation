# Semantic Segmentation Evaluation

This directory contains a Docker-based inference pipeline for running YOLO Semantic Segmentation models using three different inference backends:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The objective is to compare inference performance while maintaining consistent segmentation results across all supported model formats.

> **Note**
>
> YOLO semantic segmentation models expect an input image size of **1024 × 2048**. Input images are automatically preprocessed before inference.

---

# Directory Structure

```text
Semantic_Segmentation_Model_Evaluation/
│
├── Dockerfile
├── README.md
├── requirements.txt
├── main.py
├── config.txt
│
├── src/
|    ├── segementation_pt_model.py
|    ├── segmentation_engine_model.py
|    └── segmentation_onnx_model.py
├── Images/
├── Sem_Models/
│
├── PT_Output/
├── ONNX_Output/
└── ENGINE_Output/
```

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

# Build the Docker Image

Navigate to the `Semantic_Segmentation_Model_Evaluation` directory and build the Docker image.

```bash
cd Semantic_Segmentation_Model_Evaluation

docker build -t yolo_inference_evaluation_segmentation .
```

---

# Run the Inference

## Ultralytics YOLO (.pt)

```bash
docker run --gpus all \
    -e MODEL_TYPE=pt \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_segmentation
```

---

## ONNX Runtime (.onnx)

```bash
docker run --gpus all \
    -e MODEL_TYPE=onnx \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_segmentation
```

---

## TensorRT (.engine)

```bash
docker run --gpus all \
    -e MODEL_TYPE=engine \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_segmentation
```

---

# Configuration

Runtime settings such as model paths, input image directory, output directory, and other inference parameters are configured in:

```text
config.txt
```

Update the configuration file as required before running the container.

---

# Model Dependencies

The three model formats are generated sequentially and depend on one another.

```text
.pt
 │
 ▼
.onnx
 │
 ▼
.engine
```

* The **ONNX (`.onnx`)** model is exported from the **PyTorch (`.pt`)** model. Therefore, the corresponding `.pt` model must be available inside the `Sem_Models/` directory before generating an ONNX model.

* The **TensorRT (`.engine`)** model is generated from the **ONNX (`.onnx`)** model. Therefore, the corresponding `.onnx` model must be available inside the `Sem_Models/` directory before building a TensorRT engine.

Ensure that the required model files are placed inside the `Sem_Models/` directory before running inference.

---

# Output

Inference results are saved automatically in the corresponding output directory:

* `PT_Output/`
* `ONNX_Output/`
* `ENGINE_Output/`

---

# Notes

* NVIDIA GPU with CUDA support is required.
* NVIDIA Container Toolkit must be installed to enable GPU access inside Docker.
* Input images should be placed inside the `Images/` directory.
* Update `config.txt` if you want to change model paths, input images, or inference settings.
