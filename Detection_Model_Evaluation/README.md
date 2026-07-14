# Object Detection Evaluation

This directory provides a Docker-based pipeline for running inference and benchmarking YOLO Object Detection models using three inference backends:

* **Ultralytics YOLO (.pt)**
* **ONNX Runtime (.onnx)**
* **TensorRT (.engine)**

The objective is to compare inference performance while maintaining consistent detection results across all supported model formats.

> **Note**
>
> YOLO object detection models expect an input image size of **640 × 640**. Input images are automatically preprocessed before inference.

---

# Directory Structure

```text
Detection_Model_Evaluation/
│
├── Dockerfile
├── README.md
├── requirements.txt
├── main.py
├── config.txt
│
├── src/
|   ├── detection_engine_model.py
|   ├── detection_onnx_model.py
|   ├── detection_pt_model.py
├── Detection_Models/
├── detection-labels/
├── Images/
│
├── PT_Output/
├── ONNX_Output/
└── ENGINE_Output/
```

---

# Build the Docker Image

Navigate to the `Detection_Model_Evaluation` directory and build the Docker image.

```bash
cd Detection_Model_Evaluation

docker build -t yolo_inference_evaluation_detection .
```

---

# Run the Container

The inference backend is selected using the `MODEL_TYPE` environment variable.

## PyTorch (.pt)

```bash
docker run --gpus all \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_detection
```

or

```bash
docker run --gpus all \
    -e MODEL_TYPE=pt \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_detection
```

---

## ONNX Runtime (.onnx)

```bash
docker run --gpus all \
    -e MODEL_TYPE=onnx \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_detection
```

---

## TensorRT (.engine)

```bash
docker run --gpus all \
    -e MODEL_TYPE=engine \
    -v $(pwd):/workspace \
    yolo_inference_evaluation_detection
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

---

# Configuration

Runtime settings such as model paths, input images, output directories, confidence threshold, and other parameters are defined in `config.txt`.

Modify this file as needed before running inference.

---

# Model Dependencies

> **Important**
>
> The model formats have the following dependency chain:
>
> * **ONNX (`.onnx`)** models are exported from the original **PyTorch (`.pt`)** model. Therefore, the corresponding `.pt` model must be available inside the `Detection_Models/` directory before generating or using an ONNX model.
>
> * **TensorRT (`.engine`)** models are built from the **ONNX (`.onnx`)** model. Therefore, the corresponding `.onnx` model must be present inside the `Detection_Models/` directory before generating or using a TensorRT engine.

---

# Output

Inference results are automatically saved to the corresponding output directory based on the selected inference backend:

* `PT_Output/`
* `ONNX_Output/`
* `ENGINE_Output/`

---

# Notes

* NVIDIA GPU with CUDA support is required.
* Install the NVIDIA Container Toolkit to enable GPU access inside Docker containers.
* Place all required model files inside the `Detection_Models/` directory.
* Place the label file inside the `detection-labels/` directory.
* Input images should be placed inside the `Images/` directory.
