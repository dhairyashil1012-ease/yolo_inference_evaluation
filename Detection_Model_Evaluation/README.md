# Detection_Model_Evaluation.ipynb

This notebook demonstrates the complete inference pipeline for YOLO Object Detection models, including model conversion from PyTorch (.pt) to ONNX (.onnx) and TensorRT (.engine), followed by inference and performance benchmarking across all three model formats.

1. **Ultralytics YOLO (.pt)**
2. **ONNX Runtime (.onnx)**
3. **TensorRT (.engine)**

The notebook is divided into three phases to illustrate the complete model inference pipeline while maintaining consistent prediction outputs across all model formats.

> **Note**
>
> YOLO object detection models expect an input image size of **640 × 640**. During preprocessing, every input image is resized to **640 × 640** to ensure compatibility with the model and to obtain accurate predictions.
---

# Workflow

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

---

## NoteBook
Open Detection_Model_Evaluation.ipynb in Jupyter Notebook.

Review the notebook first, then execute the IMPORT REQUIREMENTS cell.   

# Phase 1 – Ultralytics YOLO (.pt) Inference

In this phase, inference is performed using the original pretrained Ultralytics YOLO **Object Detection** model.

The notebook covers:

- Loading a pretrained YOLO Detection model
- Image preprocessing
- Running inference on batch of images
- Postprocessing prediction results



## Downloading the Model

Ultralytics automatically downloads the model if it is not already available in your system.
Just run below code which is already present in you notebook just execute that cell

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```
After executing above code will see that model in your working directory just  manually place the downloaded model inside:

```text
Detection_Model_Evaluation/ 
            └── Detection_Models/ 
                            └── yolo26n.pt

```

If you use a different model, update the model path/name in the notebook accordingly .

---
## Images Sample Input :
Refere This Drive Link Where Sample Images are present or You can use yours 
🔗 **[Google Drive – Sample Input Images](https://drive.google.com/drive/folders/1MuNF6ytZHTcroBpnlwIUoWkaxl0oTTOS)**

---

## COCO Labels

YOLO object detection models are trained on the **COCO** dataset. During inference, Ultralytics automatically maps the predicted class indices to their corresponding COCO class names using the dataset configuration.

The COCO configuration is not required for Phase 1, but it is required for Phase 2 and Phase 3.

The COCO dataset configuration is available here:

**[ImageNet.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)**



Download the configuration file and place it in:

Detection_Model_Evaluation/
└── detection-yaml/
    └── coco.yaml








> **Troubleshooting**
>
> If you encounter any issues while running the **ONNX Runtime** model or generating/running the **TensorRT engine**, refer to **INSTALLATION.md**. It includes the required CUDA, cuDNN, TensorRT, ONNX Runtime GPU setup, version compatibility, and solutions to common installation and runtime issues.



---

# Phase 2 – ONNX Runtime Inference

In this phase, the pretrained YOLO `.pt` model is exported to the **ONNX** format.

The notebook demonstrates:

- Exporting the PyTorch model to ONNX
- Loading the exported ONNX model
- Running inference using ONNX Runtime GPU
- Implementing preprocessing and postprocessing

---

# Phase 3 – TensorRT Inference

The final phase converts the ONNX model into a **TensorRT Engine** for optimized GPU inference.

## Generate the TensorRT Engine

After exporting the ONNX model, generate the TensorRT engine using the `trtexec` utility.

Open a terminal in Ubuntu, navigate to the **Detection_Models** directory, and ensure that the ONNX model filename matches the command below.

```bash
cd Detection_Model_Evaluation/Detection_Models

trtexec \
--onnx=yolo26n.onnx \
--minShapes=images:1x3x640x640 \
--optShapes=images:4x3x640x640 \
--maxShapes=images:16x3x640x640 \
--saveEngine=yolo26n.engine
```

### Command Explanation

| Argument | Description |
|----------|-------------|
| `--onnx` | Specifies the input ONNX model. |
| `--minShapes` | Minimum supported input shape (batch size = 1). |
| `--optShapes` | Optimal input shape used for engine optimization. |
| `--maxShapes` | Maximum supported input shape (batch size = 16). |
| `--saveEngine` | Specifies the filename of the generated TensorRT engine. |

> **Note**
>
> Execute this command from the directory containing the ONNX model, or provide the full path to the ONNX file.


The notebook covers:

- Building a TensorRT engine from the ONNX model
- Loading the `.engine` model
- Running inference using TensorRT
- Implementing preprocessing and postprocessing
- Comparing predictions with the `.pt` and `.onnx` models
- Benchmarking inference performance across all three model formats

---

# Objective

By the end of this notebook, you will understand:

- How YOLO object detection inference works internally.
- How to export a `.pt` model to ONNX.
- How to build a TensorRT engine from an ONNX model.
- The preprocessing and postprocessing pipeline for each model format.
- The inference workflow for PyTorch, ONNX Runtime, and TensorRT.