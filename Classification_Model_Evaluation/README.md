# Classification_Model_Evaluation.ipynb

This notebook demonstrates the complete inference pipeline for YOLO Image Classification models, including model conversion from PyTorch (.pt) to ONNX (.onnx) and TensorRT (.engine), followed by inference and performance benchmarking across all three model formats.

1. **Ultralytics YOLO (.pt)**
2. **ONNX Runtime (.onnx)**
3. **TensorRT (.engine)**

The notebook is divided into three phases to illustrate the complete model inference pipeline while maintaining consistent prediction outputs across all model formats.

> **Note**
>
> YOLO classification models expect an input image size of **224 × 224**. During preprocessing, every input image is resized to **224 × 224** to ensure compatibility with the model and to obtain accurate predictions.
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
Open Classification_Model_Evaluation.ipynb in jupyter Notebook 

Look notebook 1st  and then run IMPORT REQUIREMENTS cell 

# Phase 1 – Ultralytics YOLO (.pt) Inference

In this phase, inference is performed using the original pretrained **Ultralytics YOLO Classification** model.

The notebook covers:

- Loading a pretrained YOLO classification model
- Image preprocessing
- Running inference on batch images
- Postprocessing prediction results



## Downloading the Model

Ultralytics automatically downloads the model if it is not already available in your system.
Just run below code which is already present in you notebook just execute that cell

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
```
After executing above code will see that model in your working directory just  manually place the downloaded model inside:

```text
Classification_Model_Evaluation/
└── Classification_Models/
    └── yolo26n-cls.pt
```

If you use a different model, update the model path/name in the notebook accordingly .




---
## Sample Input Images

Sample images used in this repository can be downloaded from the following Google Drive folder:

🔗 **[Google Drive – Sample Input Images](https://drive.google.com/drive/folders/1MuNF6ytZHTcroBpnlwIUoWkaxl0oTTOS)**

You may also use your own images for inference.

---

## ImageNet Labels

YOLO classification models are trained on the **ImageNet** dataset. During inference, 
Ultralytics automatically maps the predicted class index to its corresponding ImageNet class name using the provided dataset configuration.
So for 1st phase that not require but for 2nd and 3rd phase that requires to you 


The official ImageNet dataset configuration is available here:

**[ImageNet.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)**


This YAML file contains all ImageNet class labels used by Ultralytics for classification models.

Download it and move into correct directory.

The ImageNet configuration file is locate in:

```text
Classification_Model_Evaluation/
└── cls-yaml/
    └── ImageNet.yaml
```



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

Open a terminal in Ubuntu, navigate to the **Classification_Models** directory, and ensure that the ONNX model filename matches the command below.

```bash
cd Classification_Model_Evaluation/'Classification _Models'

trtexec \
--onnx=yolo26n-cls.onnx \
--minShapes=images:1x3x224x224 \
--optShapes=images:4x3x224x224 \
--maxShapes=images:16x3x224x224 \
--saveEngine=yolo26n-cls.engine
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

- How YOLO classification inference works internally.
- How to export a `.pt` model to ONNX.
- How to build a TensorRT engine from an ONNX model.
- The preprocessing and postprocessing pipeline for each model format.
- The inference workflow for PyTorch, ONNX Runtime, and TensorRT.