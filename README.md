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



# 1. Classification_Model_Evaluation.ipynb


# Phase 1 :- YOLO .pt ultralytics

```Following Information is only to understand you what is the actully thing is .You can directly go into notebook as well if you have already understanding of these things```

---
# Download the Classification Model

Go To your notes book and start execution of cells

Ultralytics provides pretrained classification models that can be downloaded directly.
Open Our Notebook and all above command with this following command.
You can change model according to your requirement.
Example:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
```

The model will automatically be downloaded if it is not already available.

You can also manually move the downloaded model into the `Models/Classification_Models` directory.

Example:

```text
Project/
└── Models/
      └──Classification_Models/
                    yolo26n.pt
```

---

# Verify Model Name

After downloading the model, ensure that the model filename matches the filename used inside the notebook.

Example:

```python
model = YOLO("Models/Classification _Models/yolo26n-cls.pt")
```

If your filename is different, update the notebook accordingly.

---


# ImageNet Labels

Ultralytics automatically maps prediction indices to ImageNet class names.

The ImageNet label configuration is available here:

https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml

This YAML file contains all ImageNet class labels used by Ultralytics for classification models.


---
# Image Requirements
### For you reference you can take my images as well or you can use yours as well  

```https://drive.google.com/drive/folders/1MuNF6ytZHTcroBpnlwIUoWkaxl0oTTOS?usp=sharing```

After running above command next you will see accessing the Image Folder and Model Folder . Check Its contains correct folder structure present in you project directory.

Otherwise change It.

For you kind information:- 

Ultralytics classification models are trained with an input image size of **224 × 224**.

Although the original image can have any resolution, it will be resized to **224 × 224** during preprocessing before being passed to the model.

Place all images inside:

```text
images/
```

---
# Important Note

Ultralytics already provides built-in implementations for:

- Preprocessing
- Inference
- Postprocessing

Therefore, users normally only need to execute:

```python
results = model(images)
```

However, this notebook separates these steps to demonstrate the complete inference pipeline and help users understand how each stage works internally.




--- 
# Phase 2 :- ONNX model

# ONNX Runtime GPU Installation Guide
``` pip install onnxruntime-gpu==1.19.2 ```

Follow the steps below to correctly configure **ONNX Runtime GPU** for running inference on an NVIDIA GPU. Since ONNX Runtime, CUDA Toolkit, and cuDNN have version dependencies, it is important to install them in the correct order.

---

## Step 1: Check the CUDA Version Installed on Your System

Before installing ONNX Runtime, determine which CUDA Toolkit version is currently installed.

```bash
nvcc --version
```

Example output:

```text
Cuda compilation tools, release 12.8, V12.8.93
```

Make a note of your CUDA version.

> **Note:** If `nvcc` is not found, CUDA is either not installed or its environment variables are not configured.

---

## Step 2: Find the Compatible ONNX Runtime Version

ONNX Runtime GPU supports only specific CUDA Toolkit versions. Do **not** assume that the latest ONNX Runtime release will work with your CUDA installation.

Visit the official ONNX Runtime CUDA compatibility table:

https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

Locate your installed CUDA version and identify the compatible **onnxruntime-gpu** release.

For example:

- CUDA 12.8 → Install the ONNX Runtime version recommended in the compatibility table.
- CUDA 12.6 → Install the corresponding supported ONNX Runtime version.
- CUDA 11.8 → Install the matching ONNX Runtime version.

Always follow the compatibility table rather than selecting the latest package.

---

## Step 3: Install the Compatible ONNX Runtime GPU Package

Once you have identified the correct version, install it.

Example:

```bash
pip install onnxruntime-gpu==1.19.2 \
--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
--break-system-packages
```

Replace `1.19.2` with the version that matches your CUDA Toolkit.

---

## Step 4: Verify the Installed ONNX Runtime Version

After installation, verify that the expected version has been installed.

```bash
pip show onnxruntime-gpu
```

or

```python
import onnxruntime as ort

print(ort.__version__)
```

---

## Step 5: What If Your CUDA Version Is Not Compatible?

Sometimes your currently installed CUDA Toolkit is **not supported** by the ONNX Runtime version you need.

In that case:

1. Uninstall the existing CUDA Toolkit.

```bash
sudo apt-get purge nvidia-cuda* cuda-*
sudo apt-get autoremove
```

2. Download the compatible CUDA Toolkit from the NVIDIA CUDA Toolkit Archive.

https://developer.nvidia.com/cuda-toolkit-archive

3. Install the CUDA version recommended by the ONNX Runtime compatibility table.

4. Verify the installation.

```bash
nvcc --version
```

> **Important:** It is generally easier to install the ONNX Runtime version that supports your existing CUDA installation. Only reinstall CUDA when there is no suitable ONNX Runtime version for your environment or your project requires a specific version.

---

## Step 6: Configure CUDA Environment Variables (If Required)

If the `nvcc` command is not detected after installing CUDA, configure the CUDA environment variables.

Example (CUDA 12.8):

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Verify again:

```bash
nvcc --version
```

---

## Step 7: Verify CUDA Execution Provider

Check whether ONNX Runtime can detect the CUDA Execution Provider.

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

Expected output:

```python
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If `CUDAExecutionProvider` appears in the list, your GPU environment has been configured successfully.

If only the following is displayed:

```python
['CPUExecutionProvider']
```

continue to the next step.

---

## Step 8: Install or Verify cuDNN

ONNX Runtime also depends on **cuDNN**. If the required cuDNN libraries are missing, you may encounter an error similar to:

```text
Failed to load library libonnxruntime_providers_cuda.so

libcudnn_adv.so.9:
cannot open shared object file:
No such file or directory
```

This indicates that the required cuDNN libraries are missing or incompatible with your CUDA Toolkit.

Install the cuDNN version that matches your installed CUDA Toolkit. The required version is listed in the ONNX Runtime compatibility documentation.

After installing cuDNN, check the available execution providers again:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

---

## Step 9: If CUDAExecutionProvider Is Still Not Available

If `CUDAExecutionProvider` is still missing after installing the correct CUDA Toolkit and cuDNN:

1. Verify that your CUDA, cuDNN, and ONNX Runtime versions are all compatible.
2. Ensure the CUDA environment variables are configured correctly.
3. Reinstall the compatible ONNX Runtime GPU package.

Example:

```bash
pip uninstall onnxruntime onnxruntime-gpu
```

Then reinstall the version recommended by the ONNX Runtime compatibility table:

```bash
pip install onnxruntime-gpu==<compatible_version> \
--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
--break-system-packages
```

Finally, verify the execution providers again.

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

When the output includes:

```python
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

your ONNX Runtime installation is correctly configured for GPU inference.

---

> **Note:** This guide was prepared using an Ubuntu Linux system. Some installation commands and environment variable paths may differ for Windows or other operating systems.

## Additional Notes for Phase 2

### Case A: `nvcc` is Available

Run:

```bash
nvcc --version
```

If the command returns your CUDA version, you can proceed to the next step.

Example:

```text
Cuda compilation tools, release 12.8, V12.8.93
```

---

### Case B: `nvcc` Command Not Found

If running

```bash
nvcc --version
```

returns

```text
command not found
```

then CUDA is either not installed correctly or its binary directory has not been added to your system's `PATH`.

#### Step 1: Locate the CUDA Installation

Run the following commands:

```bash
which nvcc
```

```bash
find /usr -name nvcc 2>/dev/null
```

```bash
ls /usr/local/
```

```bash
echo $PATH
```

If CUDA is installed, you may see an output similar to:

```text
/usr/local/cuda-13.3/bin/nvcc
```

---

#### Step 2: Configure CUDA Environment Variables

##### Temporary Configuration (Current Terminal Only)

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
```

> **Note:** Replace `cuda-13.3` with the CUDA version installed on your system (for example, `cuda-12.8`, `cuda-12.6`, `cuda-11.8`, etc.).

---

##### Permanent Configuration

Open your `.bashrc` file.

```bash
nano ~/.bashrc
```

Add the following lines at the end of the file:

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
```

> **Important:** Replace `cuda-13.3` with your installed CUDA version.

Save the file and reload the environment:

```bash
source ~/.bashrc
```

Verify the installation:

```bash
nvcc --version
```

If the CUDA version is displayed, the environment variables have been configured successfully.

---

# Phase 3 – TensorRT Engine Generation

## Install TensorRT

Download and install TensorRT according to your:

- Operating System
- CUDA Toolkit Version
- CPU Architecture

Download TensorRT from the official NVIDIA website:

https://developer.nvidia.com/tensorrt/download/11x

> **Note:** Always install the TensorRT version that is compatible with your installed CUDA Toolkit.

> **Note:** Do following operation if Not Done
---
```sudo cp /var/nv-tensorrt-local-repo-*/**/*keyring.gpg /usr/share/keyrings/ 2>/dev/null || true``
```sudo apt update```
```sudo apt install -y tensorrt```

Install Python bindings:

``` sudo apt install -y python3-libnvinfer python3-libnvinfer-dev```
## Verify TensorRT Installation

After installation, verify that TensorRT has been installed correctly.

```bash
trtexec --help
```

or

```bash
trtexec --version
```

If the command is recognized, TensorRT has been installed successfully.

---

## Generate the TensorRT Engine
open CLI and GO inside your Models dir and enter correct onnx model name 
Run the following command to convert the ONNX model into a TensorRT engine.

```bash
trtexec \
--onnx=yolo26n-cls.onnx \
--minShapes=images:1x3x224x224 \
--optShapes=images:4x3x224x224 \
--maxShapes=images:16x3x224x224 \
--saveEngine=yolo26n-cls.engine
```

### Explanation

- `--onnx` : Specifies the input ONNX model.
- `--minShapes` : Minimum supported batch size.
- `--optShapes` : Optimal batch size for best performance.
- `--maxShapes` : Maximum supported batch size.
- `--saveEngine` : Name of the generated TensorRT engine.

> **Note:** Execute this command in the directory where your ONNX model is located, or provide the full path to the ONNX model.

---





# 2. Detection_Model_Evaluation.ipynb


*** NOTE :- Evaluates object detection models.

### What changes in this notebook?
- Uses models from `Models/Detection_Models/`
- Uses YAML files from `yaml_files/detection-yaml/`
- Produces bounding boxes, confidence scores, and detection metrics.

---
# Phase 1 :- YOLO .pt ultralytics

```Following Information is only to understand you what is the actully thing is .You can directly go into notebook as well if you have already understanding of these things```

---
# Download the Classification Model

Go To your notes book and start execution of cells

Ultralytics provides pretrained classification models that can be downloaded directly.
Open Our Notebook and all above command with this following command.
You can change model according to your requirement.
Example:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```

The model will automatically be downloaded if it is not already available.

You can also manually move the downloaded model into the `Models/Detection_Models` directory.

Example:

```text
Project/
└── Models/
         └── Detection_Models/
                         yolo26n.pt
```

---

# Verify Model Name

After downloading the model, ensure that the model filename matches the filename used inside the notebook.

Example:

```python
model = YOLO("Models/Detection_Modelss/yolo26n.pt")
```

If your filename is different, update the notebook accordingly.

---


# COCO Labels

Ultralytics automatically maps prediction indices to ImageNet class names.

The COCO label configuration is available here:

https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

This YAML file contains all COCO class labels used by Ultralytics for classification models.

---
> **Note**
> The setup process for **ONNX Runtime** and **TensorRT** models is common across all notebooks. Please refer to the **Classification_Model_Evaluation.ipynb** notebook under the **ONNX** and **TENSORRT ENGINE** section for detailed instructions before running the Detection or Semantic Segmentation notebooks.
---

# Phase 3 – TensorRT Engine Generation


## Generate the TensorRT Engine
open CLI and GO inside your Models dir and enter correct onnx model name 
Run the following command to convert the ONNX model into a TensorRT engine.

```bash
trtexec \
--onnx=yolo26n-cls.onnx \
--minShapes=images:1x3x640x640\
--optShapes=images:4x3x640x640 \
--maxShapes=images:16x3x640x640 \
--saveEngine=yolo26n.engine
```

### Explanation

- `--onnx` : Specifies the input ONNX model.
- `--minShapes` : Minimum supported batch size.
- `--optShapes` : Optimal batch size for best performance.
- `--maxShapes` : Maximum supported batch size.
- `--saveEngine` : Name of the generated TensorRT engine.

> **Note:** Execute this command in the directory where your ONNX model is located, or provide the full path to the ONNX model.

---
# Supported Model Formats

This project demonstrates inference for multiple deployment formats.

- PyTorch (`.pt`)
- ONNX (`.onnx`)
- TensorRT Engine (`.engine`)

Each notebook is organized into the same logical stages:

- Preprocessing
- Inference
- Postprocessing

The implementation of each stage may differ depending on the model format, but the overall inference pipeline remains the same.

---

# Running the Notebook

1. Install all dependencies.
2. Install Ultralytics.
3. Download the required model.
4. Place the model inside the `models/` directory.
5. Place input images inside the `images/` directory.
6. Open the notebook.
7. Execute each stage sequentially:
   - Preprocessing
   - Inference
   - Postprocessing

---

# Summary

This notebook is intended to help users understand how a YOLO Classification model performs inference internally.

Instead of relying entirely on Ultralytics' built-in pipeline, each stage is implemented separately to provide a clear understanding of:

- Image preprocessing
- Batch creation
- Model inference
- Prediction decoding

The same workflow can later be extended to ONNX and TensorRT models while maintaining a similar inference pipeline.