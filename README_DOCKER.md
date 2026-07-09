# YOLO Inference Evaluation Setup Guide

Follow the steps below to set up the environment and run the evaluation notebooks.

## Step 1: Clone the Repository

Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/dhairyashil1012-ease/yolo_inference_evaluation

cd yolo_inference_evaluation
```

---

## Step 2: Download the TensorRT Docker Image

TensorRT Docker images are available from the NVIDIA NGC Catalog:

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags

Pull the required TensorRT image (replace the tag with the version you want to use).

```bash
docker pull nvcr.io/nvidia/tensorrt:26.06-py3
```

---

## Step 3: (Optional) Download Sample Input Images

You can use your own images for evaluation or download the sample images from the following Google Drive folder:

https://drive.google.com/drive/folders/1MuNF6ytZHTcroBpnlwIUoWkaxl0oTTOS

After downloading, extract the images into the appropriate evaluation directory corresponding to your use case.

---

## Step 4: Launch the TensorRT Docker Container

Mount the project directory inside the container and start an interactive TensorRT environment.

```bash
docker run -it \
    --gpus all \
    -p 8888:8888 \
    -v `pwd`:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:26.06-py3 \
    bash
```

---

## Step 5: Install Python Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

---

## Step 6: Install System Dependencies

Install the required system libraries.

```bash
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1
```

---

## Step 7: Start Jupyter Notebook

Launch the Jupyter Notebook server inside the Docker container.

```bash
jupyter notebook \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root
```

Copy the generated URL from the terminal and open it in your web browser.

---

## Step 8: Run the Evaluation

Navigate to the notebook corresponding to your evaluation use case and execute it.

For additional details, configuration options, and usage instructions, refer to the **README.md** file inside each evaluation directory.
