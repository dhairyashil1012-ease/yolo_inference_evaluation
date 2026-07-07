# Installation Guide

This guide describes the installation and configuration of the GPU software stack required to run the notebooks in this repository. It covers the setup of **CUDA Toolkit**, **cuDNN**, **TensorRT**, and **ONNX Runtime GPU**, along with the version compatibility between these components.

Python package installation is **not** covered in this guide, as all required Python dependencies are already listed in the project's `requirements.txt` file and can be installed by following the instructions in the main `README.md`.

> **Note**
>
> The notebooks in this repository have been developed and tested on **Ubuntu** with an **NVIDIA GPU**. To avoid compatibility issues, follow the installation steps in the order presented and ensure that the installed versions of CUDA, cuDNN, TensorRT, and ONNX Runtime GPU are compatible with each other.



# ONNX Runtime GPU Installation

This section explains how to install **ONNX Runtime GPU** with the correct **CUDA Toolkit** version.

ONNX Runtime GPU has strict compatibility requirements with **CUDA Toolkit** and **cuDNN**. Before installing `onnxruntime-gpu`, always verify your CUDA version and install the compatible ONNX Runtime release.

---

## Step 1: Check the Installed CUDA Version

Before installing ONNX Runtime GPU, determine which CUDA Toolkit version is installed on your system.

Run:

```bash
nvcc --version
```

Example output:

```text
Cuda compilation tools, release 12.8, V12.8.93
```

Make a note of the installed CUDA version.

> **Note**
>
> If the `nvcc` command is not found, CUDA Toolkit is either not installed or its environment variables have not been configured correctly.

---

## Step 2: Find the Compatible ONNX Runtime Version

ONNX Runtime GPU supports only specific CUDA Toolkit versions. **Do not assume that the latest ONNX Runtime release is compatible with your installed CUDA version.**

Refer to the official ONNX Runtime compatibility table:

```[https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)```

Locate your installed CUDA Toolkit version and identify the corresponding supported **onnxruntime-gpu** release.

For example:

* CUDA 12.8 → Install the ONNX Runtime version recommended in the compatibility table.
* CUDA 12.6 → Install the corresponding supported ONNX Runtime version.
* CUDA 11.8 → Install the matching supported ONNX Runtime version.

> **Recommendation**
>
> Always install the ONNX Runtime version recommended by the official compatibility table instead of simply choosing the latest available release.

---

## Step 3: Install ONNX Runtime GPU

Once you have identified the correct version, install it using `pip`.

Example:

```bash
pip install onnxruntime-gpu==1.19.2 \
--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
--break-system-packages
```

Replace **`1.19.2`** with the version compatible with your installed CUDA Toolkit and change also sencond like with your cuda version . But problem is this version requires matching cuda version as well.

---

## Step 4: Verify the Installation

After installation, verify that ONNX Runtime GPU has been installed successfully.

Using pip:

```bash
pip show onnxruntime-gpu
```

Or in Python:

```python
import onnxruntime as ort

print(ort.__version__)
```

Ensure that the installed version matches the version you intended to install.

---

## Step 5: If Your CUDA Version Is Not Compatible

In some cases, the installed CUDA Toolkit version may not be supported by the ONNX Runtime version required for your project.

If this happens:

### 1. Remove the Existing CUDA Toolkit

```bash
sudo apt-get purge nvidia-cuda* cuda-*
sudo apt-get autoremove
```

### 2. Download a Compatible CUDA Toolkit

Download the required CUDA Toolkit version from the NVIDIA CUDA Toolkit Archive:

```[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)```

### 3. Install the Compatible CUDA Version

Install the CUDA Toolkit version recommended by the ONNX Runtime compatibility table.

### 4. Verify the Installation

```bash
nvcc --version
```

Confirm that the installed CUDA version matches the version you intended to install.

---

> **Important**
>
> In most cases, it is easier to install an **ONNX Runtime GPU** version that is compatible with your existing CUDA Toolkit. Reinstall CUDA only when your current CUDA version is unsupported or when your project explicitly requires a different CUDA Toolkit version.

---

## Step 6: Configure CUDA Environment Variables (If Required)

If the `nvcc` command is not detected after installing the CUDA Toolkit, configure the CUDA environment variables manually.

For example, if you have installed **CUDA 12.8**, run:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

After configuring the environment variables, verify the installation again:

```bash
nvcc --version
```

> **Note**
>
> Replace `cuda-12.8` with the version of CUDA installed on your system.

---

## Step 7: Verify the CUDA Execution Provider

After installing ONNX Runtime GPU, verify that it can detect the **CUDA Execution Provider**.

Run the following Python code:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

If the installation is successful, the output should be similar to:

```python
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If **`CUDAExecutionProvider`** appears in the list, ONNX Runtime is correctly configured to use your NVIDIA GPU.

If the output is only:

```python
['CPUExecutionProvider']
```

continue to the next step.

---

## Step 8: Install or Verify cuDNN

ONNX Runtime GPU also depends on **cuDNN**. If the required cuDNN libraries are missing or incompatible, you may encounter an error similar to:

```text
Failed to load library libonnxruntime_providers_cuda.so

libcudnn_adv.so.9:
cannot open shared object file:
No such file or directory
```

This error indicates that the required cuDNN libraries are either missing or incompatible with the installed CUDA Toolkit.

Install the cuDNN version that matches your installed CUDA Toolkit. The required version can be found in the official ONNX Runtime compatibility documentation.

After installing cuDNN, verify the available execution providers again:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

---

## Step 9: If `CUDAExecutionProvider` Is Still Not Available

If `CUDAExecutionProvider` is still unavailable after installing the correct CUDA Toolkit and cuDNN, perform the following checks:

1. Verify that the installed versions of **CUDA Toolkit**, **cuDNN**, and **ONNX Runtime GPU** are mutually compatible.
2. Ensure that the CUDA environment variables are configured correctly.
3. Reinstall the compatible ONNX Runtime GPU package.

First, uninstall any existing ONNX Runtime packages:

```bash
pip uninstall onnxruntime onnxruntime-gpu
```

Then install the version recommended by the official ONNX Runtime compatibility table:

```bash
pip install onnxruntime-gpu==<compatible_version> \
--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
--break-system-packages
```

Finally, verify the available execution providers again:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

If the output contains:

```python
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

your ONNX Runtime GPU installation has been configured successfully and is ready for GPU inference.

---

> **Note**
>
> This guide has been prepared and tested on **Ubuntu Linux**. Some installation commands, package names, and environment variable paths may differ on **Windows** or other operating systems.



## Additional Notes

The following steps can help if the `nvcc` command is not detected or CUDA is not configured correctly on your system.

---

### Case A: `nvcc` Is Available

Verify that the CUDA Toolkit is installed by running:

```bash
nvcc --version
```

If the command displays your installed CUDA version, you can proceed with the remaining installation steps.

Example output:

```text
Cuda compilation tools, release 12.8, V12.8.93
```

---

### Case B: `nvcc` Command Not Found

If running

```bash
nvcc --version
```

returns:

```text
command not found
```

it usually indicates one of the following:

* CUDA Toolkit is not installed.
* CUDA is installed, but its binary directory has not been added to the system `PATH`.
* The CUDA installation path is different from the default location.

---

#### Step 1: Locate the CUDA Installation

Run the following commands to determine whether CUDA is installed and where it is located.

Locate the `nvcc` executable:

```bash
which nvcc
```


Search the system for the `nvcc` binary:

```bash
find /usr -name nvcc 2>/dev/null
```

List the contents of the CUDA installation directory:

```bash
ls /usr/local/
```

Display the current system `PATH`:

```bash
echo $PATH
```

If CUDA is installed successfully, you may see an output similar to:

```text
/usr/local/cuda-13.3/bin/nvcc
```
---

#### Step 2: Configure CUDA Environment Variables

If CUDA is installed but `nvcc` is still not detected, configure the required environment variables.

##### Temporary Configuration (Current Terminal Session)

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
```

> **Note**
>
> Replace `cuda-13.3` with the version installed on your system (for example, `cuda-12.8`, `cuda-12.6`, or `cuda-11.8`).

Verify the configuration:

```bash "
nvcc --version
```

---

##### Permanent Configuration

To make the environment variables persistent across terminal sessions, add them to your shell configuration.

Open the `.bashrc` file:

```bash
nano ~/.bashrc
```

Append the following lines to the end of the file:

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
```

> **Important**
>
> Replace `cuda-13.3` with the CUDA version installed on your system.

Save the file, then reload the shell configuration:

```bash
source ~/.bashrc
```

Finally, verify that CUDA is correctly configured:

```bash
nvcc --version
```

If the installed CUDA version is displayed, the environment variables have been configured successfully.



# TensorRT Installation

TensorRT is required to convert exported **ONNX** models into optimized **TensorRT (`.engine`)** models for high-performance GPU inference.

---

## Step 1: Download TensorRT

Download the TensorRT package that matches your:

* Operating System
* CUDA Toolkit Version

Official NVIDIA Download Page:

```[https://developer.nvidia.com/tensorrt/download/11x](https://developer.nvidia.com/tensorrt/download/11x)```

> **Note**
>
> Always install the TensorRT version that is compatible with your installed CUDA Toolkit.

---

## Step 2: Install TensorRT

If you installed TensorRT using the local NVIDIA repository package (`.deb`), execute the following commands:

```bash
sudo cp /var/nv-tensorrt-local-repo-*/*keyring.gpg /usr/share/keyrings/ 2>/dev/null || true

sudo apt update

sudo apt install -y tensorrt
```

Next, install the TensorRT Python bindings:

```bash
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev
```

---

## Step 3: Verify the Installation

After installation, verify that TensorRT is available.

```bash
trtexec --help
```

or

```bash
trtexec --version
```

If either command executes successfully, TensorRT has been installed correctly.

---

## TensorRT Engine Generation

After installing TensorRT, you can convert an exported **ONNX** model into a **TensorRT (`.engine`)** model using the `trtexec` utility.

The exact `trtexec` command depends on:

* The model type (Classification, Object Detection, or Semantic Segmentation)
* The input image size
* The ONNX model filename
* The desired batch sizes

> **Note**
>
> This installation guide only covers the TensorRT installation process. The exact `trtexec` commands for generating TensorRT engine files are provided in the corresponding notebook documentation:
>
> * `Classification_Model_Evaluation/README.md`
> * `Detection_Model_Evaluation/README.md`
> * `Semantic_Segmentation_Model_Evaluation/README.md`
>
> Follow the instructions in the appropriate README to generate the `.engine` file for your model before executing the TensorRT inference phase in the notebook.
