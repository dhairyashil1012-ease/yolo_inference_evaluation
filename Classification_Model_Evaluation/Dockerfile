# FROM nvcr.io/nvidia/tensorrt:26.06-py3

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1

# # Default backend
# ENV MODEL_TYPE=pt

# WORKDIR /workspace

# RUN apt-get update && apt-get install -y \
#     git \
#     wget \
#     curl \
#     unzip \
#     vim \
#     nano \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     libxcb1 \
#     && rm -rf /var/lib/apt/lists/*

# RUN python3 -m pip install --upgrade pip

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# ENTRYPOINT ["python3", "main.py"]
# FROM nvcr.io/nvidia/tensorrt:26.04-py3

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1

# # Default backend
# ENV MODEL_TYPE=pt

# # Route ONNX Runtime to internal container CUDA & TensorRT binaries
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# WORKDIR /workspace

# RUN apt-get update && apt-get install -y \
#     git \
#     wget \
#     curl \
#     unzip \
#     vim \
#     nano \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     libxcb1 \
#     && rm -rf /var/lib/apt/lists/*

# RUN python3 -m pip install --upgrade pip

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# ENTRYPOINT ["python3", "main.py"]

# # docker run --gpus all -it your-built-image-name

FROM nvcr.io/nvidia/tensorrt:26.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Default backend
ENV MODEL_TYPE=pt

# Point directly to CUDA and internal TensorRT system paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /workspace

# Install core dependencies safely
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    vim \
    nano \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

# CRITICAL FOR TENSORRT BASE IMAGES: 
# Ensure your requirements.txt installs 'onnxruntime-gpu'.
# Tell ORT to hook into the container's preinstalled TensorRT and CUDA instead of fetching heavy wheels.
ENV ORT_TENSORRT_VERSION=10.0
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
