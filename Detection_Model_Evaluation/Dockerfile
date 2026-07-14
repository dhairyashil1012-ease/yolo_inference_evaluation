# ==========================================================
# Base Image
# ==========================================================

FROM nvcr.io/nvidia/tensorrt:26.06-py3

# ==========================================================
# Environment Variables
# ==========================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ==========================================================
# Working Directory
# ==========================================================

WORKDIR /workspace

# ==========================================================
# Install System Dependencies
# ==========================================================

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

# ==========================================================
# Upgrade pip
# ==========================================================

RUN python3 -m pip install --upgrade pip

# ==========================================================
# Copy Requirements
# ==========================================================

COPY requirements.txt .

# ==========================================================
# Install Python Dependencies
# ==========================================================

RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# Copy Project Files
# ==========================================================

COPY . .

# # ==========================================================
# # Default Command
# # ==========================================================



ENTRYPOINT ["python3", "main.py", "-m"]


CMD ["pt"]
# docker run -it     --gpus all     -p 8888:8888     -v `pwd`:/workspace     -w /workspace     yolo_inference_evaluation:latest onnx -m pt