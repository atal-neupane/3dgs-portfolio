FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3 -m pip install --upgrade pip

RUN pip install \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip install \
    fastapi \
    uvicorn \
    plyfile \
    tqdm \
    opencv-python-headless \
    matplotlib \
    scipy \
    einops

RUN git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff-gaussian-rasterization \
    && cd /tmp/diff-gaussian-rasterization \
    && git checkout 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0

RUN pip install --no-build-isolation /tmp/diff-gaussian-rasterization

RUN git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn \
    && echo "from . import _C" > /tmp/simple-knn/simple_knn/__init__.py \
    && pip install --no-build-isolation /tmp/simple-knn

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
