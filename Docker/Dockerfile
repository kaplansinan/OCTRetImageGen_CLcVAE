# FROM tensorflow/tensorflow:2.3.1-gpu
#FROM fastdotai/fastai:2.0.8
FROM tensorflow/tensorflow:2.4.1-gpu-jupyter
#FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
ENV TZ=Europe/Helsinki
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# CUDA REPO KEY FETCH
RUN apt-get install wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    rm /etc/apt/sources.list.d/cuda.list && \ 
    rm /etc/apt/sources.list.d/nvidia-ml.list && \ 
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        python3-dev \
        python3-pip \ 
        python3-setuptools \
        cmake \
        swig \
        wget \
        unzip \
        screen \
        nano \
        lxc \
        iptables \
        ca-certificates \
        libfftw3-dev \
        expat \
        libgomp1 \
        graphviz
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
WORKDIR /tmp/
RUN pip install pip --upgrade
RUN pip install -r ./requirements.txt && pip install --upgrade opencv-python
WORKDIR /main/
# publish port
EXPOSE 61499