# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH

# --- system deps ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl git ca-certificates build-essential ffmpeg sox libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# --- Miniconda --------------------------------------------------------------
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o mc.sh && \
    bash mc.sh -b -p /opt/conda && rm mc.sh

SHELL ["/bin/bash", "-c"]

# --- copy code --------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace
COPY RUN_APP_docker.sh /workspace/

# --- build the project's conda env -----------------------------------------
RUN bash install_direct.sh

# --- expose UI --------------------------------------------------------------
EXPOSE 7860

# --- entrypoint -------------------------------------------------------------
ENV CI_DOCKER_RUN=1
CMD conda run -p /workspace/tts_env python launch.py --listen 0.0.0.0

