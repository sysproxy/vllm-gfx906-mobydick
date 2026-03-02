ARG BASE_PYTORCH_IMAGE="docker.io/mixa3607/pytorch-gfx906:v2.9.1-rocm-6.3.4"
ARG VLLM_REPO="https://github.com/ai-infos/vllm-gfx906-mobydick.git"
ARG VLLM_BRANCH="main"
ARG TRITON_REPO="https://github.com/ai-infos/triton-gfx906.git"
ARG TRITON_BRANCH="v3.5.1+gfx906"

############# Base image #############
FROM ${BASE_PYTORCH_IMAGE} AS rocm_base

# Keep runtime surface similar to ML-gfx906 image behavior.
RUN pip install amdsmi==$(cat /opt/ROCM_VERSION_FULL)

ENV PYTORCH_ROCM_ARCH=gfx906
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
ENV TOKENIZERS_PARALLELISM=false
ENV HIP_FORCE_DEV_KERNARG=1
ENV VLLM_TARGET_DEVICE=rocm

############# Build base #############
FROM rocm_base AS build_base
RUN pip3 install "cmake<4" ninja wheel pybind11 "setuptools>=77.0.3,<80.0.0" setuptools_scm jinja2 packaging

############# Build triton #############
FROM build_base AS build_triton
ARG TRITON_REPO
ARG TRITON_BRANCH
WORKDIR /app
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 --branch ${TRITON_BRANCH} ${TRITON_REPO} triton
WORKDIR /app/triton
# Handle layout differences between triton branches.
RUN if [ ! -f setup.py ]; then cd python; fi; python3 setup.py bdist_wheel --dist-dir=/dist
RUN ls -la /dist

############# Build vllm #############
FROM build_base AS build_vllm
ARG VLLM_REPO
ARG VLLM_BRANCH
WORKDIR /app
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm
WORKDIR /app/vllm
RUN pip install -r requirements/rocm.txt
RUN python3 setup.py bdist_wheel --dist-dir=/dist
RUN ls -la /dist

############# Install all #############
FROM rocm_base AS final
WORKDIR /app/vllm
RUN --mount=type=bind,from=build_vllm,src=/app/vllm/requirements,target=/app/vllm/requirements \
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist_vllm \
    --mount=type=bind,from=build_triton,src=/dist/,target=/dist_triton \
    pip install /dist_triton/*.whl /dist_vllm/*.whl && \
    pip install -r requirements/rocm.txt && \
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-semantic-conventions-ai opentelemetry-exporter-otlp && \
    pip install modelscope && \
    true

CMD ["/bin/bash"]
