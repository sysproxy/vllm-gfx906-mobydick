# ============================================================
# Configurable versions
# ============================================================
ARG BASE_ROCM_IMAGE="rocm/dev-ubuntu-24.04:6.3.4-complete"
ARG ROCM_ARCH="gfx906"

ARG ROCBLAS_REPO="https://github.com/ROCm/rocBLAS"
ARG TENSILE_REPO="https://github.com/ROCm/Tensile"
ARG RCCL_REPO="https://github.com/ROCm/rccl"

ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
ARG PYTORCH_BRANCH="v2.10.0"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG PYTORCH_VISION_BRANCH="v0.25.0"
ARG PYTORCH_AUDIO_REPO="https://github.com/pytorch/audio.git"
ARG PYTORCH_AUDIO_BRANCH="v2.10.0"

ARG TRITON_REPO="https://github.com/ai-infos/triton-gfx906.git"
ARG TRITON_BRANCH="v3.6.0+gfx906"

ARG VLLM_REPO="https://github.com/ai-infos/vllm-gfx906-mobydick.git"
ARG VLLM_BRANCH="main"

ARG MAX_JOBS=""

# ============================================================
# rocm_base: detect ROCm version, install Python 3.12
# ============================================================
FROM ${BASE_ROCM_IMAGE} AS rocm_base

RUN ROCM_VERSION_MAJOR=$(ls /opt/ | sed -nE 's|rocm-([0-9]+)\.([0-9]+)\.([0-9]+)|\1|1p') && \
    ROCM_VERSION_MINOR=$(ls /opt/ | sed -nE 's|rocm-([0-9]+)\.([0-9]+)\.([0-9]+)|\2|1p') && \
    ROCM_VERSION_PATCH=$(ls /opt/ | sed -nE 's|rocm-([0-9]+)\.([0-9]+)\.([0-9]+)|\3|1p') && \
    echo "$ROCM_VERSION_MAJOR" > /opt/ROCM_VERSION_MAJOR && \
    echo "$ROCM_VERSION_MINOR" > /opt/ROCM_VERSION_MINOR && \
    echo "$ROCM_VERSION_PATCH" > /opt/ROCM_VERSION_PATCH && \
    echo "$ROCM_VERSION_MAJOR.$ROCM_VERSION_MINOR" > /opt/ROCM_VERSION && \
    echo "$ROCM_VERSION_MAJOR.$ROCM_VERSION_MINOR.$ROCM_VERSION_PATCH" > /opt/ROCM_VERSION_FULL && \
    echo "Detected ROCm version is $(cat /opt/ROCM_VERSION_FULL)"

RUN apt-get update && apt-get install -y software-properties-common git python3-pip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv \
    python3.12-lib2to3 python-is-python3 python3.12-full && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    ln -sf /usr/bin/python3.12-config /usr/bin/python3-config && \
    python3 -m pip config set global.break-system-packages true && \
    pip install amdsmi==$(cat /opt/ROCM_VERSION_FULL) && \
    true

ARG ROCM_ARCH
ENV ROCM_ARCH=${ROCM_ARCH}
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}
ENV PATH=/opt/rocm/llvm/bin:$PATH
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
ENV TOKENIZERS_PARALLELISM=false
ENV HIP_FORCE_DEV_KERNARG=1
ENV VLLM_TARGET_DEVICE=rocm

# ============================================================
# build_base: common build dependencies
# ============================================================
FROM rocm_base AS build_base
RUN apt-get update && apt-get install -y git cmake libfmt-dev

# ============================================================
# build_rocblas: rebuild rocBLAS for target architecture
# ============================================================
FROM build_base AS build_rocblas
ARG ROCBLAS_REPO
ARG TENSILE_REPO
ARG ROCM_ARCH

WORKDIR /rebuild-deps
RUN git clone --depth 1 --branch rocm-$(cat /opt/ROCM_VERSION_FULL) ${ROCBLAS_REPO} rocBLAS && \
    git clone --depth 1 --branch rocm-$(cat /opt/ROCM_VERSION_FULL) ${TENSILE_REPO} Tensile

WORKDIR /rebuild-deps/rocBLAS
ENV PACKAGE_NAME=rocblas
RUN dpkg -s ${PACKAGE_NAME}
RUN ./install.sh --dependencies --rmake_invoked
RUN export INSTALLED_PACKAGE_VERSION=$(dpkg -s ${PACKAGE_NAME} | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    echo "Installed package version is \"$INSTALLED_PACKAGE_VERSION\"" && \
    export ROCM_LIBPATCH_VERSION=$(echo "$INSTALLED_PACKAGE_VERSION" | sed -E 's|^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)-(.*)|\4|1') && \
    echo "Set ROCM_LIBPATCH_VERSION to \"$ROCM_LIBPATCH_VERSION\"" && \
    export CPACK_DEBIAN_PACKAGE_RELEASE=$(echo "$INSTALLED_PACKAGE_VERSION" | sed -E 's|^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)-(.*)|\5|1') && \
    echo "Set CPACK_DEBIAN_PACKAGE_RELEASE to \"$CPACK_DEBIAN_PACKAGE_RELEASE\"" && \
    python3 ./rmake.py \
      --install_invoked \
      --build_dir=$(realpath ./build) \
      --src_path=$(realpath .) \
      --architecture ${ROCM_ARCH} \
      --test_local_path=$(realpath ../Tensile) && \
    cd ./build/release && \
    make package && \
    mkdir -p /dist && cp *.deb /dist
RUN export INSTALLED_PACKAGE_VERSION=$(dpkg -s ${PACKAGE_NAME} | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    export BUILDED_PACKAGE_VERSION=$(dpkg -I /dist/${PACKAGE_NAME}_*.deb | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    if [ "$BUILDED_PACKAGE_VERSION" != "$INSTALLED_PACKAGE_VERSION" ]; then \
      echo "ERR: Built version is $BUILDED_PACKAGE_VERSION but expected $INSTALLED_PACKAGE_VERSION"; exit 10; \
    fi

# ============================================================
# build_rccl: rebuild RCCL for target architecture
# ============================================================
FROM build_base AS build_rccl
ARG RCCL_REPO
ARG ROCM_ARCH

WORKDIR /rebuild-deps
RUN git clone --depth 1 --branch rocm-$(cat /opt/ROCM_VERSION_FULL) ${RCCL_REPO} rccl

WORKDIR /rebuild-deps/rccl
ENV PACKAGE_NAME=rccl
RUN dpkg -s ${PACKAGE_NAME}
RUN export INSTALLED_PACKAGE_VERSION=$(dpkg -s ${PACKAGE_NAME} | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    echo "Installed package version is \"$INSTALLED_PACKAGE_VERSION\"" && \
    export ROCM_LIBPATCH_VERSION=$(echo "$INSTALLED_PACKAGE_VERSION" | sed -E 's|^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)-(.*)|\4|1') && \
    echo "Set ROCM_LIBPATCH_VERSION to \"$ROCM_LIBPATCH_VERSION\"" && \
    export CPACK_DEBIAN_PACKAGE_RELEASE=$(echo "$INSTALLED_PACKAGE_VERSION" | sed -E 's|^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)-(.*)|\5|1') && \
    echo "Set CPACK_DEBIAN_PACKAGE_RELEASE to \"$CPACK_DEBIAN_PACKAGE_RELEASE\"" && \
    ./install.sh --package_build --amdgpu_targets ${ROCM_ARCH} && \
    mkdir -p /dist && cp ./build/release/*.deb /dist
RUN export INSTALLED_PACKAGE_VERSION=$(dpkg -s ${PACKAGE_NAME} | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    export BUILDED_PACKAGE_VERSION=$(dpkg -I /dist/${PACKAGE_NAME}_*.deb | sed -nE 's|^ *Version: (.+)$|\1|p') && \
    if [ "$BUILDED_PACKAGE_VERSION" != "$INSTALLED_PACKAGE_VERSION" ]; then \
      echo "ERR: Built version is $BUILDED_PACKAGE_VERSION but expected $INSTALLED_PACKAGE_VERSION"; exit 10; \
    fi

# ============================================================
# rocm_patched: install rebuilt rocBLAS + RCCL
# ============================================================
FROM rocm_base AS rocm_patched
RUN apt-get update && apt-get install -y libfmt-dev
RUN --mount=type=bind,from=build_rocblas,src=/dist/,target=/dist \
    dpkg -i /dist/*.deb
RUN --mount=type=bind,from=build_rccl,src=/dist/,target=/dist \
    dpkg -i /dist/*.deb
RUN apt-get install

# ============================================================
# build_torch: build PyTorch from source
# ============================================================
FROM rocm_patched AS build_torch
RUN pip install setuptools wheel packaging cmake ninja setuptools_scm jinja2 pybind11 mkl-static mkl-include

ARG PYTORCH_REPO
ARG PYTORCH_BRANCH
ARG MAX_JOBS

WORKDIR /build/pytorch
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 \
    --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO}" .
RUN pip install -r requirements.txt
RUN python3 tools/amd_build/build_amd.py

ENV USE_ROCM=1
RUN export MAX_JOBS="${MAX_JOBS:-$(nproc)}" && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import sys; print(sys.prefix)')" && \
    pip wheel --no-build-isolation -v -w /dist . 2>&1 | tee /tmp/torch_build.log
RUN pip install /dist/torch*.whl

# ============================================================
# build_vision: build torchvision
# ============================================================
FROM build_torch AS build_vision
RUN apt-get update && apt-get install -y libpng-dev libjpeg-dev ffmpeg

ARG PYTORCH_VISION_REPO
ARG PYTORCH_VISION_BRANCH

WORKDIR /build/vision
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 \
    --branch "${PYTORCH_VISION_BRANCH}" "${PYTORCH_VISION_REPO}" .
ENV FORCE_CUDA=1
ENV USE_ROCM=1
RUN python3 setup.py bdist_wheel --dist-dir=/dist_vision

# ============================================================
# build_audio: build torchaudio
# ============================================================
FROM build_torch AS build_audio

ARG PYTORCH_AUDIO_REPO
ARG PYTORCH_AUDIO_BRANCH

WORKDIR /build/audio
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 \
    --branch "${PYTORCH_AUDIO_BRANCH}" "${PYTORCH_AUDIO_REPO}" .
ENV USE_ROCM=1
RUN python3 setup.py bdist_wheel --dist-dir=/dist_audio

# ============================================================
# build_triton: build Triton with gfx906 support
# ============================================================
FROM rocm_patched AS build_triton
RUN pip install setuptools wheel packaging cmake ninja pybind11

ARG TRITON_REPO
ARG TRITON_BRANCH

WORKDIR /build/triton
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 \
    --branch "${TRITON_BRANCH}" "${TRITON_REPO}" .
RUN pip install -r python/requirements.txt
ENV TRITON_CODEGEN_BACKENDS="amd"
RUN pip wheel --no-build-isolation -w /dist . 2>&1 | tee /tmp/triton_build.log

# ============================================================
# build_vllm: build vLLM
# ============================================================
FROM build_torch AS build_vllm
RUN pip install "cmake<4" ninja wheel pybind11 "setuptools>=77.0.3,<80.0.0" setuptools_scm jinja2 packaging

ARG VLLM_REPO
ARG VLLM_BRANCH

WORKDIR /build/vllm
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 \
    --branch "${VLLM_BRANCH}" "${VLLM_REPO}" .
COPY use_existing_torch.py /tmp/use_existing_torch.py
RUN python3 /tmp/use_existing_torch.py --prefix
RUN pip install -r requirements/rocm.txt
RUN python3 setup.py bdist_wheel --dist-dir=/dist

# ============================================================
# final: assemble all components
# ============================================================
FROM rocm_patched AS final

WORKDIR /app/vllm
RUN --mount=type=bind,from=build_torch,src=/dist/,target=/dist_torch \
    --mount=type=bind,from=build_vision,src=/dist_vision/,target=/dist_vision \
    --mount=type=bind,from=build_audio,src=/dist_audio/,target=/dist_audio \
    --mount=type=bind,from=build_triton,src=/dist/,target=/dist_triton \
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist_vllm \
    --mount=type=bind,from=build_vllm,src=/build/vllm/requirements,target=/app/vllm/requirements \
    pip install /dist_torch/torch*.whl \
               /dist_vision/torchvision*.whl \
               /dist_audio/torchaudio*.whl \
               /dist_triton/triton*.whl \
               /dist_vllm/*.whl && \
    pip install -r requirements/rocm.txt && \
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-semantic-conventions-ai opentelemetry-exporter-otlp && \
    pip install modelscope && \
    true

CMD ["/bin/bash"]
