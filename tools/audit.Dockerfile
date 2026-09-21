FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential clang libclang-rt-dev cmake ninja-build git ca-certificates python3 golang-go libvulkan-dev glslc spirv-headers && rm -rf /var/lib/apt/lists/*
ENV GOMAXPROCS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
WORKDIR /work
