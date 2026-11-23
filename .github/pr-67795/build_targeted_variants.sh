#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

readonly build_dir=/workspace/vllm/build-targeted-gptq
readonly variants=/opt/vllm-variants
readonly candidate_source=/opt/q_gemm.candidate.cu
readonly baseline_sha=5e51c0ca1041168de7aa9e096982aebaeebabd38
readonly candidate_sha=2c94a97dede4fef539e9bba8ddd9e5def7d845bf

python_path=$(
  .venv/bin/python -c 'import sys; print(":".join(sys.path))'
)

cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=/workspace/vllm/.venv/bin/python \
  -DVLLM_PYTHON_PATH="$python_path" \
  -DFETCHCONTENT_BASE_DIR=/workspace/vllm/.deps \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DNVCC_THREADS=1 \
  -DCMAKE_JOB_POOL_COMPILE=compile \
  -DCMAKE_JOB_POOLS=compile=2

mkdir -p "$variants/baseline" "$variants/candidate"

build_variant() {
  local variant=$1
  local sha=$2
  local extension

  cmake --build "$build_dir" --target _C_stable_libtorch --parallel 2
  extension=$(find "$build_dir" -type f \
    -name '_C_stable_libtorch*.so' -print -quit)
  test -n "$extension"
  cp "$extension" "$variants/$variant/"
  printf '%s\n' "$sha" >"$variants/$variant/commit"
}

build_variant baseline "$baseline_sha"
cp "$candidate_source" csrc/quantization/gptq/q_gemm.cu
build_variant candidate "$candidate_sha"

extension_name=$(find "$variants/candidate" -maxdepth 1 \
  -name '_C_stable_libtorch*.so' -exec basename {} \; -quit)
active_extension=$(find vllm -maxdepth 1 \
  -name '_C_stable_libtorch*.so' -print -quit)
test -n "$extension_name"
test -n "$active_extension"
test "$(sha256sum "$variants/baseline/$extension_name" | cut -d' ' -f1)" != \
  "$(sha256sum "$variants/candidate/$extension_name" | cut -d' ' -f1)"
cp "$variants/candidate/$extension_name" "$active_extension"
printf '%s\n' "$active_extension" >"$variants/extension-path"
