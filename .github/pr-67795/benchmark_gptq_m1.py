#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-check and benchmark the fused 4-bit GPTQ kernel at M=1."""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types


def gpu_state() -> dict[str, str]:
    fields = (
        "name,uuid,driver_version,pstate,clocks.current.sm,clocks.max.sm,"
        "power.draw,power.limit,power.default_limit,temperature.gpu"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    return dict(zip(fields.split(","), values, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ramp-seconds", type=float, default=60.0)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    size_m, size_k, size_n, group_size = 1, 4096, 4096, 128
    torch.manual_seed(20261122)
    q = torch.randint(0, 16, (size_k, size_n), dtype=torch.int32, device="cuda")
    scales = (
        torch.rand(size_k // group_size, size_n, device="cuda") * 0.02 + 0.001
    ).to(torch.float16)
    zeros = torch.full(
        (size_k // group_size, size_n),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device="cuda",
    )
    qweight = pack_quantized_values_into_int32(
        q, scalar_types.uint4b8, packed_dim=0
    ).contiguous()
    qzeros = pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)
    ops.gptq_shuffle(qweight, 4)
    x = torch.randn(size_m, size_k, device="cuda", dtype=torch.float16)

    def run() -> torch.Tensor:
        return ops.gptq_gemm(x, qweight, qzeros, scales, True, False, 4)

    output = run().float()
    weight = (q.float() - scalar_types.uint4b8.bias) * scales.float().repeat_interleave(
        group_size, 0
    )
    reference = x.float() @ weight
    rel_l2 = ((output - reference).norm() / reference.norm()).item()
    if rel_l2 >= 0.01:
        raise RuntimeError(f"relative L2 error {rel_l2} exceeds 0.01")
    del output, weight, reference, q
    torch.accelerator.empty_cache()

    state_before_ramp = gpu_state()
    ramp_start = time.monotonic()
    ramp_calls = 0
    while time.monotonic() - ramp_start < args.ramp_seconds:
        for _ in range(100):
            run()
        torch.accelerator.synchronize()
        ramp_calls += 100

    state_before_timing = gpu_state()
    trials_us = [
        statistics.median(
            bench_gpu_time_with_cupti(
                run,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_iters=10,
                repeat_iters=args.repeat,
            )
        )
        * 1000
        for _ in range(args.trials)
    ]
    state_after_timing = gpu_state()
    median_us = statistics.median(trials_us)
    minimum_bytes = (
        x.numel() * x.element_size()
        + qweight.numel() * qweight.element_size()
        + qzeros.numel() * qzeros.element_size()
        + scales.numel() * scales.element_size()
        + size_m * size_n * 2
    )
    result = {
        "label": args.label,
        "commit": args.commit,
        "shape": {
            "m": size_m,
            "k": size_k,
            "n": size_n,
            "group_size": group_size,
            "bits": 4,
        },
        "correctness": {"relative_l2": rel_l2, "limit": 0.01},
        "timing": {
            "method": "FlashInfer CUPTI, CUDA graph, cold L2",
            "median_us": median_us,
            "trials_us": trials_us,
            "repeat_iters": args.repeat,
            "ramp_seconds": args.ramp_seconds,
            "ramp_calls": ramp_calls,
            "minimum_effective_gbps": minimum_bytes / (median_us * 1000),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(torch.accelerator.current_accelerator()),
            "before_ramp": state_before_ramp,
            "before_timing": state_before_timing,
            "after_timing": state_after_timing,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
