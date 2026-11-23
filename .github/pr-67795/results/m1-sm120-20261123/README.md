# PR #67795: M=1 sm_120 validation

## Question

Does capping the fused 4-bit GPTQ split-K launch at four blocks improve the
`M=1, K=N=4096` kernel on an unrestricted RTX PRO 6000 Blackwell Server
Edition?

## Builds

- Baseline: `5e51c0ca1041168de7aa9e096982aebaeebabd38`
- Candidate: `2c94a97dede4fef539e9bba8ddd9e5def7d845bf`
- Image source: `1cd8f855ca965fae62b573577e05a0ddd4ee94bf`
- GitHub Actions run:
  [45090213570](https://github.com/tryambak2019/vllm/actions/runs/45090213570)

Both `_C_stable_libtorch` variants were built in `Release` mode for `sm_120`
from one CMake tree. The baseline was saved first; only `q_gemm.cu` was then
replaced before the incremental candidate rebuild. The official exact-commit
`cu132` wheel supplied the common Python/runtime environment, not either
measured extension.

## Protocol

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`)
- Power limit: 600 W enforced / 600 W default
- PyTorch: `2.14.0+cu132`
- CUDA: 13.2
- CUPTI Python: 13.5.0
- Shape: GPTQ 4-bit, `M=1`, `K=N=4096`, group size 128
- Timing: FlashInfer CUPTI, CUDA graph, cold L2
- Stabilization: 60 seconds per arm
- Order: baseline A, candidate, baseline B
- Trials: seven medians of 100 repetitions per arm
- Correctness limit: relative L2 below 0.01 against an FP32-dequantized
  reference

## Result

| Arm | Median latency | Relative L2 |
| --- | ---: | ---: |
| Baseline A | 16.640 us | 0.0009221 |
| Candidate | 81.824 us | 0.0004965 |
| Baseline B | 16.608 us | 0.0009705 |

- Baseline mean: 16.624 us
- Candidate / baseline: 4.922x
- Regression: 392.2%
- Baseline bookend drift: 0.192%

All arms passed correctness. The baseline bookends agree closely, and the
candidate regression is much larger than the small clock difference between
arms. This result supports the PR's existing negative-result disposition: at
`M=1` on this full-power server SKU, the loss of split-K parallelism dominates
the reduction in atomic fan-in.

## Artifacts

- [`baseline-a.json`](baseline-a.json)
- [`candidate.json`](candidate.json)
- [`baseline-b.json`](baseline-b.json)
- [`SHA256SUMS`](SHA256SUMS)

The first M=1 attempt silently fell back from CUPTI and did not flush L2, so it
is intentionally excluded. These artifacts are from the corrected fail-closed
CUPTI/cold-L2 run only.

AI assistance was used for the validation workflow, benchmark harness, and
analysis.
