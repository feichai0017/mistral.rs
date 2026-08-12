# Matched M=1 GEMV benchmark

`oxide_gemv_bench` measures the existing Mistral.rs custom CUDA GEMV on the
five Qwen2.5-1.5B census shapes. It uses the same bit-exact dyadic fixture and
JSON Lines contract as the paired Oxide Infer native and cuBLASLt runners.

The timed interval contains only launches into a preallocated output on
Candle's current CUDA stream. Tensor construction, host-to-device copies, CPU
reference calculation, output readback, and correctness comparison remain
outside the interval.

Run the H20 baseline with a committed source identity:

```bash
NVCC=/usr/local/cuda/bin/nvcc \
CUDA_COMPUTE_CAP=90 \
MISTRALRS_SOURCE_COMMIT="$(git rev-parse HEAD)" \
OXIDE_BENCH_RUN_LABEL=mistral_first \
cargo +nightly-2026-04-03 run --release -p mistralrs-quant \
  --bin oxide_gemv_bench --features gemv-bench
```

Use `OXIDE_BENCH_WARMUP`, `OXIDE_BENCH_LAUNCHES`, and
`OXIDE_BENCH_SAMPLES` to override the defaults of 20, 100, and 30. For a
promotion decision, run Mistral/Oxide and Oxide/Mistral process pairs with
distinct labels. These operator records do not establish engine-level TPOT or
throughput claims.
