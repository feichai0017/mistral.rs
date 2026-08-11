# Qwen GEMM shape census

`loom_gemm_census` records untimed dense-linear host dispatches for one fixed
Qwen2.5-1.5B-Instruct request. The binary requires a clean checkout and writes
one `loom.gemm-shape-census.v1` JSON object.

The runner admits one NVIDIA H20, GPU zero, BF16, tensor parallel size one,
one request, and disabled CUDA Graphs. It rejects a missing provenance value,
an unexpected Qwen call count, or an existing output file.

This census uses host-side Mistral.rs instrumentation and the normal CUDA
backend. The active census graph does not execute Loom or cuda-oxide device
code.

## Prepare the record

Set every value from the checkout and H20 environment used for the run:

```bash
export LOOM_MODEL_PATH=/workspace/models/qwen2.5-1.5b-instruct
export LOOM_GEMM_CENSUS_OUTPUT=/tmp/qwen25-1.5b-gemm-census.jsonl
export LOOM_GEMM_CENSUS_RUN_ID=qwen25-1.5b-h20-001
export LOOM_GEMM_CENSUS_REMOTE=fork
export LOOM_GEMM_CENSUS_REPOSITORY=git@github.com:feichai0017/mistral.rs.git
export LOOM_GEMM_CENSUS_SOURCE_COMMIT="$(git rev-parse HEAD)"
export LOOM_GEMM_CENSUS_WEIGHTS_SHA256="$(sha256sum "$LOOM_MODEL_PATH/model.safetensors" | cut -d' ' -f1)"
export LOOM_GEMM_CENSUS_CONFIG_SHA256="$(sha256sum "$LOOM_MODEL_PATH/config.json" | cut -d' ' -f1)"
export LOOM_GEMM_CENSUS_TOKENIZER_SHA256="$(sha256sum "$LOOM_MODEL_PATH/tokenizer.json" | cut -d' ' -f1)"
export LOOM_GEMM_CENSUS_GPU='NVIDIA H20'
export LOOM_GEMM_CENSUS_COMPUTE_CAPABILITY=9.0
export LOOM_GEMM_CENSUS_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
export MISTRALRS_CUDA_GRAPHS=0
```

The build requires an absolute `NVCC` path and `CUDA_COMPUTE_CAP=90`. It embeds
the canonical NVCC path, complete `nvcc --version` output, and Rust compiler.
The runner checks these values before it creates a record. Cudaforge 0.1.4 maps
compute capability 90 to `sm_90a`, which the record reports.

The runner also compares Git, GPU, and driver values with local state. It
canonicalizes the model directory and hashes each admitted file before and
after the request. The files are `model.safetensors`, `config.json`, and
`tokenizer.json`.

The runner writes and syncs a temporary file in the output directory. It then
publishes the record with a no-clobber hard link, removes the temporary file,
and syncs the directory.

Run the fixed smoke request with ordinary Cargo:

```bash
NVCC=/usr/local/cuda/bin/nvcc \
CUDA_COMPUTE_CAP=90 \
cargo +nightly-2026-04-03 run --release --bin loom_gemm_census \
  --features cuda,gemm-census
```

Run the no-CUDA producer unit tests with the explicit test-only build mode:

```bash
LOOM_GEMM_CENSUS_TEST_BUILD=1 \
cargo +nightly-2026-04-03 test -p mistralrs --bin loom_gemm_census \
  --features gemm-census
```

The build rejects `LOOM_GEMM_CENSUS_TEST_BUILD` when it enables `cuda`.

Validate the raw record with Loom commit
`8b971b064d4246b2cd5cbc74f9902a51c720aefa`:

```bash
python3 ../loom-infer/tools/gemm/shape_census.py validate \
  /tmp/qwen25-1.5b-gemm-census.jsonl
```

This record supports dispatch-count and shape claims only. It does not contain
timing or CUDA kernel launch counts.
