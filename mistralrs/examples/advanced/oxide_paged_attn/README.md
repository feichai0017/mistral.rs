# Oxide paged-decode proof of concept

This binary routes Mistral.rs decode attention through Oxide Infer. It is an
isolated, single-GPU proof of concept. It is not a general or production-safe
provider.

## Checkout

Keep both repositories under one parent directory:

```text
workspace/
|-- oxide-infer/  # d27b6e5
`-- mistral.rs/  # this overlay, based on 8010b6a0
```

The optional Cargo dependencies use `../oxide-infer`. A standalone Mistral.rs
checkout cannot resolve the `oxide-infer` feature.

## Admitted contract

The proof of concept requires:

- Linux, one NVIDIA GPU, one model, and one ordinary CUDA stream.
- Candle event tracking enabled and CUDA Graphs disabled.
- BF16 query and KV cache, head dimension 128, page size 16, and HND KV layout.
- One query token per request and a default `1 / sqrt(128)` attention scale.
- no ALiBi, sinks, sliding window, softcap, or custom attention mask.

The adapter rejects other configurations. It does not silently select a
fallback provider.

## Run

Run the Oxide path:

```bash
MISTRALRS_OXIDE_INFER=1 \
MISTRALRS_CUDA_GRAPHS=0 \
OXIDE_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct \
cargo +nightly-2026-04-03 oxide run --bin oxide_paged_attn \
  --features oxide-infer --arch sm_90
```

Run the same deterministic request through the default Mistral.rs provider:

```bash
OXIDE_BASELINE=1 \
MISTRALRS_CUDA_GRAPHS=0 \
OXIDE_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct \
cargo +nightly-2026-04-03 oxide run --bin oxide_paged_attn \
  --features oxide-infer --arch sm_90
```

For Qwen2.5-1.5B-Instruct, the default provider is Mistral.rs standard paged
attention because its GQA group size is 6.

Run the adapter recovery gate from the paged-attention crate:

```bash
cd mistralrs-paged-attn
cargo +nightly-2026-04-03 oxide run --bin oxide_adapter_h20 \
  --features oxide-infer --arch sm_90
```

## Historical H20 results

The archived 2026-08-11 run predates the project rename. It used an NVIDIA H20,
CUDA 13.1, cuda-oxide `868f8ec`, project commit `d27b6e5`, and Mistral.rs base
`8010b6a0`. The model smoke ran source `9f6acf2a`; the recovery gate ran source
`805dc8f1`. Commit `4f096d7c` later recorded both results.

- The model weight SHA-256 was
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- The native provider completed 196 of 196 paged-decode operator submissions with no error.
  This is 28 layers over seven decode steps, not 196 kernel launches.
- The provider used HND layout and the 8-warp token-parallel algorithm for
  Qwen's 12 query heads and 2 KV heads. It does not mean eight-GPU tensor
  parallelism.
- The adapter retained nine external regions and issued no device-to-device
  copy. This does not prove that the full model is zero-copy.
- The native and standard providers emitted the same eight selected token strings
  and decoded text.
- The selected-token log-probability absolute difference had maximum
  `0.066255` and mean `0.0195841382`.

This run proves provider selection, completion, zero-copy adapter submission,
and one real-model output path for source `9f6acf2a`. It does not prove bitwise
numerical equivalence or a performance advantage.

See [the machine-readable H20 record](./h20-smoke-20260811.json) for source
hashes, raw selected-token log-probabilities, and command outcomes.

### Adapter recovery gate

The H20 adapter recovery gate queued three valid commands and one
device-rejected invalid CSR command. The first drain settled all four commands
and returned `PageIndexOutOfRange` at FIFO position 2.

Each valid output matched the CPU oracle. A fifth valid command then completed
on the same runtime. See
[the adapter recovery record](./h20-adapter-recovery-20260811.json) for the
source hashes and command output.

### Provider boundary

Only decode attention uses Oxide. Prefill and KV cache writes keep their existing
Mistral.rs implementations.

## Model-owned runtime requalification

Commit `b4e4a1c8` moved the runtime, completion FIFO, and provider counters from
process-global state into `NormalPipeline`. Commit `84602212` fixed the
feature-only lifetime errors found by the first H20 build. The validated source
manifest matched `84602212` and used project commit `d27b6e5`.

The adapter gate completed seven commands. It returned one typed
`PageIndexOutOfRange` rejection at FIFO position two, reused the same runtime,
and serialized two concurrent drain callers over a two-command FIFO. All six
valid outputs matched the CPU oracle.

The Qwen model path completed 196 of 196 paged-decode operator calls with no
provider error. The native and standard providers selected the same eight token
strings and decoded text.

The adapter recorded nine external regions and no
adapter-issued device-to-device copy. This remains a statement about the
paged-decode adapter boundary, not the full model.

See the [model-owned runtime record](./h20-model-owned-runtime-84602212-20260811.json)
for source hashes, commands, log hashes, model output, and excluded claims.

## Qualification gaps

Before this provider becomes a general engine option:

- Carry a typed, linear runner authority through the model forward path.
- Define fail-closed behavior for a panic or abandoned model forward.
- Model HND cache writes with explicit read-write storage guards.
- Replace the sibling path dependency with an immutable published source.
- Qualify Graph, speculative decode, tensor parallelism, multiple GPUs,
  multiple streams, multiple models, and larger batches.
