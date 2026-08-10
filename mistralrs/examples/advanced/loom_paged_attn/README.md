# Loom paged-decode proof of concept

This binary routes Mistral.rs decode attention through Loom Infer. It is an
isolated, single-GPU proof of concept. It is not a general or production-safe
provider.

## Checkout

Keep both repositories under one parent directory:

```text
workspace/
|-- loom-infer/  # d27b6e5
`-- mistral.rs/  # this overlay, based on 8010b6a0
```

The optional Cargo dependencies use `../loom-infer`. A standalone Mistral.rs
checkout cannot resolve the `loom-infer` feature.

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

Run the Loom path:

```bash
MISTRALRS_LOOM_INFER=1 \
MISTRALRS_CUDA_GRAPHS=0 \
LOOM_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct \
cargo +nightly-2026-04-03 oxide run --bin loom_paged_attn \
  --features loom-infer --arch sm_90
```

Run the same deterministic request through the default Mistral.rs provider:

```bash
LOOM_BASELINE=1 \
MISTRALRS_CUDA_GRAPHS=0 \
LOOM_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct \
cargo +nightly-2026-04-03 oxide run --bin loom_paged_attn \
  --features loom-infer --arch sm_90
```

For Qwen2.5-1.5B-Instruct, the default provider is Mistral.rs standard paged
attention because its GQA group size is 6.

## H20 result

The 2026-08-11 run used an NVIDIA H20, CUDA 13.1, cuda-oxide `868f8ec`, Loom
`d27b6e5`, and the Mistral.rs base `8010b6a0`.

- The model weight SHA-256 was
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- Loom completed 196 of 196 paged-decode operator submissions with no error.
  This is 28 layers over seven decode steps, not 196 kernel launches.
- The provider used HND layout and the 8-warp token-parallel algorithm for
  Qwen's 12 query heads and 2 KV heads. It does not mean eight-GPU tensor
  parallelism.
- The adapter retained nine external regions and issued no device-to-device
  copy. This does not prove that the full model is zero-copy.
- Loom and the standard provider emitted the same eight selected token strings
  and decoded text.
- The selected-token log-probability absolute difference had maximum
  `0.066255` and mean `0.0195841382`.

This run proves provider selection, lifecycle completion, zero-copy adapter
submission, and one real-model output path. It does not prove bitwise numerical
equivalence or a performance advantage.

Only decode attention uses Loom. Prefill and KV cache writes keep their existing
Mistral.rs implementations.

## Qualification gaps

Before this provider becomes a general engine option:

- Carry a typed, linear runner authority through the model forward path.
- replace the process-global runtime and completion queue with model-owned
  state.
- Prevent raw forward calls from bypassing completion drain.
- Model HND cache writes with explicit read-write storage guards.
- Test adapter-level invalid metadata, FIFO drain, and same-runtime recovery.
- Replace the sibling path dependency with an immutable published source.
- Qualify Graph, speculative decode, tensor parallelism, multiple GPUs,
  multiple streams, multiple models, and larger batches.
