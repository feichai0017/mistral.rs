# Oxide paged-decode proof of concept

This binary routes Mistral.rs decode attention through Oxide Infer. It is an
isolated, single-GPU proof of concept. It is not a general or production-safe
provider.

## Checkout

The optional Cargo dependencies resolve Oxide Infer from the immutable Git
commit `02faf27b116f05831dca7261fb605be13faa2df4`. A standalone Mistral.rs
checkout can therefore build the `oxide-infer` feature without a sibling
Oxide Infer checkout.

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
cargo +nightly-2026-04-03 oxide run --bin oxide_adapter_gate \
  --features oxide-infer --arch sm_90
```

## Current source requalification

The 2026-08-12 run used the immutable Oxide Infer source above and exercised
two local BF16 model configurations on one GPU:

- Qwen2.5-1.5B-Instruct, 12 query heads, 2 KV heads, GQA group size 6.
- Qwen2.5-7B-Instruct, 28 query heads, 4 KV heads, GQA group size 7.

For each model, Oxide completed 196 of 196 paged-decode operator submissions
with no provider error. Oxide and the default Mistral.rs provider emitted the
same eight selected token strings and decoded text. The adapter recorded HND,
the 8-warp token-parallel algorithm, nine external regions, and no
adapter-issued device-to-device copy.

The generic adapter recovery gate completed seven commands, reported one typed
`PageIndexOutOfRange` failure at FIFO position two, and reused the same runtime.
All six valid outputs matched the CPU oracle exactly.

See the [current-source requalification record](./h20-current-oxide-02faf27-two-model-requalification-20260812.json)
for source and model hashes, raw selected-token log probabilities, command
outcomes, and excluded claims. The recorded request timings are observations,
not a performance comparison: the runs did not use a repeated, counterbalanced
benchmark protocol and CUDA driver JIT caching differed between runs.

## Steady-state benchmark

Build the benchmark once through cuda-oxide, then run its six-block suite:

```bash
cargo +nightly-2026-04-03 oxide build --arch sm_90 -- \
  --bin oxide_paged_attn_bench --features oxide-infer --release

target/release/oxide_paged_attn_bench suite \
  --model-path /path/to/qwen2.5-1.5b-instruct \
  --model-name Qwen2.5-1.5B-Instruct \
  --output /path/to/result.json
```

The suite uses an Oxide, baseline, baseline, Oxide, Oxide, baseline schedule.
Each block runs in a fresh process, loads one model, disables prefix caching and
CUDA Graphs, performs five unmeasured warmups, and then measures 20 streaming
requests. The reported TTFT starts before request submission and ends at the
first non-empty generated content. TPOT covers the remaining completion tokens.
The suite also records end-to-end latency, decode throughput, CUDA driver
device-used memory deltas from the post-context baseline, per-block medians,
provider counters, and pooled nearest-rank P50/P95 values. The memory delta is a
device-wide steady-state observation, not a process-private or allocator peak.

The fixed prompt is expected to reach the 64-token cap. The suite fails closed
if a request ends early, output changes within or across provider blocks, an
Oxide command fails, or the adapter issues a device-to-device copy.

### Current steady-state evidence

The 2026-08-13 run used commit `7b6f9575`, Oxide Infer `02faf27b`, BF16, one
stream, and one recorded NVIDIA H20. Each row pools 60 measured requests per
provider across three fresh-process blocks. Both model directories matched the
file hashes in the current-source requalification record above. Lower TTFT and
TPOT are better; higher decode throughput is better.

| Model | TTFT P50, Oxide / standard | TPOT P50, Oxide / standard | Decode P50, Oxide / standard | Oxide / standard decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 12.52 / 9.46 ms | 6.430 / 4.819 ms | 155.51 / 207.49 tok/s | 0.749x |
| Qwen2.5-7B-Instruct | 15.06 / 14.75 ms | 11.054 / 9.470 ms | 90.46 / 105.60 tok/s | 0.857x |

All 211,680 measured Oxide layer-decode submissions completed with zero
provider failure and zero adapter-issued device-to-device copy. Both providers
had the same post-warmup device-memory delta for each model: 4,674 MiB for 1.5B
and 22,146 MiB for 7B. The current result establishes a stable integration, not
a performance advantage: Oxide remains 25.1% below standard decode throughput
for 1.5B and 14.3% below for 7B in this request shape.

FlashInfer is not a third row in this comparison. These models use GQA group
sizes 6 and 7, which are outside the FlashInfer decode dispatch supported here,
so the matched baseline is standard Mistral.rs paged attention. See the full
[1.5B record](./h20-steady-state-qwen2.5-1.5b-02faf27-20260813.json) and
[7B record](./h20-steady-state-qwen2.5-7b-02faf27-20260813.json) for raw samples,
P95 values, counters, protocol metadata, and excluded claims.

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
- Promote the immutable Git pin to a released crate dependency when available.
- Qualify Graph, speculative decode, tensor parallelism, multiple GPUs,
  multiple streams, multiple models, and larger batches.
