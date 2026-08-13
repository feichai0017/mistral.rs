# Oxide paged-decode proof of concept

This binary routes Mistral.rs decode attention through Oxide Infer. It is an
isolated, single-GPU proof of concept. It is not a general or production-safe
provider.

## Checkout

The optional Cargo dependencies resolve Oxide Infer from the immutable Git
commit `af4eae60f330f9b14123da6cc30a7dc0d1117f32`. A standalone Mistral.rs
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

The 2026-08-12 run used Oxide Infer commit `02faf27b` and exercised
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

## Serving benchmark

Build the benchmark once through cuda-oxide, then run its six-block suite:

```bash
cargo +nightly-2026-04-03 oxide build --arch sm_90 -- \
  --bin oxide_paged_attn_bench --features oxide-infer --release

MISTRALRS_OXIDE_PROFILE=1 target/release/oxide_paged_attn_bench suite \
  --model-path /path/to/qwen2.5-1.5b-instruct \
  --model-name Qwen2.5-1.5B-Instruct \
  --output /path/to/result.json \
  --concurrency 4
```

The suite uses an Oxide, baseline, baseline, Oxide, Oxide, baseline schedule.
Each block runs in a fresh process, loads one model, disables prefix caching and
CUDA Graphs, performs five unmeasured waves, and then measures 20 streaming
waves. A barrier releases the requested number of requests together, and the
model scheduler admits up to the same number of sequences. Concurrency defaults
to one when the flag is omitted.

The reported TTFT starts before each request submission and ends at its first
non-empty generated content. TPOT covers the remaining completion tokens. Each
wave additionally reports aggregate output tokens per second and requests per
second from coordinated release through the final request completion. The suite
also records end-to-end latency, CUDA driver device-used memory deltas from the
post-context baseline, per-block medians, provider counters, and pooled
nearest-rank P50/P95 values. The memory delta is a device-wide steady-state
observation, not a process-private or allocator peak.

The fixed prompt is expected to reach the 64-token cap. The suite fails closed
if a request ends early, output changes within or across provider blocks, an
Oxide command fails, or the adapter issues a device-to-device copy.

### Current steady-state evidence

The 2026-08-13 binding-reuse run used commit `d7c86540`, Oxide Infer `840b0658`,
BF16, one stream, one request per wave, and one recorded NVIDIA H20. Each row
pools 60 measured requests per provider across three fresh-process blocks. Both
model directories matched the file hashes in the current-source requalification
record above. Lower TTFT and TPOT are better; higher decode throughput is better.

| Model | TTFT P50, Oxide / standard | TPOT P50, Oxide / standard | Decode P50, Oxide / standard | Oxide / standard decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 12.46 / 9.51 ms | 5.083 / 4.826 ms | 196.74 / 207.16 tok/s | 0.950x |
| Qwen2.5-7B-Instruct | 15.14 / 14.76 ms | 9.788 / 9.477 ms | 102.17 / 105.52 tok/s | 0.968x |

All 211,680 measured Oxide layer-decode submissions completed with zero
provider failure and zero adapter-issued device-to-device copy. Both providers
had the same post-warmup device-memory delta for each model: 4,674 MiB for 1.5B
and 22,146 MiB for 7B. Reusing settled binding storage raised Oxide decode
throughput by 26.5% for 1.5B and 12.9% for 7B relative to the earlier run.
Oxide remains 5.0% below standard decode throughput for 1.5B and 3.2% below for
7B in this request shape. Optional adapter profiling measured average host
enqueue costs of 20.60 and 21.17 microseconds per layer, respectively.

FlashInfer is not a third row in this comparison. These models use GQA group
sizes 6 and 7, which are outside the FlashInfer decode dispatch supported here,
so the matched baseline is standard Mistral.rs paged attention. See the full
[1.5B record](./h20-binding-reuse-qwen2.5-1.5b-840b065-20260813.json) and
[7B record](./h20-binding-reuse-qwen2.5-7b-840b065-20260813.json) for raw
samples, P95 values, counters, protocol metadata, and excluded claims.

### Current concurrent serving evidence

The 2026-08-13 serving matrix used commit `ac2979e2`, Oxide Infer `840b0658`,
BF16, one stream, and one recorded NVIDIA H20. Each row pools 60 measured waves
per provider across three fresh-process blocks. The request sample count is 60
times the concurrency. Aggregate output throughput is the sum of completion
tokens divided by wall time from coordinated wave release through the last
completion.

| Model | Concurrency | Aggregate output P50, Oxide / standard | Oxide / standard | TTFT P95, Oxide / standard | End-to-end P95, Oxide / standard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 1 | 189.44 / 204.32 tok/s | 0.927x | 12.76 / 9.53 ms | 340.64 / 314.30 ms |
| Qwen2.5-1.5B-Instruct | 4 | 514.23 / 544.85 tok/s | 0.944x | 28.23 / 17.14 ms | 513.37 / 470.95 ms |
| Qwen2.5-1.5B-Instruct | 8 | 728.35 / 763.44 tok/s | 0.954x | 28.57 / 28.60 ms | 707.25 / 673.19 ms |
| Qwen2.5-1.5B-Instruct | 16 | 2,090.68 / 3,043.22 tok/s | 0.687x | 45.43 / 47.04 ms | 494.73 / 409.89 ms |
| Qwen2.5-7B-Instruct | 1 | 101.24 / 104.74 tok/s | 0.967x | 23.54 / 14.82 ms | 646.27 / 611.77 ms |
| Qwen2.5-7B-Instruct | 4 | 219.55 / 225.35 tok/s | 0.974x | 41.18 / 38.68 ms | 1,169.82 / 1,137.13 ms |
| Qwen2.5-7B-Instruct | 8 | 254.21 / 257.59 tok/s | 0.987x | 71.13 / 69.63 ms | 2,016.94 / 1,988.84 ms |
| Qwen2.5-7B-Instruct | 16 | 1,490.61 / 1,595.24 tok/s | 0.934x | 131.23 / 128.61 ms | 694.86 / 643.30 ms |

Oxide aggregate output throughput scaled from concurrency 1 to 16 by 11.0x for
1.5B and 14.7x for 7B. It remained within 7.3% of standard through concurrency
8 for 1.5B and within 6.6% at every measured concurrency for 7B. The 1.5B
concurrency-16 result exposes a specific optimization gap: Oxide was 31.3%
below standard despite continuing to scale in absolute throughput.

Across both models and all four concurrency levels, all 846,720 measured Oxide
layer-decode submissions completed with zero provider failure and zero
adapter-issued device-to-device copy. Every request reached 64 completion tokens,
and both providers returned the same deterministic text. Oxide recorded
`Bf16PagedBatchDecode`, HND, and the 8-warp token-parallel algorithm. The raw
records retain every request, wave, block median, provider counter, memory
observation, and excluded claim. They are stored as gzip-compressed JSON and can
be inspected with `gzip -cd FILE.json.gz | jq`:

- 1.5B: [c1](./h20-serving-qwen2.5-1.5b-c1-ac2979e-20260813.json.gz),
  [c4](./h20-serving-qwen2.5-1.5b-c4-ac2979e-20260813.json.gz),
  [c8](./h20-serving-qwen2.5-1.5b-c8-ac2979e-20260813.json.gz), and
  [c16](./h20-serving-qwen2.5-1.5b-c16-ac2979e-20260813.json.gz).
- 7B: [c1](./h20-serving-qwen2.5-7b-c1-ac2979e-20260813.json.gz),
  [c4](./h20-serving-qwen2.5-7b-c4-ac2979e-20260813.json.gz),
  [c8](./h20-serving-qwen2.5-7b-c8-ac2979e-20260813.json.gz), and
  [c16](./h20-serving-qwen2.5-7b-c16-ac2979e-20260813.json.gz).

This is a fixed-concurrency wave comparison, not a saturation or production
capacity claim. FlashInfer remains excluded for these model shapes because GQA
group sizes 6 and 7 are outside the decode dispatch supported by this adapter.

### Concurrency-16 host-overhead diagnosis

The 2026-08-13 diagnostic run used clean commit `f51603235`, Oxide Infer
`c113d2c0`, Qwen2.5-1.5B-Instruct, BF16, concurrency 16, and optional host
profiling. Three fresh Oxide processes completed 26,460 measured layer-decode
submissions with zero failure and zero adapter-issued device-to-device copy.
The values below are per-operator means pooled from the three Oxide blocks.

| Host enqueue stage | Mean per operator | Share of 24.435 us total |
| --- | ---: | ---: |
| Output and scratch allocation | 4.110 us | 16.8% |
| Candle storage guards | 1.393 us | 5.7% |
| External-region binding | 1.160 us | 4.7% |
| Engine interop, including provider | 13.375 us | 54.7% |

The engine-interoperability total splits into 0.655 us for the pre-event
handoff, 9.492 us for provider submission, 2.103 us for device-status readback,
and 0.585 us for the post-event handoff. Inside provider submission, argument
preflight took 0.172 us, the metadata-validation launch took 4.240 us, and the
attention launch took 4.830 us. Repeated metadata validation plus status
readback therefore accounts for 6.343 us per layer, or 26.0% of the complete
host enqueue path. Both event handoffs together account for 1.240 us, or 5.1%.

This evidence rejects cross-stream handoff as the first optimization target.
The next target is an explicit trusted-metadata capability issued where
Mistral.rs constructs and validates the CPU page table. The ordinary checked
Oxide path and its typed recovery test must remain available. The diagnostic
suite used only five measured waves per block with profiling enabled, so its
throughput observations are not a replacement for the serving matrix above.
See the [compressed raw record](./h20-r7-provider-profile-qwen2.5-1.5b-c16-f516032-20260813.json.gz)
for every request, wave, counter, timing accumulator, and excluded claim.

### Trusted metadata result

Commit `8a52830c3` pins Oxide Infer `af4eae60` and lets the serialized
model path issue a trusted-metadata capability for page tables constructed on
the CPU from scheduler-owned block tables and context lengths. The public
checked adapter path remains the default. Its H20 recovery gate still rejected
the invalid physical page with `PageIndexOutOfRange`, settled all seven queued
commands, reused the runtime, and matched all six valid outputs exactly.

The profiled Qwen2.5-1.5B-Instruct concurrency-16 run repeated the R7 protocol:
two warmup waves and five measured waves in each of six fresh-process blocks.
All 26,460 measured Oxide layer-decode submissions completed with zero failure
and zero adapter-issued device-to-device copy. Every Oxide block recorded
`TrustedByAdapter`.

| Pooled host stage | R7 checked | R8 trusted | Change |
| --- | ---: | ---: | ---: |
| Complete enqueue | 24.435 us | 18.332 us | -25.0% |
| Engine interop | 13.864 us | 7.909 us | -42.9% |
| Provider metadata launch | 4.240 us | 0.000 us | removed |
| Status-readback timing bucket | 2.103 us | 0.057 us | -97.3% |

The trusted command reserves no device-status readback. Its residual 0.057 us
is the measured host timer and no-op finalization bucket. The complete enqueue
path saved 6.103 us per layer relative to the earlier checked run. See the
[profile record](./h20-r8-trusted-profile-qwen2.5-1.5b-c16-8a52830-20260813.json.gz).

A second run disabled profiling and restored the full serving protocol of five
warmup and 20 measured waves per block. It completed 105,840 measured Oxide
layer-decode submissions and 960 requests per provider with no failure. The
table compares it with the earlier checked concurrency-16 serving record on the
same H20 and benchmark protocol.

| Metric | Checked `ac2979e2` | Trusted `8a52830c3` | Change |
| --- | ---: | ---: | ---: |
| Oxide aggregate output P50 | 2,090.68 tok/s | 2,134.58 tok/s | +2.10% |
| Oxide decode P50 | 141.87 tok/s | 145.29 tok/s | +2.41% |
| Oxide / standard aggregate P50 | 0.687x | 0.700x | +1.87% relative |
| Oxide / standard decode P50 | 0.657x | 0.671x | +2.06% relative |

Trusted metadata improves the targeted path, but it does not close the
concurrency-16 gap: aggregate output remains 30.0% below standard in the new
run. The next investigation should measure device time for the actual batched
engine shapes rather than further optimize metadata validation or stream
handoff. See the [serving record](./h20-r8-trusted-serving-qwen2.5-1.5b-c16-8a52830-20260813.json.gz).

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
