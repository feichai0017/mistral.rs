# Oxide paged-decode proof of concept

This binary routes Mistral.rs decode attention through Oxide Infer. It is an
isolated, single-GPU proof of concept. It is not a general or production-safe
provider.

## Checkout

The optional Cargo dependencies resolve Oxide Infer from the immutable Git
commit `fbf722985d8e8ff9595731bd75f20ead446371cb`. A standalone Mistral.rs
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

Build and run the same-process checked/trusted adapter benchmark from the
workspace root:

```bash
cargo +nightly-2026-04-03 oxide build --arch sm_90 -- \
  --release -p mistralrs-paged-attn --features oxide-infer \
  --bin oxide_adapter_bench
target/release/oxide_adapter_bench --output /tmp/oxide-adapter.json \
  --warmups 100 --iterations 1000
```

Add `--comparison oxide-standard` to compare the trusted Oxide provider with
standard Mistral.rs paged attention instead of the default checked/trusted A/B.
Use `--comparison bridge-direct` to isolate stream-handoff cost, or
`--comparison direct-standard` to compare direct Oxide with standard paged
attention. Direct-stream comparisons reject device profiling because that
diagnostic is defined only for the event bridge.
Add `--oxide-device-profile 1` to collect diagnostic CUDA-event segments for
the Oxide cross-stream handoff and provider work. This option is disabled by
default and its extra events intentionally perturb the outer device window.

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

Commit `bda213bc7` pins Oxide Infer `d3f7e4e3` and lets the serialized model
path issue a single-use trusted-metadata capability for page tables constructed
on the CPU from scheduler-owned block tables and context lengths. The public
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
| Complete enqueue | 24.435 us | 18.245 us | -25.3% |
| Engine interop | 13.864 us | 7.825 us | -43.6% |
| Provider metadata launch | 4.240 us | 0.000 us | removed |
| Status-readback timing bucket | 2.103 us | 0.050 us | -97.6% |

The trusted command reserves no device-status readback. Its residual 0.050 us
is the measured host timer and no-op finalization bucket. The complete enqueue
path saved 6.190 us per layer relative to the earlier checked run. See the
[profile record](./h20-r8-trusted-profile-qwen2.5-1.5b-c16-bda213b-20260813.json.gz).

Two runs disabled profiling and restored the full serving protocol of five
warmup and 20 measured waves per block. Together they completed 211,680
measured Oxide layer-decode submissions and 1,920 requests per provider with no
failure. Their Oxide/standard aggregate-output ratios were 0.725x and 0.691x,
which exposes material run-to-run drift. The table pools the raw samples from
both trusted runs with the same nearest-rank method and compares them with the
earlier checked concurrency-16 serving record.

| Metric | Checked `ac2979e2` | Trusted `bda213bc7` | Change |
| --- | ---: | ---: | ---: |
| Oxide aggregate output P50 | 2,090.68 tok/s | 2,069.24 tok/s | -1.03% |
| Oxide decode P50 | 141.87 tok/s | 140.49 tok/s | -0.98% |
| Oxide / standard aggregate P50 | 0.687x | 0.696x | +1.33% relative |
| Oxide / standard decode P50 | 0.657x | 0.667x | +1.55% relative |

The stable claim is the 25.3% reduction in measured host enqueue cost. These
cross-run serving results do not establish an end-to-end speedup: absolute
Oxide throughput is slightly lower, the normalized ratio is slightly higher,
and aggregate output remains 30.4% below standard after pooling. The next
investigation should use a same-binary checked/trusted comparison or measure
CUDA-event device time for the actual batched engine shapes. See serving
[replicate one](./h20-r8-trusted-serving-r1-qwen2.5-1.5b-c16-bda213b-20260813.json.gz)
and [replicate two](./h20-r8-trusted-serving-r2-qwen2.5-1.5b-c16-bda213b-20260813.json.gz).

### Same-process checked/trusted A/B

Commit `717c486bc` adds an adapter microbenchmark that switches between the
public checked and trusted paths in one binary, process, runtime, stream, and
set of immutable tensors. Its BF16/HND shape uses batch 16, 12 query heads, two
KV heads, D128, page size 16, and six pages per sequence. The balanced schedule
is checked, trusted, trusted, checked, checked, trusted. CUDA events on the
external stream bracket the full Oxide pre-event, provider work, and post-event
bridge.

Two fresh processes each ran 100 warmups and 1,000 measured submissions per
block. The pooled record contains 6,000 measured submissions per path. All
12,000 submissions completed with zero failure and zero adapter-issued D2D
copy. Both paths matched the CPU reference with maximum absolute error
`1.1920929e-7`.

| Pooled P50 | Checked | Trusted | Change |
| --- | ---: | ---: | ---: |
| Adapter host enqueue | 19.810 us | 14.139 us | -28.6% |
| External-stream CUDA window | 34.048 us | 23.360 us | -31.4% |

The profiled per-submission means also move in the expected stages: interop
falls from 12.292 to 6.450 us, metadata launch host work falls from 3.938 us to
zero, and the status-readback host bucket falls from 1.792 to 0.036 us. The
block P50 values do not reverse direction in either process.

The CUDA result proves that trusted metadata removes device-visible stream
work as well as host submission work. It is not an attention-kernel-only
measurement: the 10.688 us saved includes the removed metadata validator,
status transfer, and their scheduling effects. The attention algorithm is the
same `PagedBatchDecodeTokenParallel8` kernel in both paths. The next comparison
should bracket trusted Oxide and standard Mistral.rs paged attention at this
same layer shape to locate the remaining serving gap.

See [replicate one](./h20-r9-checked-trusted-r1-717c486-20260813.json.gz) and
[replicate two](./h20-r9-checked-trusted-r2-717c486-20260813.json.gz). Their
SHA-256 digests are `6f3440544f9865f60f646cb39671afef927f188db6fe02477297615e9d74537c`
and `ddcecae521f3c8345706dc018280bc9547770998949a9749f17ac11534a1610a`.

### Same-process Oxide/standard A/B

Commit `4c883aff8` extends the same binary with an `oxide-standard` comparison.
Both providers receive the same BF16 query, logical KV values, physical-page
mapping, context lengths, scale, and batch-16 D128/GQA6 shape. Their real cache
layouts are materialized before timing: HND for Oxide and the packed key/value
layouts used by standard paged attention. Oxide selects
`PagedBatchDecodeTokenParallel8`; standard selects `PagedAttentionV1` for the
recorded 81-to-96-token contexts. Optional Oxide host profiling is disabled so
the outer host timer treats both providers equally.

Two fresh processes used the trusted, standard, standard, trusted, trusted,
standard schedule, with 100 warmups and 1,000 measured calls per block. The
pooled result contains 6,000 calls per provider. All 6,000 Oxide submissions
completed without failure or adapter-issued D2D copy. Both providers matched
the CPU reference: maximum absolute error was `1.1920929e-7` for Oxide and
`9.765625e-4` for standard.

| Pooled P50 | Oxide trusted | Standard V1 | Standard change |
| --- | ---: | ---: | ---: |
| Host provider call | 13.218 us | 6.369 us | -51.8% |
| Complete CUDA window | 22.368 us | 16.096 us | -28.0% |

The device ratio is `0.720x` standard/Oxide, leaving a 6.272 us gap in one
provider invocation. This is close in magnitude to the earlier serving gap and
shows that host metadata validation was not the remaining primary bottleneck.

This is a provider-path comparison, not a pure kernel comparison. The Oxide
window includes its cross-stream event bridge and its attention work; its
adapter also allocates output, LSE, and status tensors, while standard launches
directly on the engine stream and returns only output. Cache layout setup is
excluded for both providers. The next diagnostic should time the Oxide
attention launch on its internal stream and test an engine path that does not
request unused LSE/status storage.

See [replicate one](./h20-r10-oxide-standard-r1-4c883af-20260813.json.gz) and
[replicate two](./h20-r10-oxide-standard-r2-4c883af-20260813.json.gz). Their
SHA-256 digests are `fdf2d7f642e6b2439ec5747fa7d91cdd5f95fd5b402afefa970db834c340bb65`
and `0a6290b1db754b646299f5fcaf91f7e359d40c3fe4b5e6b3e9b55cf4dbe8551a`.

### Oxide device-segment diagnostic

Mistral.rs commit `278972158` pins Oxide Infer `39058b6d` and enables optional
timing events in each interop handoff slot. Successful traces report four
ordered CUDA timeline intervals: external-stream start to provider start,
provider start to provider/status completion, provider completion to external
stream reacquisition, and the complete internal interval. The ordinary queue
does not create or record these timing events.

Two fresh H20 processes repeated the balanced Oxide/standard schedule with 100
warmups and 1,000 measured calls per block. The records contain 6,000 measured
Oxide calls and 6,000 standard calls. Every Oxide call completed, emitted one
device profile, retained nine external regions, and issued no adapter D2D copy.
Maximum absolute error against the CPU reference remained `1.1920929e-7` for
Oxide and `9.765625e-4` for standard.

| Instrumented Oxide interval | Replicate 1 mean | Replicate 2 mean | Pooled mean | Pooled share |
| --- | ---: | ---: | ---: | ---: |
| Pre-handoff bridge | 4.201 us | 4.312 us | 4.257 us | 16.4% |
| Provider work | 12.373 us | 13.186 us | 12.779 us | 49.3% |
| Post-handoff bridge | 9.094 us | 8.679 us | 8.887 us | 34.3% |
| Both bridge segments | 13.296 us | 12.991 us | 13.144 us | 50.7% |
| Complete internal interval | 25.669 us | 26.177 us | 25.923 us | 100.0% |

The pooled standard outer-window P50 was `16.032 us`. Oxide's provider-only
mean was `12.779 us`, but these are not matched timing boundaries or a pure
kernel A/B: Oxide runs on its private stream, while the standard window is
recorded on the engine stream. No kernel-performance advantage is claimed.
The profiled Oxide outer-window P50 was `35.008 us`; it is excluded from the
provider comparison because four diagnostic events add observable overhead.

This result changes the next optimization target. In the instrumented Oxide
timeline the two event-bridge segments are 50.7%, slightly larger than provider
work. A direct, non-owning engine-stream submission path should therefore be
tested before changing the attention kernel. LSE cannot simply be removed:
the current kernel writes it. Trusted metadata does not read status, but status
remains part of the checked binding protocol, so removing that allocation also
requires an explicit trusted-only ABI rather than an adapter shortcut.

See [replicate one](./h20-r11-device-profile-r1-2789721-20260813.json.gz) and
[replicate two](./h20-r11-device-profile-r2-2789721-20260813.json.gz). Their
SHA-256 digests are `a9f0c46897185d0a0b1c86de3d8b169477a0af2a896a0ad5f55c4c2543d3b300`
and `a4dce4b27fa68a51232891b1dc1de7bbe19286cba1a795e1db4fe51bed4d5ef1`.

### Direct engine-stream result

Mistral.rs commit `18628b249` pins Oxide Infer `fbf72298` and cuda-oxide
`e107c06d`. The direct path wraps the engine stream without destruction
ownership and submits through the same typed command queue, checked bindings,
provider dispatch, device-status protocol, FIFO completion, and recovery path.
The event-bridged path remains the default.

The H20 recovery gate passed in both modes. Each mode settled seven commands,
reported the intentional invalid page as a typed failure at FIFO position two,
reused the same runtime, matched every valid output exactly, and synchronized
the external stream after dropping the runtime. This verifies that the direct
wrapper does not destroy the engine-owned stream in the exercised lifecycle.

Two fresh processes per comparison used 100 warmups and 1,000 measured calls
in each of six balanced blocks. Each row pools 6,000 calls per path. All 18,000
measured Oxide calls completed without failure or adapter-issued D2D copy.
Oxide's maximum absolute error was `1.1920929e-7`; standard's was
`9.765625e-4`.

| Pooled P50 | Event bridge | Direct stream | Direct change |
| --- | ---: | ---: | ---: |
| Host provider call | 13.341 us | 12.473 us | -6.5% |
| Complete CUDA window | 22.496 us | 18.752 us | -16.6% |

Every bridge/direct block moved in the same direction. The direct result also
remained stable across the separate standard comparison: its device P50 was
`18.528 us`, versus `15.936 us` for standard V1.

| Pooled P50 | Oxide direct | Standard V1 | Standard change |
| --- | ---: | ---: | ---: |
| Host provider call | 12.193 us | 6.310 us | -48.2% |
| Complete CUDA window | 18.528 us | 15.936 us | -14.0% |

Direct submission removes 3.744 us from the matched bridge comparison, while
2.592 us remains between direct Oxide and standard in the second matched
comparison. The earlier R10 gap was 6.272 us, so the residual is 58.7% smaller;
that percentage is cross-run context, not a single-process three-way result.
The next diagnostic should separate host setup, output/LSE allocation, and
attention-kernel work; another stream-handoff change is not the first target.
These are full provider windows for one synthetic shape, not pure-kernel or
end-to-end model timings.

See bridge/direct [replicate one](./h20-r12-bridge-direct-r1-18628b2-20260813.json.gz)
and [replicate two](./h20-r12-bridge-direct-r2-18628b2-20260813.json.gz), plus
direct/standard [replicate one](./h20-r12-direct-standard-r1-18628b2-20260813.json.gz)
and [replicate two](./h20-r12-direct-standard-r2-18628b2-20260813.json.gz).
Their SHA-256 digests, in that order, are
`f04144bdefe81dc75e5b1be3f00e205a56974a770ece10a1307afc9d89dfb820`,
`84e2a99bc23e1362e69b5c7cc0f97f6f4cec152f34fbeb4a61e25eca5ed3d5fc`,
`bcc3a49247340d066d25fd8f3dbd20efaf49a5d9b13822940806bdabe2628ed9`,
and `3062b73f2a1f2d5277fdb0a8e5de689c42871852dc52e2d95dc5a8c9245223ab`.

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
