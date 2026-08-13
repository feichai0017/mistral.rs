#![deny(unsafe_code)]

use candle_core::cuda::cudarc::driver::{sys::CUevent_flags, CudaStream};
use candle_core::{Device, Tensor};
use half::bf16;
use mistralrs_paged_attn::{paged_attention, OxidePagedDecodeRuntime, OxidePagedDecodeStats};
use oxide_infer::{paged_batch_decode_bf16_reference, Bf16PagedBatchDecodeSpec, PagedKvLayout};
use oxide_infer_cuda::interop::{
    EngineAlgorithm, EngineMetadataValidation, EngineOperator, EngineStreamHandoff,
};
use serde::Serialize;
use std::cmp::Ordering;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CHECKED_TRUSTED_SCHEMA: &str = "mistralrs.oxide-adapter-checked-trusted.v2";
const OXIDE_STANDARD_SCHEMA: &str = "mistralrs.oxide-adapter-provider.v2";
const BRIDGE_DIRECT_SCHEMA: &str = "mistralrs.oxide-adapter-stream-handoff.v1";
const DIRECT_STANDARD_SCHEMA: &str = "mistralrs.oxide-adapter-provider.v3";
const DEFAULT_WARMUPS: usize = 50;
const DEFAULT_ITERATIONS: usize = 500;
const BATCH_SIZE: usize = 16;
const MAX_NUM_PAGES: usize = 128;
const QUERY_HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const PAGE_SIZE: usize = 16;
const PAGES_PER_SEQUENCE: usize = 6;
const LOGICAL_PAGE_COUNT: usize = BATCH_SIZE * PAGES_PER_SEQUENCE;
const STANDARD_KEY_PACK: usize = 16 / std::mem::size_of::<bf16>();
const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const CHECKED_TRUSTED_SCHEDULE: [Validation; 6] = [
    Validation::Checked,
    Validation::Trusted,
    Validation::Trusted,
    Validation::Checked,
    Validation::Checked,
    Validation::Trusted,
];
const OXIDE_STANDARD_SCHEDULE: [Validation; 6] = [
    Validation::Trusted,
    Validation::Standard,
    Validation::Standard,
    Validation::Trusted,
    Validation::Trusted,
    Validation::Standard,
];
const BRIDGE_DIRECT_SCHEDULE: [Validation; 6] = [
    Validation::Trusted,
    Validation::Direct,
    Validation::Direct,
    Validation::Trusted,
    Validation::Trusted,
    Validation::Direct,
];
const DIRECT_STANDARD_SCHEDULE: [Validation; 6] = [
    Validation::Direct,
    Validation::Standard,
    Validation::Standard,
    Validation::Direct,
    Validation::Direct,
    Validation::Standard,
];

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Validation {
    Checked,
    Trusted,
    Direct,
    Standard,
}

impl Validation {
    const fn expected_metadata_validation(self) -> Option<EngineMetadataValidation> {
        match self {
            Self::Checked => Some(EngineMetadataValidation::DeviceChecked),
            Self::Trusted | Self::Direct => Some(EngineMetadataValidation::TrustedByAdapter),
            Self::Standard => None,
        }
    }

    const fn expected_handoff(self) -> Option<EngineStreamHandoff> {
        match self {
            Self::Checked | Self::Trusted => Some(EngineStreamHandoff::ExternalEventBridge),
            Self::Direct => Some(EngineStreamHandoff::ExternalStreamDirect),
            Self::Standard => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ComparisonMode {
    CheckedTrusted,
    OxideStandard,
    BridgeDirect,
    DirectStandard,
}

impl ComparisonMode {
    const fn schema(self) -> &'static str {
        match self {
            Self::CheckedTrusted => CHECKED_TRUSTED_SCHEMA,
            Self::OxideStandard => OXIDE_STANDARD_SCHEMA,
            Self::BridgeDirect => BRIDGE_DIRECT_SCHEMA,
            Self::DirectStandard => DIRECT_STANDARD_SCHEMA,
        }
    }

    const fn schedule(self) -> [Validation; 6] {
        match self {
            Self::CheckedTrusted => CHECKED_TRUSTED_SCHEDULE,
            Self::OxideStandard => OXIDE_STANDARD_SCHEDULE,
            Self::BridgeDirect => BRIDGE_DIRECT_SCHEDULE,
            Self::DirectStandard => DIRECT_STANDARD_SCHEDULE,
        }
    }

    const fn paths(self) -> (Validation, Validation) {
        match self {
            Self::CheckedTrusted => (Validation::Checked, Validation::Trusted),
            Self::OxideStandard => (Validation::Trusted, Validation::Standard),
            Self::BridgeDirect => (Validation::Trusted, Validation::Direct),
            Self::DirectStandard => (Validation::Direct, Validation::Standard),
        }
    }

    const fn includes_standard(self) -> bool {
        matches!(self, Self::OxideStandard | Self::DirectStandard)
    }

    const fn includes_direct(self) -> bool {
        matches!(self, Self::BridgeDirect | Self::DirectStandard)
    }
}

#[derive(Debug)]
struct Args {
    output: PathBuf,
    warmups: usize,
    iterations: usize,
    comparison: ComparisonMode,
    oxide_device_profile: bool,
}

#[derive(Serialize)]
struct Shape {
    batch_size: usize,
    max_num_pages: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    pages_per_sequence: usize,
    logical_page_count: usize,
    context_lengths: Vec<usize>,
    layout: &'static str,
    oxide_algorithm: &'static str,
    standard_algorithm: Option<&'static str>,
}

#[derive(Serialize)]
struct Protocol {
    same_process: bool,
    same_runtime: bool,
    same_tensors: bool,
    same_logical_inputs: bool,
    comparison: ComparisonMode,
    oxide_device_profile_enabled: bool,
    schedule: Vec<Validation>,
    warmups_per_block: usize,
    measured_submissions_per_block: usize,
    cache_layout_materialization_timed: bool,
    host_timing_definition: &'static str,
    device_timing_definition: &'static str,
    oxide_internal_device_timing_definition: &'static str,
    percentile_method: &'static str,
}

#[derive(Serialize)]
struct Distribution {
    count: usize,
    min: f64,
    mean: f64,
    population_stddev: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

#[derive(Serialize)]
struct StatsDelta {
    submitted: u64,
    completed: u64,
    failed: u64,
    profile_enabled: bool,
    device_profile_enabled: bool,
    device_profiled_completions: u64,
    enqueue_host_microseconds_per_submission: f64,
    interop_host_microseconds_per_submission: f64,
    engine_provider_metadata_host_microseconds_per_submission: f64,
    engine_status_readback_host_microseconds_per_submission: f64,
    drain_host_microseconds_per_submission: f64,
    engine_total_device_microseconds_per_submission: f64,
    engine_pre_handoff_device_microseconds_per_submission: f64,
    engine_provider_device_microseconds_per_submission: f64,
    engine_post_handoff_device_microseconds_per_submission: f64,
}

#[derive(Serialize)]
struct BlockRecord {
    block_index: usize,
    validation: Validation,
    metadata_validation: Option<String>,
    stream_handoff: Option<String>,
    output_max_abs: f32,
    host_enqueue_samples_microseconds: Vec<f64>,
    device_window_samples_microseconds: Vec<f64>,
    host_enqueue_microseconds: Distribution,
    device_window_microseconds: Distribution,
    stats: Option<StatsDelta>,
}

#[derive(Serialize)]
struct ValidationSummary {
    validation: Validation,
    block_count: usize,
    host_enqueue_microseconds: Distribution,
    device_window_microseconds: Distribution,
    block_host_enqueue_p50_microseconds: Vec<f64>,
    block_device_window_p50_microseconds: Vec<f64>,
}

#[derive(Serialize)]
struct Comparison {
    left: Validation,
    right: Validation,
    right_over_left_host_enqueue_p50: f64,
    right_over_left_device_window_p50: f64,
    right_minus_left_host_enqueue_p50_microseconds: f64,
    right_minus_left_device_window_p50_microseconds: f64,
}

#[derive(Serialize)]
struct Hardware {
    gpu: String,
    compute_capability: String,
    memory_total_mib: u64,
    driver: String,
}

#[derive(Serialize)]
struct Record {
    schema: &'static str,
    unix_time_seconds: u64,
    source_commit: String,
    source_worktree_clean: bool,
    hardware: Hardware,
    shape: Shape,
    protocol: Protocol,
    blocks: Vec<BlockRecord>,
    summaries: Vec<ValidationSummary>,
    comparison: Comparison,
    excluded_claims: [&'static str; 4],
}

#[derive(Clone, Copy)]
struct StatsSnapshot {
    submitted: u64,
    completed: u64,
    failed: u64,
    device_profiled_completions: u64,
    enqueue_host_nanoseconds: u64,
    interop_host_nanoseconds: u64,
    engine_provider_metadata_host_nanoseconds: u64,
    engine_status_readback_host_nanoseconds: u64,
    drain_host_nanoseconds: u64,
    engine_total_device_nanoseconds: u64,
    engine_pre_handoff_device_nanoseconds: u64,
    engine_provider_device_nanoseconds: u64,
    engine_post_handoff_device_nanoseconds: u64,
}

struct Inputs {
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    standard_key_cache: Tensor,
    standard_value_cache: Tensor,
    page_indptr: Tensor,
    page_indices: Tensor,
    last_page_len: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    context_lengths: Vec<usize>,
    max_context_len: usize,
    expected_output: Vec<bf16>,
}

struct BenchContext<'a> {
    runtime: &'a OxidePagedDecodeRuntime,
    direct_runtime: &'a OxidePagedDecodeRuntime,
    inputs: &'a Inputs,
    stream: &'a CudaStream,
    profile_enabled: bool,
    device_profile_enabled: bool,
}

impl BenchContext<'_> {
    fn runtime(&self, validation: Validation) -> &OxidePagedDecodeRuntime {
        match validation {
            Validation::Direct => self.direct_runtime,
            Validation::Checked | Validation::Trusted | Validation::Standard => self.runtime,
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.oxide_device_profile && args.comparison.includes_direct() {
        return Err("direct-stream comparisons do not support device profiling".into());
    }
    let profile_enabled = args.comparison == ComparisonMode::CheckedTrusted;
    std::env::set_var(
        "MISTRALRS_OXIDE_PROFILE",
        if profile_enabled { "1" } else { "0" },
    );
    std::env::set_var(
        "MISTRALRS_OXIDE_DEVICE_PROFILE",
        if args.oxide_device_profile { "1" } else { "0" },
    );
    std::env::set_var("MISTRALRS_OXIDE_DIRECT_STREAM", "0");
    let device = Device::new_cuda_with_stream(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let runtime = OxidePagedDecodeRuntime::new();
    let direct_runtime = OxidePagedDecodeRuntime::new_direct();
    let inputs = make_inputs(&device)?;
    let schedule = args.comparison.schedule();
    let (left_path, right_path) = args.comparison.paths();
    let context = BenchContext {
        runtime: &runtime,
        direct_runtime: &direct_runtime,
        inputs: &inputs,
        stream: &stream,
        profile_enabled,
        device_profile_enabled: args.oxide_device_profile,
    };

    verify_mode(&context, left_path)?;
    verify_mode(&context, right_path)?;

    let mut blocks = Vec::with_capacity(schedule.len());
    for (block_index, validation) in schedule.into_iter().enumerate() {
        eprintln!(
            "running block {}/{} validation={validation:?}",
            block_index + 1,
            schedule.len()
        );
        blocks.push(run_block(
            block_index,
            validation,
            args.warmups,
            args.iterations,
            &context,
        )?);
    }

    let left = summarize(left_path, &blocks)?;
    let right = summarize(right_path, &blocks)?;
    let comparison = Comparison {
        left: left_path,
        right: right_path,
        right_over_left_host_enqueue_p50: ratio(
            right.host_enqueue_microseconds.p50,
            left.host_enqueue_microseconds.p50,
        )?,
        right_over_left_device_window_p50: ratio(
            right.device_window_microseconds.p50,
            left.device_window_microseconds.p50,
        )?,
        right_minus_left_host_enqueue_p50_microseconds: right.host_enqueue_microseconds.p50
            - left.host_enqueue_microseconds.p50,
        right_minus_left_device_window_p50_microseconds: right.device_window_microseconds.p50
            - left.device_window_microseconds.p50,
    };
    let record = Record {
        schema: args.comparison.schema(),
        unix_time_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        source_commit: git_output(&["rev-parse", "HEAD"] )?,
        source_worktree_clean: git_output(&["status", "--porcelain"] )?.is_empty(),
        hardware: query_hardware()?,
        shape: Shape {
            batch_size: BATCH_SIZE,
            max_num_pages: MAX_NUM_PAGES,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: PAGE_SIZE,
            pages_per_sequence: PAGES_PER_SEQUENCE,
            logical_page_count: LOGICAL_PAGE_COUNT,
            context_lengths: inputs.context_lengths.clone(),
            layout: match args.comparison {
                ComparisonMode::CheckedTrusted | ComparisonMode::BridgeDirect => "HND",
                ComparisonMode::OxideStandard | ComparisonMode::DirectStandard => {
                    "HND and standard packed cache layouts"
                }
            },
            oxide_algorithm: "PagedBatchDecodeTokenParallel8",
            standard_algorithm: args
                .comparison
                .includes_standard()
                .then_some("PagedAttentionV1"),
        },
        protocol: Protocol {
            same_process: true,
            same_runtime: args.comparison == ComparisonMode::CheckedTrusted,
            same_tensors: !args.comparison.includes_standard(),
            same_logical_inputs: true,
            comparison: args.comparison,
            oxide_device_profile_enabled: args.oxide_device_profile,
            schedule: schedule.to_vec(),
            warmups_per_block: args.warmups,
            measured_submissions_per_block: args.iterations,
            cache_layout_materialization_timed: false,
            host_timing_definition: "wall time of one adapter enqueue call, excluding drain",
            device_timing_definition: "CUDA events on the common external stream around one complete provider invocation",
            oxide_internal_device_timing_definition: "ordered CUDA events from external-stream start to provider start, provider end, and external-stream reacquisition",
            percentile_method: "nearest-rank",
        },
        blocks,
        summaries: vec![left, right],
        comparison,
        excluded_claims: [
            "The device window covers a complete provider call and is not a pure-kernel timing boundary.",
            "This single-layer synthetic shape does not establish end-to-end model or serving throughput.",
            "Results apply only to the recorded shape, software, hardware, and timing protocol.",
            "When Oxide internal device profiling is enabled, its extra timing events make the outer Oxide-versus-standard device comparison diagnostic only.",
        ],
    };
    write_json(&args.output, &record)
}

fn parse_args() -> Result<Args> {
    let mut output = None;
    let mut warmups = DEFAULT_WARMUPS;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut comparison = ComparisonMode::CheckedTrusted;
    let mut oxide_device_profile = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--output" => output = Some(PathBuf::from(value)),
            "--warmups" => warmups = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            "--comparison" => {
                comparison = match value.as_str() {
                    "checked-trusted" => ComparisonMode::CheckedTrusted,
                    "oxide-standard" => ComparisonMode::OxideStandard,
                    "bridge-direct" => ComparisonMode::BridgeDirect,
                    "direct-standard" => ComparisonMode::DirectStandard,
                    _ => return Err(format!("unknown comparison {value}").into()),
                }
            }
            "--oxide-device-profile" => {
                oxide_device_profile = match value.as_str() {
                    "0" | "false" => false,
                    "1" | "true" => true,
                    _ => return Err(format!("invalid boolean {value}").into()),
                }
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if warmups == 0 || iterations == 0 {
        return Err("warmups and iterations must be positive".into());
    }
    Ok(Args {
        output: output.ok_or("--output is required")?,
        warmups,
        iterations,
        comparison,
        oxide_device_profile,
    })
}

fn make_inputs(device: &Device) -> Result<Inputs> {
    let spec = Bf16PagedBatchDecodeSpec::new(
        BATCH_SIZE,
        MAX_NUM_PAGES,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        PagedKvLayout::Hnd,
    )?;
    let query_host = deterministic_bf16(spec.query_numel(), 101);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), 211);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), 307);
    let standard_key_host = standard_key_cache(&key_host);
    let standard_value_host = standard_value_cache(&value_host);
    let page_indptr_host = (0..=BATCH_SIZE)
        .map(|index| i32::try_from(index * PAGES_PER_SEQUENCE))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let page_indices_host = (0..LOGICAL_PAGE_COUNT)
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let last_page_len_host = (0..BATCH_SIZE)
        .map(|index| i32::try_from(1 + (index * 3) % PAGE_SIZE))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let context_lens_host = last_page_len_host
        .iter()
        .map(|last_page_len| {
            u32::try_from((PAGES_PER_SEQUENCE - 1) * PAGE_SIZE + *last_page_len as usize)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let max_context_len = context_lens_host
        .iter()
        .copied()
        .max()
        .ok_or("context lengths are empty")? as usize;
    let context_lengths = context_lens_host
        .iter()
        .map(|value| *value as usize)
        .collect();
    let block_tables_host = (0..LOGICAL_PAGE_COUNT)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut expected_output = vec![bf16::ZERO; spec.output_numel()];
    let mut expected_lse = vec![0.0_f32; spec.lse_numel()];
    paged_batch_decode_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;
    Ok(Inputs {
        query: Tensor::from_vec(query_host, (BATCH_SIZE, QUERY_HEADS, HEAD_DIM), device)?,
        key_cache: Tensor::from_vec(
            key_host,
            (MAX_NUM_PAGES, KV_HEADS, PAGE_SIZE, HEAD_DIM),
            device,
        )?,
        value_cache: Tensor::from_vec(
            value_host,
            (MAX_NUM_PAGES, KV_HEADS, PAGE_SIZE, HEAD_DIM),
            device,
        )?,
        standard_key_cache: Tensor::from_vec(
            standard_key_host,
            (
                MAX_NUM_PAGES,
                KV_HEADS,
                HEAD_DIM / STANDARD_KEY_PACK,
                PAGE_SIZE,
                STANDARD_KEY_PACK,
            ),
            device,
        )?,
        standard_value_cache: Tensor::from_vec(
            standard_value_host,
            (MAX_NUM_PAGES, KV_HEADS, HEAD_DIM, PAGE_SIZE),
            device,
        )?,
        page_indptr: Tensor::from_vec(page_indptr_host, BATCH_SIZE + 1, device)?,
        page_indices: Tensor::from_vec(page_indices_host, LOGICAL_PAGE_COUNT, device)?,
        last_page_len: Tensor::from_vec(last_page_len_host, BATCH_SIZE, device)?,
        block_tables: Tensor::from_vec(
            block_tables_host,
            (BATCH_SIZE, PAGES_PER_SEQUENCE),
            device,
        )?,
        context_lens: Tensor::from_vec(context_lens_host, BATCH_SIZE, device)?,
        context_lengths,
        max_context_len,
        expected_output,
    })
}

fn standard_key_cache(hnd: &[bf16]) -> Vec<bf16> {
    let mut standard = vec![bf16::ZERO; hnd.len()];
    for block in 0..MAX_NUM_PAGES {
        for head in 0..KV_HEADS {
            for token in 0..PAGE_SIZE {
                for offset in 0..HEAD_DIM {
                    let source =
                        (((block * KV_HEADS + head) * PAGE_SIZE + token) * HEAD_DIM) + offset;
                    let target = ((((block * KV_HEADS + head) * (HEAD_DIM / STANDARD_KEY_PACK)
                        + offset / STANDARD_KEY_PACK)
                        * PAGE_SIZE
                        + token)
                        * STANDARD_KEY_PACK)
                        + offset % STANDARD_KEY_PACK;
                    standard[target] = hnd[source];
                }
            }
        }
    }
    standard
}

fn standard_value_cache(hnd: &[bf16]) -> Vec<bf16> {
    let mut standard = vec![bf16::ZERO; hnd.len()];
    for block in 0..MAX_NUM_PAGES {
        for head in 0..KV_HEADS {
            for token in 0..PAGE_SIZE {
                for offset in 0..HEAD_DIM {
                    let source =
                        (((block * KV_HEADS + head) * PAGE_SIZE + token) * HEAD_DIM) + offset;
                    let target =
                        (((block * KV_HEADS + head) * HEAD_DIM + offset) * PAGE_SIZE) + token;
                    standard[target] = hnd[source];
                }
            }
        }
    }
    standard
}

fn run_block(
    block_index: usize,
    validation: Validation,
    warmups: usize,
    iterations: usize,
    context: &BenchContext<'_>,
) -> Result<BlockRecord> {
    let runtime = context.runtime(validation);
    for _ in 0..warmups {
        let output = enqueue(runtime, context.inputs, validation)?;
        settle_submission(runtime, validation, context.stream)?;
        drop(output);
    }
    let before = stats_snapshot(runtime.stats()?);
    let start_event = context
        .stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end_event = context
        .stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut host_enqueue_microseconds = Vec::with_capacity(iterations);
    let mut device_window_microseconds = Vec::with_capacity(iterations);
    let mut output_max_abs = 0.0_f32;
    for iteration in 0..iterations {
        start_event.record(context.stream)?;
        let host_started = Instant::now();
        let output = enqueue(runtime, context.inputs, validation)?;
        host_enqueue_microseconds.push(host_started.elapsed().as_secs_f64() * 1_000_000.0);
        end_event.record(context.stream)?;
        if validation != Validation::Standard && runtime.drain()? != 1 {
            return Err("measured Oxide drain did not settle one submission".into());
        }
        device_window_microseconds.push(f64::from(start_event.elapsed_ms(&end_event)?) * 1_000.0);
        if iteration == 0 {
            output_max_abs = compare_output(&output, &context.inputs.expected_output)?;
        }
        drop(output);
    }
    let actual = runtime.stats()?;
    validate_stats(
        actual,
        before,
        validation,
        iterations,
        context.profile_enabled,
        context.device_profile_enabled,
    )?;
    let host_enqueue_distribution = distribution(host_enqueue_microseconds.iter().copied())?;
    let device_window_distribution = distribution(device_window_microseconds.iter().copied())?;
    Ok(BlockRecord {
        block_index,
        validation,
        metadata_validation: validation
            .expected_metadata_validation()
            .map(|value| format!("{value:?}")),
        stream_handoff: validation
            .expected_handoff()
            .map(|value| format!("{value:?}")),
        output_max_abs,
        host_enqueue_samples_microseconds: host_enqueue_microseconds,
        device_window_samples_microseconds: device_window_microseconds,
        host_enqueue_microseconds: host_enqueue_distribution,
        device_window_microseconds: device_window_distribution,
        stats: (validation != Validation::Standard)
            .then(|| stats_delta(before, stats_snapshot(actual), actual.profile_enabled()))
            .transpose()?,
    })
}

fn verify_mode(context: &BenchContext<'_>, validation: Validation) -> Result<()> {
    let runtime = context.runtime(validation);
    let before = stats_snapshot(runtime.stats()?);
    let output = enqueue(runtime, context.inputs, validation)?;
    settle_submission(runtime, validation, context.stream)?;
    compare_output(&output, &context.inputs.expected_output)?;
    validate_stats(
        runtime.stats()?,
        before,
        validation,
        1,
        context.profile_enabled,
        context.device_profile_enabled,
    )
}

fn settle_submission(
    runtime: &OxidePagedDecodeRuntime,
    validation: Validation,
    stream: &CudaStream,
) -> Result<()> {
    if validation == Validation::Standard {
        stream.synchronize()?;
    } else if runtime.drain()? != 1 {
        return Err("Oxide drain did not settle one submission".into());
    }
    Ok(())
}

#[allow(unsafe_code)]
fn enqueue(
    runtime: &OxidePagedDecodeRuntime,
    inputs: &Inputs,
    validation: Validation,
) -> candle_core::Result<Tensor> {
    match validation {
        Validation::Checked => {
            // SAFETY: the benchmark uses one thread, one device, one ordinary stream, and immutable metadata.
            unsafe {
                runtime.enqueue_paged_decode(
                    &inputs.query,
                    &inputs.key_cache,
                    &inputs.value_cache,
                    &inputs.page_indptr,
                    &inputs.page_indices,
                    LOGICAL_PAGE_COUNT,
                    &inputs.last_page_len,
                )
            }
        }
        Validation::Trusted | Validation::Direct => {
            // SAFETY: the benchmark additionally keeps the valid CSR metadata immutable for every submission.
            unsafe {
                runtime.enqueue_trusted_paged_decode(
                    &inputs.query,
                    &inputs.key_cache,
                    &inputs.value_cache,
                    &inputs.page_indptr,
                    &inputs.page_indices,
                    LOGICAL_PAGE_COUNT,
                    &inputs.last_page_len,
                )
            }
        }
        Validation::Standard => paged_attention(
            &inputs.query,
            None,
            None,
            &inputs.standard_key_cache,
            &inputs.standard_value_cache,
            &inputs.block_tables,
            &inputs.context_lens,
            None,
            inputs.max_context_len,
            (HEAD_DIM as f32).sqrt().recip(),
            1.0,
            None,
        ),
    }
}

fn validate_stats(
    actual: OxidePagedDecodeStats,
    before: StatsSnapshot,
    validation: Validation,
    submissions: usize,
    profile_enabled: bool,
    device_profile_enabled: bool,
) -> Result<()> {
    let expected = u64::try_from(submissions)?;
    let after = stats_snapshot(actual);
    if validation == Validation::Standard {
        if after.submitted != before.submitted
            || after.completed != before.completed
            || after.failed != before.failed
            || after.device_profiled_completions != before.device_profiled_completions
        {
            return Err(format!("standard path changed Oxide stats: {actual:?}").into());
        }
        return Ok(());
    }
    if after.submitted.checked_sub(before.submitted) != Some(expected)
        || after.completed.checked_sub(before.completed) != Some(expected)
        || after.failed.checked_sub(before.failed) != Some(0)
        || actual.last_operator() != Some(EngineOperator::Bf16PagedBatchDecode)
        || actual.last_layout() != Some(PagedKvLayout::Hnd)
        || actual.last_algorithm() != Some(EngineAlgorithm::PagedBatchDecodeTokenParallel8)
        || actual.last_metadata_validation() != validation.expected_metadata_validation()
        || actual.last_stream_handoff() != validation.expected_handoff()
        || !actual.adapter_zero_copy()
        || actual.external_regions() != 9
        || actual.adapter_device_to_device_copies() != 0
        || actual.profile_enabled() != profile_enabled
        || actual.device_profile_enabled() != device_profile_enabled
        || after
            .device_profiled_completions
            .checked_sub(before.device_profiled_completions)
            != Some(if device_profile_enabled { expected } else { 0 })
    {
        return Err(format!("adapter stats mismatch for {validation:?}: {actual:?}").into());
    }
    Ok(())
}

fn stats_snapshot(stats: OxidePagedDecodeStats) -> StatsSnapshot {
    StatsSnapshot {
        submitted: stats.submitted(),
        completed: stats.completed(),
        failed: stats.failed(),
        device_profiled_completions: stats.device_profiled_completions(),
        enqueue_host_nanoseconds: stats.enqueue_host_nanoseconds(),
        interop_host_nanoseconds: stats.interop_host_nanoseconds(),
        engine_provider_metadata_host_nanoseconds: stats
            .engine_provider_metadata_host_nanoseconds(),
        engine_status_readback_host_nanoseconds: stats.engine_status_readback_host_nanoseconds(),
        drain_host_nanoseconds: stats.drain_host_nanoseconds(),
        engine_total_device_nanoseconds: stats.engine_total_device_nanoseconds(),
        engine_pre_handoff_device_nanoseconds: stats.engine_pre_handoff_device_nanoseconds(),
        engine_provider_device_nanoseconds: stats.engine_provider_device_nanoseconds(),
        engine_post_handoff_device_nanoseconds: stats.engine_post_handoff_device_nanoseconds(),
    }
}

fn stats_delta(
    before: StatsSnapshot,
    after: StatsSnapshot,
    profile_enabled: bool,
) -> Result<StatsDelta> {
    let submitted = checked_delta(after.submitted, before.submitted, "submitted")?;
    if submitted == 0 {
        return Err("stats delta has no submissions".into());
    }
    Ok(StatsDelta {
        submitted,
        completed: checked_delta(after.completed, before.completed, "completed")?,
        failed: checked_delta(after.failed, before.failed, "failed")?,
        profile_enabled,
        device_profile_enabled: after.device_profiled_completions
            > before.device_profiled_completions,
        device_profiled_completions: checked_delta(
            after.device_profiled_completions,
            before.device_profiled_completions,
            "device profiled completions",
        )?,
        enqueue_host_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.enqueue_host_nanoseconds,
                before.enqueue_host_nanoseconds,
                "enqueue host nanoseconds",
            )?,
            submitted,
        ),
        interop_host_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.interop_host_nanoseconds,
                before.interop_host_nanoseconds,
                "interop host nanoseconds",
            )?,
            submitted,
        ),
        engine_provider_metadata_host_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_provider_metadata_host_nanoseconds,
                before.engine_provider_metadata_host_nanoseconds,
                "metadata host nanoseconds",
            )?,
            submitted,
        ),
        engine_status_readback_host_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_status_readback_host_nanoseconds,
                before.engine_status_readback_host_nanoseconds,
                "status host nanoseconds",
            )?,
            submitted,
        ),
        drain_host_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.drain_host_nanoseconds,
                before.drain_host_nanoseconds,
                "drain host nanoseconds",
            )?,
            submitted,
        ),
        engine_total_device_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_total_device_nanoseconds,
                before.engine_total_device_nanoseconds,
                "total device nanoseconds",
            )?,
            submitted,
        ),
        engine_pre_handoff_device_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_pre_handoff_device_nanoseconds,
                before.engine_pre_handoff_device_nanoseconds,
                "pre-handoff device nanoseconds",
            )?,
            submitted,
        ),
        engine_provider_device_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_provider_device_nanoseconds,
                before.engine_provider_device_nanoseconds,
                "provider device nanoseconds",
            )?,
            submitted,
        ),
        engine_post_handoff_device_microseconds_per_submission: per_submission_microseconds(
            checked_delta(
                after.engine_post_handoff_device_nanoseconds,
                before.engine_post_handoff_device_nanoseconds,
                "post-handoff device nanoseconds",
            )?,
            submitted,
        ),
    })
}

fn checked_delta(after: u64, before: u64, name: &str) -> Result<u64> {
    after
        .checked_sub(before)
        .ok_or_else(|| format!("{name} counter regressed").into())
}

fn per_submission_microseconds(nanoseconds: u64, submissions: u64) -> f64 {
    nanoseconds as f64 / submissions as f64 / 1_000.0
}

fn compare_output(output: &Tensor, expected: &[bf16]) -> Result<f32> {
    let actual = output.flatten_all()?.to_vec1::<bf16>()?;
    if actual.len() != expected.len() {
        return Err("output length mismatch".into());
    }
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    if !max_abs.is_finite() || max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(
            format!("output max abs {max_abs:.9e} exceeds {OUTPUT_MAX_ABS_LIMIT:.9e}").into(),
        );
    }
    Ok(max_abs)
}

fn summarize(validation: Validation, blocks: &[BlockRecord]) -> Result<ValidationSummary> {
    let selected = blocks
        .iter()
        .filter(|block| block.validation == validation)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!("schedule has no {validation:?} blocks").into());
    }
    Ok(ValidationSummary {
        validation,
        block_count: selected.len(),
        host_enqueue_microseconds: distribution(
            selected
                .iter()
                .flat_map(|block| block.host_enqueue_samples_microseconds.iter().copied()),
        )?,
        device_window_microseconds: distribution(
            selected
                .iter()
                .flat_map(|block| block.device_window_samples_microseconds.iter().copied()),
        )?,
        block_host_enqueue_p50_microseconds: selected
            .iter()
            .map(|block| block.host_enqueue_microseconds.p50)
            .collect(),
        block_device_window_p50_microseconds: selected
            .iter()
            .map(|block| block.device_window_microseconds.p50)
            .collect(),
    })
}

fn distribution(values: impl IntoIterator<Item = f64>) -> Result<Distribution> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("distribution requires finite observations".into());
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    Ok(Distribution {
        count,
        min: values[0],
        mean,
        population_stddev: variance.sqrt(),
        p50: nearest_rank(&values, 50)?,
        p95: nearest_rank(&values, 95)?,
        max: values[count - 1],
    })
}

fn nearest_rank(sorted: &[f64], percentile: usize) -> Result<f64> {
    if sorted.is_empty() || !(1..=100).contains(&percentile) {
        return Err("invalid nearest-rank input".into());
    }
    let rank = (percentile * sorted.len()).div_ceil(100);
    Ok(sorted[rank - 1])
}

fn ratio(numerator: f64, denominator: f64) -> Result<f64> {
    if denominator <= 0.0 {
        return Err("ratio denominator must be positive".into());
    }
    Ok(numerator / denominator)
}

fn deterministic_bf16(len: usize, seed: u64) -> Vec<bf16> {
    (0..len)
        .map(|index| {
            let value = (index as u64)
                .wrapping_mul(73)
                .wrapping_add(seed.wrapping_mul(41))
                % 257;
            bf16::from_f32((value as f32 - 128.0) / 256.0)
        })
        .collect()
}

fn query_hardware() -> Result<Hardware> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err("nvidia-smi hardware query failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let fields = text
        .lines()
        .next()
        .ok_or("nvidia-smi returned no GPUs")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("unexpected nvidia-smi hardware output".into());
    }
    Ok(Hardware {
        gpu: fields[0].to_string(),
        compute_capability: fields[1].to_string(),
        memory_total_mib: fields[2].parse()?,
        driver: fields[3].to_string(),
    })
}

fn git_output(args: &[&str]) -> Result<String> {
    let output = Command::new("git").args(args).output()?;
    if !output.status.success() {
        return Err("git command failed".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn write_json(path: &Path, record: &Record) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        distribution, nearest_rank, Validation, CHECKED_TRUSTED_SCHEDULE, OXIDE_STANDARD_SCHEDULE,
    };

    #[test]
    fn checked_trusted_schedule_is_position_balanced() {
        assert_eq!(
            CHECKED_TRUSTED_SCHEDULE
                .iter()
                .filter(|validation| **validation == Validation::Checked)
                .count(),
            3
        );
        assert_eq!(
            CHECKED_TRUSTED_SCHEDULE
                .iter()
                .filter(|validation| **validation == Validation::Trusted)
                .count(),
            3
        );
        assert_eq!(CHECKED_TRUSTED_SCHEDULE[0], CHECKED_TRUSTED_SCHEDULE[4]);
        assert_eq!(CHECKED_TRUSTED_SCHEDULE[1], CHECKED_TRUSTED_SCHEDULE[2]);
    }

    #[test]
    fn oxide_standard_schedule_is_position_balanced() {
        assert_eq!(
            OXIDE_STANDARD_SCHEDULE
                .iter()
                .filter(|validation| **validation == Validation::Trusted)
                .count(),
            3
        );
        assert_eq!(
            OXIDE_STANDARD_SCHEDULE
                .iter()
                .filter(|validation| **validation == Validation::Standard)
                .count(),
            3
        );
        assert_eq!(OXIDE_STANDARD_SCHEDULE[0], OXIDE_STANDARD_SCHEDULE[4]);
        assert_eq!(OXIDE_STANDARD_SCHEDULE[1], OXIDE_STANDARD_SCHEDULE[2]);
    }

    #[test]
    fn nearest_rank_uses_one_based_ceiling() {
        let values = (1..=20).map(f64::from).collect::<Vec<_>>();
        assert_eq!(nearest_rank(&values, 50).unwrap(), 10.0);
        assert_eq!(nearest_rank(&values, 95).unwrap(), 19.0);
    }

    #[test]
    fn distribution_reports_population_statistics() {
        let stats = distribution([1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.p50, 2.0);
        assert_eq!(stats.p95, 4.0);
    }
}
