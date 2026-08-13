use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::{Args, Parser, Subcommand, ValueEnum};
use mistralrs::{
    DeviceMapSetting, MemoryGpuConfig, Model, ModelDType, PagedAttentionMetaBuilder,
    RequestBuilder, Response, TextMessageRole, TextMessages, TextModelBuilder, Usage,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "mistralrs.oxide-paged-decode-steady-state.v1";
const PROMPT: &str =
    "Output the integers from 1 through 200 in ascending order, separated by one space. Output only the integers.";
const DEFAULT_WARMUPS: usize = 5;
const DEFAULT_ITERATIONS: usize = 20;
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 64;
const PAGE_SIZE: usize = 16;
const CONTEXT_SIZE: usize = 4096;
const MAX_NUM_SEQUENCES: usize = 1;
const GPU_QUERY_FIELDS: &str = "name,compute_cap,memory.total,driver_version";
const BYTES_PER_MIB: f64 = 1_048_576.0;
const DEFAULT_SCHEDULE: [Provider; 6] = [
    Provider::Oxide,
    Provider::Baseline,
    Provider::Baseline,
    Provider::Oxide,
    Provider::Oxide,
    Provider::Baseline,
];

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Mode,
}

#[derive(Debug, Subcommand)]
enum Mode {
    Suite(SuiteArgs),
    #[command(hide = true)]
    Worker(WorkerArgs),
}

#[derive(Debug, Args)]
struct SuiteArgs {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    model_name: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = DEFAULT_WARMUPS)]
    warmups: usize,
    #[arg(long, default_value_t = DEFAULT_ITERATIONS)]
    iterations: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_TOKENS)]
    max_output_tokens: usize,
}

#[derive(Debug, Args)]
struct WorkerArgs {
    #[arg(long)]
    provider: Provider,
    #[arg(long)]
    block_index: usize,
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    model_name: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    warmups: usize,
    #[arg(long)]
    iterations: usize,
    #[arg(long)]
    max_output_tokens: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Provider {
    Oxide,
    Baseline,
}

impl Provider {
    const fn label(self) -> &'static str {
        match self {
            Self::Oxide => "oxide",
            Self::Baseline => "baseline",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Measurement {
    iteration: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    ttft_ms: f64,
    tpot_ms: f64,
    end_to_end_ms: f64,
    decode_tokens_per_second: f64,
    completion_tokens_per_second: f64,
    engine_prompt_ms: f64,
    engine_completion_ms: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct OxideStatsDelta {
    submitted: u64,
    completed: u64,
    failed: u64,
    last_operator: String,
    last_layout: String,
    last_algorithm: String,
    adapter_zero_copy: bool,
    external_regions: usize,
    adapter_device_to_device_copies: usize,
    profile_enabled: bool,
    enqueue_host_nanoseconds: u64,
    enqueue_host_microseconds_per_operator: Option<f64>,
    drain_host_nanoseconds: u64,
    drain_calls: u64,
    drain_host_microseconds_per_forward: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkerRecord {
    provider: Provider,
    block_index: usize,
    model_name: String,
    warmups: usize,
    iterations: usize,
    max_output_tokens: usize,
    response_text: String,
    device_used_memory_baseline_mib: f64,
    device_memory_delta_after_warmup_mib: f64,
    device_memory_delta_after_measurements_mib: f64,
    oxide_stats_delta: Option<OxideStatsDelta>,
    measurements: Vec<Measurement>,
}

#[derive(Debug, Serialize)]
struct Distribution {
    count: usize,
    min: f64,
    mean: f64,
    population_stddev: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct ProviderSummary {
    provider: Provider,
    block_count: usize,
    sample_count: usize,
    ttft_ms: Distribution,
    tpot_ms: Distribution,
    end_to_end_ms: Distribution,
    decode_tokens_per_second: Distribution,
    completion_tokens_per_second: Distribution,
    device_memory_delta_after_warmup_mib: Distribution,
    device_memory_delta_after_measurements_mib: Distribution,
    block_tpot_p50_ms: Vec<f64>,
    block_decode_tokens_per_second_p50: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Comparison {
    response_text_equal: bool,
    oxide_over_baseline_ttft_p50: f64,
    oxide_over_baseline_tpot_p50: f64,
    oxide_over_baseline_end_to_end_p50: f64,
    oxide_over_baseline_decode_tps_p50: f64,
    oxide_minus_baseline_device_memory_delta_after_warmup_mib: f64,
}

#[derive(Debug, Serialize)]
struct Hardware {
    gpu: String,
    compute_capability: String,
    memory_total_mib: u64,
    driver: String,
}

#[derive(Debug, Serialize)]
struct Protocol {
    schedule: Vec<Provider>,
    process_isolation_per_block: bool,
    warmups_per_block: usize,
    measured_requests_per_block: usize,
    max_output_tokens: usize,
    temperature: f64,
    streaming: bool,
    prefix_cache_enabled: bool,
    cuda_graph_enabled: bool,
    max_num_sequences: usize,
    page_size: usize,
    context_size: usize,
    percentile_method: &'static str,
    ttft_definition: &'static str,
    tpot_definition: &'static str,
    memory_definition: &'static str,
}

#[derive(Debug, Serialize)]
struct SuiteRecord {
    schema: &'static str,
    unix_time_seconds: u64,
    source_commit: String,
    source_worktree_clean: bool,
    model_name: String,
    hardware: Hardware,
    prompt: &'static str,
    protocol: Protocol,
    blocks: Vec<WorkerRecord>,
    providers: Vec<ProviderSummary>,
    comparison: Comparison,
    excluded_claims: [&'static str; 4],
}

#[derive(Clone, Copy)]
struct StatsSnapshot {
    submitted: u64,
    completed: u64,
    failed: u64,
    enqueue_host_nanoseconds: u64,
    drain_host_nanoseconds: u64,
    drain_calls: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Mode::Suite(args) => run_suite(args),
        Mode::Worker(args) => run_worker(args).await,
    }
}

fn run_suite(args: SuiteArgs) -> Result<()> {
    validate_counts(args.warmups, args.iterations, args.max_output_tokens)?;
    let current_exe = std::env::current_exe().context("failed to locate benchmark binary")?;
    let temp_dir =
        std::env::temp_dir().join(format!("mistralrs-oxide-bench-{}", std::process::id()));
    fs::create_dir(&temp_dir)
        .with_context(|| format!("failed to create {}", temp_dir.display()))?;

    let suite_result = collect_suite(&args, &current_exe, &temp_dir);
    let cleanup_result = fs::remove_dir_all(&temp_dir)
        .with_context(|| format!("failed to remove {}", temp_dir.display()));
    let record = suite_result?;
    cleanup_result?;
    write_json(&args.output, &record)
}

fn collect_suite(args: &SuiteArgs, current_exe: &Path, temp_dir: &Path) -> Result<SuiteRecord> {
    let mut blocks = Vec::with_capacity(DEFAULT_SCHEDULE.len());
    for (block_index, provider) in DEFAULT_SCHEDULE.into_iter().enumerate() {
        eprintln!(
            "running block {}/{}, provider={}",
            block_index + 1,
            DEFAULT_SCHEDULE.len(),
            provider.label()
        );
        let block_output = temp_dir.join(format!("block-{block_index}.json"));
        let status = ProcessCommand::new(current_exe)
            .arg("worker")
            .arg("--provider")
            .arg(provider.label())
            .arg("--block-index")
            .arg(block_index.to_string())
            .arg("--model-path")
            .arg(&args.model_path)
            .arg("--model-name")
            .arg(&args.model_name)
            .arg("--output")
            .arg(&block_output)
            .arg("--warmups")
            .arg(args.warmups.to_string())
            .arg("--iterations")
            .arg(args.iterations.to_string())
            .arg("--max-output-tokens")
            .arg(args.max_output_tokens.to_string())
            .stdin(Stdio::null())
            .status()
            .with_context(|| format!("failed to start {provider:?} block {block_index}"))?;
        if !status.success() {
            bail!("{provider:?} block {block_index} failed with {status}");
        }
        let bytes = fs::read(&block_output)
            .with_context(|| format!("failed to read {}", block_output.display()))?;
        let block: WorkerRecord = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid worker record in {}", block_output.display()))?;
        if block.provider != provider || block.block_index != block_index {
            bail!("worker record identity mismatch for block {block_index}");
        }
        eprintln!(
            "completed block {}/{}",
            block_index + 1,
            DEFAULT_SCHEDULE.len()
        );
        blocks.push(block);
    }

    let first_response = &blocks[0].response_text;
    let response_text_equal = blocks
        .iter()
        .all(|block| block.response_text == *first_response);
    if !response_text_equal {
        bail!("providers did not produce one stable response text");
    }

    let oxide = summarize_provider(Provider::Oxide, &blocks)?;
    let baseline = summarize_provider(Provider::Baseline, &blocks)?;
    let comparison = Comparison {
        response_text_equal,
        oxide_over_baseline_ttft_p50: ratio(oxide.ttft_ms.p50, baseline.ttft_ms.p50)?,
        oxide_over_baseline_tpot_p50: ratio(oxide.tpot_ms.p50, baseline.tpot_ms.p50)?,
        oxide_over_baseline_end_to_end_p50: ratio(
            oxide.end_to_end_ms.p50,
            baseline.end_to_end_ms.p50,
        )?,
        oxide_over_baseline_decode_tps_p50: ratio(
            oxide.decode_tokens_per_second.p50,
            baseline.decode_tokens_per_second.p50,
        )?,
        oxide_minus_baseline_device_memory_delta_after_warmup_mib: oxide
            .device_memory_delta_after_warmup_mib
            .p50
            - baseline.device_memory_delta_after_warmup_mib.p50,
    };

    Ok(SuiteRecord {
        schema: SCHEMA,
        unix_time_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock predates Unix epoch")?
            .as_secs(),
        source_commit: git_output(&["rev-parse", "HEAD"])?,
        source_worktree_clean: git_worktree_clean()?,
        model_name: args.model_name.clone(),
        hardware: query_hardware()?,
        prompt: PROMPT,
        protocol: Protocol {
            schedule: DEFAULT_SCHEDULE.to_vec(),
            process_isolation_per_block: true,
            warmups_per_block: args.warmups,
            measured_requests_per_block: args.iterations,
            max_output_tokens: args.max_output_tokens,
            temperature: 0.0,
            streaming: true,
            prefix_cache_enabled: false,
            cuda_graph_enabled: false,
            max_num_sequences: MAX_NUM_SEQUENCES,
            page_size: PAGE_SIZE,
            context_size: CONTEXT_SIZE,
            percentile_method: "nearest-rank",
            ttft_definition: "request submission to first non-empty generated content",
            tpot_definition: "(stream completion - TTFT) / (completion tokens - 1)",
            memory_definition: "CUDA driver device-used memory delta from the post-context baseline, sampled after warmup and after measured requests",
        },
        blocks,
        providers: vec![oxide, baseline],
        comparison,
        excluded_claims: [
            "Results apply only to the recorded model, request shape, software, and hardware.",
            "Single-request measurements do not establish concurrent serving throughput.",
            "Device memory deltas are steady-state observations, not process-private, allocator, or system-wide peaks.",
            "The benchmark does not qualify CUDA Graphs, prefix caching, speculative decode, tensor parallelism, multiple GPUs, or multiple streams.",
        ],
    })
}

async fn run_worker(args: WorkerArgs) -> Result<()> {
    validate_counts(args.warmups, args.iterations, args.max_output_tokens)?;
    configure_provider(args.provider);
    let device = Device::new_cuda_with_stream(0)?;
    let device_used_memory_baseline_mib = query_device_used_memory_mib(&device)?;
    let memory_device = device.clone();
    let model_path = args
        .model_path
        .to_str()
        .context("model path is not valid UTF-8")?;
    let model = TextModelBuilder::new(model_path)
        .with_dtype(ModelDType::BF16)
        .with_device(device)
        .with_device_mapping(DeviceMapSetting::dummy())
        .with_max_num_seqs(MAX_NUM_SEQUENCES)
        .with_prefix_cache_n(None)
        .with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(PAGE_SIZE)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(CONTEXT_SIZE))
                .build()?,
        )
        .build()
        .await?;

    let mut response_text = None;
    for _ in 0..args.warmups {
        let observation = measure_request(&model, args.max_output_tokens).await?;
        verify_response(&mut response_text, &observation.response_text)?;
    }
    let device_used_memory_after_warmup_mib = query_device_used_memory_mib(&memory_device)?;
    let device_memory_delta_after_warmup_mib = memory_delta_mib(
        device_used_memory_after_warmup_mib,
        device_used_memory_baseline_mib,
    )?;
    let stats_before = oxide_stats_snapshot(&model, args.provider).await?;

    let mut measurements = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let observation = measure_request(&model, args.max_output_tokens).await?;
        verify_response(&mut response_text, &observation.response_text)?;
        measurements.push(observation.into_measurement(iteration));
    }
    let stats_after = oxide_stats_snapshot(&model, args.provider).await?;
    let device_used_memory_after_measurements_mib = query_device_used_memory_mib(&memory_device)?;
    let device_memory_delta_after_measurements_mib = memory_delta_mib(
        device_used_memory_after_measurements_mib,
        device_used_memory_baseline_mib,
    )?;
    let oxide_stats_delta =
        oxide_stats_delta(&model, args.provider, stats_before, stats_after).await?;

    write_json(
        &args.output,
        &WorkerRecord {
            provider: args.provider,
            block_index: args.block_index,
            model_name: args.model_name,
            warmups: args.warmups,
            iterations: args.iterations,
            max_output_tokens: args.max_output_tokens,
            response_text: response_text.context("worker produced no response")?,
            device_used_memory_baseline_mib,
            device_memory_delta_after_warmup_mib,
            device_memory_delta_after_measurements_mib,
            oxide_stats_delta,
            measurements,
        },
    )
}

struct Observation {
    response_text: String,
    usage: Usage,
    ttft: Duration,
    end_to_end: Duration,
}

impl Observation {
    fn into_measurement(self, iteration: usize) -> Measurement {
        let ttft_seconds = self.ttft.as_secs_f64();
        let end_to_end_seconds = self.end_to_end.as_secs_f64();
        let decode_seconds = end_to_end_seconds - ttft_seconds;
        let decode_tokens = self.usage.completion_tokens - 1;
        Measurement {
            iteration,
            prompt_tokens: self.usage.prompt_tokens,
            completion_tokens: self.usage.completion_tokens,
            ttft_ms: ttft_seconds * 1000.0,
            tpot_ms: decode_seconds * 1000.0 / decode_tokens as f64,
            end_to_end_ms: end_to_end_seconds * 1000.0,
            decode_tokens_per_second: decode_tokens as f64 / decode_seconds,
            completion_tokens_per_second: self.usage.completion_tokens as f64 / end_to_end_seconds,
            engine_prompt_ms: f64::from(self.usage.total_prompt_time_sec) * 1000.0,
            engine_completion_ms: f64::from(self.usage.total_completion_time_sec) * 1000.0,
        }
    }
}

async fn measure_request(model: &Model, max_output_tokens: usize) -> Result<Observation> {
    let messages = TextMessages::new().add_message(TextMessageRole::User, PROMPT);
    let request = RequestBuilder::from(messages)
        .set_sampler_temperature(0.0)
        .set_sampler_max_len(max_output_tokens);
    let start = Instant::now();
    let mut stream = model.stream_chat_request(request).await?;
    let mut first_token = None;
    let mut response_text = String::new();
    let mut final_usage = None;

    while let Some(response) = stream.next().await {
        match response {
            Response::Chunk(chunk) => {
                if let Some(usage) = chunk.usage {
                    final_usage = Some(usage);
                }
                if let Some(content) = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.as_deref())
                {
                    if !content.is_empty() && first_token.is_none() {
                        first_token = Some(start.elapsed());
                    }
                    response_text.push_str(content);
                }
            }
            Response::InternalError(error) | Response::ValidationError(error) => {
                return Err(anyhow::anyhow!(error.to_string()));
            }
            Response::ModelError(message, _) => bail!("model error: {message}"),
            _ => bail!("unexpected response variant during streaming benchmark"),
        }
    }
    let end_to_end = start.elapsed();
    let usage = final_usage.context("stream ended without final usage")?;
    let ttft = first_token.context("stream ended without generated content")?;
    if usage.completion_tokens != max_output_tokens {
        bail!(
            "expected {max_output_tokens} completion tokens, got {}",
            usage.completion_tokens
        );
    }
    if usage.completion_tokens < 2 || end_to_end <= ttft {
        bail!("request did not produce a measurable decode interval");
    }
    Ok(Observation {
        response_text,
        usage,
        ttft,
        end_to_end,
    })
}

fn verify_response(expected: &mut Option<String>, actual: &str) -> Result<()> {
    if let Some(expected) = expected {
        if expected != actual {
            bail!("deterministic response changed within one benchmark block");
        }
    } else {
        *expected = Some(actual.to_string());
    }
    Ok(())
}

fn configure_provider(provider: Provider) {
    let oxide = if provider == Provider::Oxide {
        "1"
    } else {
        "0"
    };
    std::env::set_var("MISTRALRS_OXIDE_INFER", oxide);
    std::env::set_var("MISTRALRS_FLASHINFER_DECODE", "0");
    std::env::set_var("MISTRALRS_CUDA_GRAPHS", "0");
}

async fn oxide_stats_snapshot(model: &Model, provider: Provider) -> Result<Option<StatsSnapshot>> {
    let stats = model.oxide_paged_decode_stats().await?;
    match (provider, stats) {
        (Provider::Oxide, Some(stats)) => Ok(Some(StatsSnapshot {
            submitted: stats.submitted(),
            completed: stats.completed(),
            failed: stats.failed(),
            enqueue_host_nanoseconds: stats.enqueue_host_nanoseconds(),
            drain_host_nanoseconds: stats.drain_host_nanoseconds(),
            drain_calls: stats.drain_calls(),
        })),
        (Provider::Oxide, None) => bail!("Oxide worker has no paged-decode runtime"),
        (Provider::Baseline, None) => Ok(None),
        (Provider::Baseline, Some(_)) => bail!("baseline worker unexpectedly has an Oxide runtime"),
    }
}

async fn oxide_stats_delta(
    model: &Model,
    provider: Provider,
    before: Option<StatsSnapshot>,
    after: Option<StatsSnapshot>,
) -> Result<Option<OxideStatsDelta>> {
    if provider == Provider::Baseline {
        return Ok(None);
    }
    let before = before.context("missing Oxide stats before measurements")?;
    let after = after.context("missing Oxide stats after measurements")?;
    let stats = model
        .oxide_paged_decode_stats()
        .await?
        .context("missing final Oxide stats")?;
    let submitted = after
        .submitted
        .checked_sub(before.submitted)
        .context("Oxide submitted counter regressed")?;
    let completed = after
        .completed
        .checked_sub(before.completed)
        .context("Oxide completed counter regressed")?;
    let failed = after
        .failed
        .checked_sub(before.failed)
        .context("Oxide failed counter regressed")?;
    let enqueue_host_nanoseconds = after
        .enqueue_host_nanoseconds
        .checked_sub(before.enqueue_host_nanoseconds)
        .context("Oxide enqueue profile counter regressed")?;
    let drain_host_nanoseconds = after
        .drain_host_nanoseconds
        .checked_sub(before.drain_host_nanoseconds)
        .context("Oxide drain profile counter regressed")?;
    let drain_calls = after
        .drain_calls
        .checked_sub(before.drain_calls)
        .context("Oxide drain call counter regressed")?;
    let profile_enabled = stats.profile_enabled();
    if submitted == 0
        || completed != submitted
        || failed != 0
        || !stats.adapter_zero_copy()
        || stats.adapter_device_to_device_copies() != 0
        || (profile_enabled && drain_calls == 0)
    {
        bail!(
            "Oxide provider validation failed: submitted={submitted}, completed={completed}, \
             failed={failed}, zero_copy={}, d2d_copies={}, profile_enabled={profile_enabled}, \
             drain_calls={drain_calls}",
            stats.adapter_zero_copy(),
            stats.adapter_device_to_device_copies()
        );
    }
    let delta = OxideStatsDelta {
        submitted,
        completed,
        failed,
        last_operator: format!("{:?}", stats.last_operator()),
        last_layout: format!("{:?}", stats.last_layout()),
        last_algorithm: format!("{:?}", stats.last_algorithm()),
        adapter_zero_copy: stats.adapter_zero_copy(),
        external_regions: stats.external_regions(),
        adapter_device_to_device_copies: stats.adapter_device_to_device_copies(),
        profile_enabled,
        enqueue_host_nanoseconds,
        enqueue_host_microseconds_per_operator: profile_enabled
            .then_some(enqueue_host_nanoseconds as f64 / submitted as f64 / 1_000.0),
        drain_host_nanoseconds,
        drain_calls,
        drain_host_microseconds_per_forward: profile_enabled
            .then_some(drain_host_nanoseconds as f64 / drain_calls as f64 / 1_000.0),
    };
    Ok(Some(delta))
}

fn summarize_provider(provider: Provider, blocks: &[WorkerRecord]) -> Result<ProviderSummary> {
    let selected = blocks
        .iter()
        .filter(|block| block.provider == provider)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("schedule has no {provider:?} blocks");
    }
    let measurements = selected
        .iter()
        .flat_map(|block| block.measurements.iter())
        .collect::<Vec<_>>();
    let block_tpot_p50_ms = selected
        .iter()
        .map(|block| {
            distribution(block.measurements.iter().map(|item| item.tpot_ms)).map(|x| x.p50)
        })
        .collect::<Result<Vec<_>>>()?;
    let block_decode_tokens_per_second_p50 = selected
        .iter()
        .map(|block| {
            distribution(
                block
                    .measurements
                    .iter()
                    .map(|item| item.decode_tokens_per_second),
            )
            .map(|x| x.p50)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ProviderSummary {
        provider,
        block_count: selected.len(),
        sample_count: measurements.len(),
        ttft_ms: distribution(measurements.iter().map(|item| item.ttft_ms))?,
        tpot_ms: distribution(measurements.iter().map(|item| item.tpot_ms))?,
        end_to_end_ms: distribution(measurements.iter().map(|item| item.end_to_end_ms))?,
        decode_tokens_per_second: distribution(
            measurements
                .iter()
                .map(|item| item.decode_tokens_per_second),
        )?,
        completion_tokens_per_second: distribution(
            measurements
                .iter()
                .map(|item| item.completion_tokens_per_second),
        )?,
        device_memory_delta_after_warmup_mib: distribution(
            selected
                .iter()
                .map(|block| block.device_memory_delta_after_warmup_mib),
        )?,
        device_memory_delta_after_measurements_mib: distribution(
            selected
                .iter()
                .map(|block| block.device_memory_delta_after_measurements_mib),
        )?,
        block_tpot_p50_ms,
        block_decode_tokens_per_second_p50,
    })
}

fn distribution(values: impl IntoIterator<Item = f64>) -> Result<Distribution> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("distribution requires finite observations");
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
        bail!("invalid nearest-rank input");
    }
    let rank = (percentile * sorted.len()).div_ceil(100);
    Ok(sorted[rank - 1])
}

fn ratio(numerator: f64, denominator: f64) -> Result<f64> {
    if denominator <= 0.0 {
        bail!("ratio denominator must be positive");
    }
    Ok(numerator / denominator)
}

fn validate_counts(warmups: usize, iterations: usize, max_output_tokens: usize) -> Result<()> {
    if warmups == 0 || iterations == 0 || max_output_tokens < 2 {
        bail!("warmups and iterations must be positive, and max output tokens must be at least 2");
    }
    Ok(())
}

fn query_device_used_memory_mib(device: &Device) -> Result<f64> {
    let stream = device.as_cuda_device()?.cuda_stream();
    let (free, total) = stream
        .context()
        .mem_get_info()
        .context("CUDA device memory query failed")?;
    let used = total
        .checked_sub(free)
        .context("CUDA free memory exceeded total memory")?;
    Ok(used as f64 / BYTES_PER_MIB)
}

fn memory_delta_mib(current: f64, baseline: f64) -> Result<f64> {
    let delta = current - baseline;
    if delta < 0.0 {
        bail!("CUDA device-used memory fell below the post-context baseline");
    }
    Ok(delta)
}

fn query_hardware() -> Result<Hardware> {
    let output = ProcessCommand::new("nvidia-smi")
        .args([
            &format!("--query-gpu={GPU_QUERY_FIELDS}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to run nvidia-smi hardware query")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi hardware query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let text = String::from_utf8(output.stdout).context("nvidia-smi returned non-UTF-8 output")?;
    let line = text.lines().next().context("nvidia-smi returned no GPUs")?;
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 4 {
        bail!("unexpected nvidia-smi hardware output: {line}");
    }
    Ok(Hardware {
        gpu: fields[0].to_string(),
        compute_capability: fields[1].to_string(),
        memory_total_mib: fields[2].parse().context("invalid total GPU memory")?,
        driver: fields[3].to_string(),
    })
}

fn git_output(args: &[&str]) -> Result<String> {
    let output = ProcessCommand::new("git")
        .args(args)
        .output()
        .context("failed to run git")?;
    if !output.status.success() {
        bail!(
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8(output.stdout)
        .context("git returned non-UTF-8 output")?
        .trim()
        .to_string())
}

fn git_worktree_clean() -> Result<bool> {
    Ok(git_output(&["status", "--porcelain"])?.is_empty())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{distribution, nearest_rank, Provider, DEFAULT_SCHEDULE};

    #[test]
    fn schedule_is_position_balanced() {
        assert_eq!(
            DEFAULT_SCHEDULE
                .iter()
                .filter(|provider| **provider == Provider::Oxide)
                .count(),
            3
        );
        assert_eq!(
            DEFAULT_SCHEDULE
                .iter()
                .filter(|provider| **provider == Provider::Baseline)
                .count(),
            3
        );
        assert_eq!(DEFAULT_SCHEDULE[0], DEFAULT_SCHEDULE[4]);
        assert_eq!(DEFAULT_SCHEDULE[1], DEFAULT_SCHEDULE[2]);
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
