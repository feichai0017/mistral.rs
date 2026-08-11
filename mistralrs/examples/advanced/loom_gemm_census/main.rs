use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs::{hard_link, read, remove_file, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{bail, Context, Result};
use candle_core::Device;
use mistralrs::{
    gemm_census::{
        begin_gemm_census_run, GemmCensusDeviceKind, GemmCensusEntry, GemmCensusObservedPath,
        GemmCensusPhase,
    },
    DeviceMapSetting, MemoryGpuConfig, ModelDType, PagedAttentionMetaBuilder, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "loom.gemm-shape-census.v1";
const LOOM_SCHEMA_COMMIT: &str = "8b971b064d4246b2cd5cbc74f9902a51c720aefa";
const RUST_TOOLCHAIN: &str = "+nightly-2026-04-03";
const BUILD_RUSTC_VERSION: &str = env!("LOOM_GEMM_CENSUS_BUILD_RUSTC_VERSION");
const BUILD_NVCC_PATH: &str = env!("LOOM_GEMM_CENSUS_BUILD_NVCC_PATH");
const BUILD_NVCC_VERSION_HEX: &str = env!("LOOM_GEMM_CENSUS_BUILD_NVCC_VERSION_HEX");
const BUILD_CUDA_COMPUTE_CAP: &str = env!("LOOM_GEMM_CENSUS_BUILD_CUDA_COMPUTE_CAP");
const BUILD_CUDA_ARCH: &str = env!("LOOM_GEMM_CENSUS_BUILD_CUDA_ARCH");
const MODEL_NAME: &str = "Qwen2.5-1.5B-Instruct";
const PROMPT: &str = "Reply with one short sentence that explains what a CUDA kernel does.";
const MAX_OUTPUT_TOKENS: usize = 8;
const TOP_LOGPROBS: usize = 5;
const QWEN_LAYER_COUNT: usize = 28;
const QWEN_LAYER_LINEAR_COUNT: usize = 6;
const PAGE_SIZE: usize = 16;
const MAX_CONTEXT_SIZE: usize = 4096;
const TEMP_CREATE_ATTEMPTS: usize = 1024;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const ACCEPTED_CLAIMS: [&str; 1] =
    ["exact dense-linear host dispatch counts for the pinned model and workload"];
const EXCLUDED_CLAIMS: [&str; 5] = [
    "latency",
    "throughput",
    "CUDA kernel launch counts",
    "general model coverage",
    "production workload frequency",
];

#[derive(Serialize)]
struct CensusRecord {
    schema: &'static str,
    run_id: String,
    source: Source,
    hardware: Hardware,
    environment: Environment,
    model: Model,
    workload: Workload,
    capture: Capture,
    entries: Vec<GemmCensusEntry>,
    accepted_claims: [&'static str; 1],
    excluded_claims: [&'static str; 5],
}

#[derive(Serialize)]
struct Source {
    producer: &'static str,
    repository: String,
    commit: String,
    worktree_clean: bool,
    cargo_lock_sha256: String,
    binary_sha256: String,
    loom_schema_commit: &'static str,
}

#[derive(Serialize)]
struct Hardware {
    gpu: String,
    compute_capability: String,
}

#[derive(Serialize)]
struct Environment {
    host_os: &'static str,
    host_arch: &'static str,
    rustc_version: String,
    cuda_toolkit_version: String,
    driver_version: String,
    cuda_arch: String,
}

#[derive(Serialize)]
struct Model {
    name: &'static str,
    weights_sha256: String,
    config_sha256: String,
    tokenizer_sha256: String,
    tensor_parallel_size: usize,
}

#[derive(Serialize)]
struct Workload {
    scheduler: &'static str,
    request_count: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_forward_steps: usize,
    decode_forward_steps: usize,
    canonical_request: CanonicalRequest,
    canonical_request_sha256: String,
    cuda_graph_enabled: bool,
}

#[derive(Serialize)]
struct CanonicalRequest {
    messages: [CanonicalMessage; 1],
    sampler: CanonicalSampler,
}

#[derive(Serialize)]
struct CanonicalMessage {
    content: &'static str,
    role: &'static str,
}

#[derive(Serialize)]
struct CanonicalSampler {
    max_output_tokens: usize,
    return_logprobs: bool,
    temperature: f64,
    top_logprobs: usize,
}

#[derive(Serialize)]
struct Capture {
    boundary: &'static str,
    count_semantics: &'static str,
    sample_rate: usize,
    complete: bool,
    timed: bool,
    mistral_gemv_enabled: bool,
}

struct SourceSnapshot {
    root: PathBuf,
    remote: String,
    repository: String,
    commit: String,
    cargo_lock_sha256: String,
    binary_sha256: String,
}

#[derive(Debug, Eq, PartialEq)]
struct ModelSnapshot {
    weights_sha256: String,
    config_sha256: String,
    tokenizer_sha256: String,
}

struct TempOutput {
    path: Option<PathBuf>,
}

impl TempOutput {
    fn remove(mut self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("temporary output path was already removed"))?;
        remove_file(path)
            .with_context(|| format!("failed to remove temporary output {}", path.display()))?;
        self.path = None;
        Ok(())
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = remove_file(path);
        }
    }
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} must be set"))?;
    if value.trim().is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn required_exact_env(name: &str, expected: &str) -> Result<()> {
    let actual = required_env(name)?;
    if actual != expected {
        bail!("{name} must be {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| parse_env_flag(&value))
}

fn parse_env_flag(value: &str) -> bool {
    value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
}

fn require_cuda_feature() -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    bail!("loom_gemm_census must be built with the cuda feature");
    #[cfg(feature = "cuda")]
    Ok(())
}

fn required_hex_env(name: &str, length: usize) -> Result<String> {
    let value = required_env(name)?;
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{name} must be a lowercase {length}-character hexadecimal digest");
    }
    Ok(value)
}

fn required_remote_env() -> Result<String> {
    let remote = required_env("LOOM_GEMM_CENSUS_REMOTE")?;
    if remote.starts_with('-')
        || !remote
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        bail!("LOOM_GEMM_CENSUS_REMOTE is not a valid Git remote name");
    }
    Ok(remote)
}

fn command_output(command: impl AsRef<OsStr>, args: &[&str]) -> Result<String> {
    let command = command.as_ref();
    let command_name = command.to_string_lossy();
    let output = Command::new(command)
        .args(args)
        .output()
        .with_context(|| format!("failed to execute {command_name}"))?;
    if !output.status.success() {
        bail!(
            "{command_name} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8(output.stdout)
        .with_context(|| format!("{command_name} output was not UTF-8"))?;
    let stdout = stdout.trim().to_string();
    if stdout.is_empty() {
        bail!("{command_name} returned empty output");
    }
    Ok(stdout)
}

fn sha256_reader(mut reader: impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    sha256_reader(BufReader::new(file))
        .with_context(|| format!("failed to hash {}", path.display()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    let root = root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("source path is not UTF-8"))?;
    let mut command_args = vec!["-C", root];
    command_args.extend_from_slice(args);
    command_output("git", &command_args)
}

fn assert_clean_worktree(root: &Path) -> Result<()> {
    let root = root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("source path is not UTF-8"))?;
    let output = Command::new("git")
        .args([
            "-C",
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ])
        .output()
        .context("failed to inspect the source worktree")?;
    if !output.status.success() {
        bail!(
            "git status failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    if !output.stdout.is_empty() {
        bail!("GEMM census requires a clean source worktree");
    }
    Ok(())
}

fn capture_source_snapshot() -> Result<SourceSnapshot> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow::anyhow!("mistralrs package has no workspace parent"))?
        .canonicalize()
        .context("failed to resolve the source root")?;
    let expected_commit = required_hex_env("LOOM_GEMM_CENSUS_SOURCE_COMMIT", 40)?;
    let commit = git_output(&root, &["rev-parse", "HEAD"])?;
    if commit != expected_commit {
        bail!("LOOM_GEMM_CENSUS_SOURCE_COMMIT is {expected_commit}, but HEAD is {commit}");
    }
    let remote = required_remote_env()?;
    let expected_repository = required_env("LOOM_GEMM_CENSUS_REPOSITORY")?;
    let repository = git_output(&root, &["remote", "get-url", &remote])?;
    if repository != expected_repository {
        bail!(
            "LOOM_GEMM_CENSUS_REPOSITORY is {expected_repository:?}, but {remote} is {repository:?}"
        );
    }
    assert_clean_worktree(&root)?;

    let cargo_lock = read(root.join("Cargo.lock")).context("failed to read Cargo.lock")?;
    let cargo_lock_sha256 = sha256_bytes(&cargo_lock);
    let binary_sha256 = sha256_file(&std::env::current_exe()?)?;
    Ok(SourceSnapshot {
        root,
        remote,
        repository,
        commit,
        cargo_lock_sha256,
        binary_sha256,
    })
}

fn verify_source_snapshot(snapshot: &SourceSnapshot) -> Result<()> {
    let commit = git_output(&snapshot.root, &["rev-parse", "HEAD"])?;
    if commit != snapshot.commit {
        bail!("source HEAD changed during the census run");
    }
    if git_output(&snapshot.root, &["remote", "get-url", &snapshot.remote])? != snapshot.repository
    {
        bail!("source remote changed during the census run");
    }
    assert_clean_worktree(&snapshot.root)?;
    if sha256_file(&snapshot.root.join("Cargo.lock"))? != snapshot.cargo_lock_sha256 {
        bail!("Cargo.lock changed during the census run");
    }
    if sha256_file(&std::env::current_exe()?)? != snapshot.binary_sha256 {
        bail!("the census binary changed during the census run");
    }
    Ok(())
}

fn verify_expected_sha(name: &str, path: &Path, actual: &str) -> Result<()> {
    let expected = required_hex_env(name, 64)?;
    if actual != expected {
        bail!(
            "{name} is {expected}, but {} hashes to {actual}",
            path.display()
        );
    }
    Ok(())
}

fn hash_model_snapshot(model_path: &Path) -> Result<ModelSnapshot> {
    Ok(ModelSnapshot {
        weights_sha256: sha256_file(&model_path.join("model.safetensors"))?,
        config_sha256: sha256_file(&model_path.join("config.json"))?,
        tokenizer_sha256: sha256_file(&model_path.join("tokenizer.json"))?,
    })
}

fn capture_model_snapshot(model_path: &Path) -> Result<ModelSnapshot> {
    let snapshot = hash_model_snapshot(model_path)?;
    for (name, path, actual) in [
        (
            "LOOM_GEMM_CENSUS_WEIGHTS_SHA256",
            model_path.join("model.safetensors"),
            snapshot.weights_sha256.as_str(),
        ),
        (
            "LOOM_GEMM_CENSUS_CONFIG_SHA256",
            model_path.join("config.json"),
            snapshot.config_sha256.as_str(),
        ),
        (
            "LOOM_GEMM_CENSUS_TOKENIZER_SHA256",
            model_path.join("tokenizer.json"),
            snapshot.tokenizer_sha256.as_str(),
        ),
    ] {
        verify_expected_sha(name, &path, actual)?;
    }
    Ok(snapshot)
}

fn verify_model_snapshot(model_path: &Path, expected: &ModelSnapshot) -> Result<()> {
    let actual = hash_model_snapshot(model_path)?;
    if &actual != expected {
        bail!("model files changed during the census run");
    }
    Ok(())
}

fn capture_hardware() -> Result<(Hardware, String)> {
    let output = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
    )?;
    let lines = output.lines().collect::<Vec<_>>();
    if lines.len() != 1 {
        bail!("nvidia-smi must return exactly one row for GPU zero");
    }
    let fields = lines[0].split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 3 || fields.iter().any(|field| field.is_empty()) {
        bail!("nvidia-smi returned an unexpected GPU record");
    }
    for (name, actual) in [
        ("LOOM_GEMM_CENSUS_GPU", fields[0]),
        ("LOOM_GEMM_CENSUS_COMPUTE_CAPABILITY", fields[1]),
        ("LOOM_GEMM_CENSUS_DRIVER_VERSION", fields[2]),
    ] {
        let expected = required_env(name)?;
        if actual != expected {
            bail!("{name} is {expected:?}, but nvidia-smi reports {actual:?}");
        }
    }
    if fields[0] != "NVIDIA H20" || fields[1] != "9.0" {
        bail!("loom_gemm_census admits NVIDIA H20 compute capability 9.0 only");
    }
    Ok((
        Hardware {
            gpu: fields[0].to_string(),
            compute_capability: fields[1].to_string(),
        },
        fields[2].to_string(),
    ))
}

fn compute_cap_to_cuda_arch(compute_cap: &str) -> Result<&'static str> {
    match compute_cap {
        "90" => Ok("sm_90a"),
        value => bail!("unsupported build CUDA compute capability {value:?}"),
    }
}

fn resolve_nvcc_path(raw_path: &str) -> Result<PathBuf> {
    let path = Path::new(raw_path);
    if !path.is_absolute() {
        bail!("NVCC must be an absolute path, got {raw_path:?}");
    }
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize NVCC={raw_path:?}"))?;
    if !path.is_file() {
        bail!("NVCC must resolve to a file, got {}", path.display());
    }
    Ok(path)
}

fn capture_environment(driver_version: String) -> Result<Environment> {
    let runtime_rustc_version = command_output("rustc", &[RUST_TOOLCHAIN, "--version"])?;
    if runtime_rustc_version != BUILD_RUSTC_VERSION {
        bail!(
            "runtime rustc is {runtime_rustc_version:?}, but the binary used {BUILD_RUSTC_VERSION:?}"
        );
    }
    let runtime_nvcc_path = resolve_nvcc_path(&required_env("NVCC")?)?;
    if runtime_nvcc_path != Path::new(BUILD_NVCC_PATH) {
        bail!(
            "runtime NVCC resolves to {}, but the binary used {BUILD_NVCC_PATH}",
            runtime_nvcc_path.display()
        );
    }
    let runtime_nvcc_version = command_output(runtime_nvcc_path.as_os_str(), &["--version"])?;
    if hex_bytes(runtime_nvcc_version.as_bytes()) != BUILD_NVCC_VERSION_HEX {
        bail!("runtime nvcc version does not match the nvcc version embedded by the build");
    }
    required_exact_env("CUDA_COMPUTE_CAP", BUILD_CUDA_COMPUTE_CAP)?;
    let cuda_arch = compute_cap_to_cuda_arch(BUILD_CUDA_COMPUTE_CAP)?;
    if cuda_arch != BUILD_CUDA_ARCH {
        bail!("embedded CUDA architecture does not match the compute capability");
    }
    Ok(Environment {
        host_os: std::env::consts::OS,
        host_arch: std::env::consts::ARCH,
        rustc_version: BUILD_RUSTC_VERSION.to_string(),
        cuda_toolkit_version: runtime_nvcc_version,
        driver_version,
        cuda_arch: BUILD_CUDA_ARCH.to_string(),
    })
}

fn canonical_request() -> CanonicalRequest {
    CanonicalRequest {
        messages: [CanonicalMessage {
            content: PROMPT,
            role: "user",
        }],
        sampler: CanonicalSampler {
            max_output_tokens: MAX_OUTPUT_TOKENS,
            return_logprobs: true,
            temperature: 0.0,
            top_logprobs: TOP_LOGPROBS,
        },
    }
}

fn expected_sites() -> BTreeSet<String> {
    let suffixes = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.merged_gate_up",
        "mlp.down_proj",
    ];
    let mut sites = BTreeSet::new();
    for layer in 0..QWEN_LAYER_COUNT {
        for suffix in suffixes {
            sites.insert(format!("model.layers.{layer}.{suffix}"));
        }
    }
    sites.insert("lm_head".to_string());
    sites
}

fn validate_entries(entries: &[GemmCensusEntry], decode_forward_steps: usize) -> Result<()> {
    if entries.is_empty() {
        bail!("GEMM census collector returned no entries");
    }
    let expected_sites = expected_sites();
    let expected_site_count = QWEN_LAYER_COUNT
        .checked_mul(QWEN_LAYER_LINEAR_COUNT)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("expected site count overflowed"))?;
    if expected_sites.len() != expected_site_count {
        bail!("internal Qwen site list has the wrong size");
    }

    let mut calls = BTreeMap::<(GemmCensusPhase, String), u64>::new();
    let mut observed_gemv = false;
    for entry in entries {
        if entry.device_kind != GemmCensusDeviceKind::Cuda || entry.device_ordinal != 0 {
            bail!(
                "GEMM census entry {} did not run on CUDA device zero",
                entry.site
            );
        }
        if !expected_sites.contains(&entry.site) {
            bail!("GEMM census recorded unexpected site {}", entry.site);
        }
        observed_gemv |= entry.observed_path == GemmCensusObservedPath::MistralCudaGemv;
        let count = calls.entry((entry.phase, entry.site.clone())).or_default();
        *count = count
            .checked_add(entry.host_calls)
            .ok_or_else(|| anyhow::anyhow!("host call count overflowed"))?;
    }
    if !observed_gemv {
        bail!("GEMM census did not observe the enabled Mistral CUDA GEMV path");
    }

    for site in &expected_sites {
        let prefill_calls = calls
            .get(&(GemmCensusPhase::Prefill, site.clone()))
            .copied()
            .unwrap_or_default();
        if prefill_calls != 1 {
            bail!("prefill site {site} recorded {prefill_calls} calls, expected one");
        }
        let decode_calls = calls
            .get(&(GemmCensusPhase::Decode, site.clone()))
            .copied()
            .unwrap_or_default();
        if decode_calls != decode_forward_steps as u64 {
            bail!(
                "decode site {site} recorded {decode_calls} calls, expected {decode_forward_steps}"
            );
        }
    }
    Ok(())
}

fn output_path(source_root: &Path) -> Result<PathBuf> {
    let raw = PathBuf::from(required_env("LOOM_GEMM_CENSUS_OUTPUT")?);
    let file_name = raw
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("LOOM_GEMM_CENSUS_OUTPUT must name a file"))?;
    let parent = raw
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("failed to resolve the census output parent")?;
    let path = parent.join(file_name);
    if path.starts_with(source_root) {
        bail!("LOOM_GEMM_CENSUS_OUTPUT must be outside the source worktree");
    }
    if path.try_exists()? {
        bail!("LOOM_GEMM_CENSUS_OUTPUT already exists");
    }
    Ok(path)
}

fn create_temp_output(parent: &Path) -> Result<(File, TempOutput)> {
    for _ in 0..TEMP_CREATE_ATTEMPTS {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".loom-gemm-census.{}.{counter}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                return Ok((file, TempOutput { path: Some(path) }));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).context("failed to create temporary census output");
            }
        }
    }
    bail!("failed to allocate a unique temporary census output");
}

fn publish_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("census output has no parent directory"))?;
    let (mut output, temporary) = create_temp_output(parent)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    drop(output);

    let temporary_path = temporary
        .path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("temporary census output path is missing"))?;
    hard_link(temporary_path, path)
        .with_context(|| format!("failed to publish {} without overwrite", path.display()))?;
    temporary.remove()?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn write_record(path: &Path, record: &CensusRecord) -> Result<()> {
    let mut bytes = serde_json::to_vec(record)?;
    bytes.push(b'\n');
    publish_bytes(path, &bytes)
}

#[tokio::main]
async fn main() -> Result<()> {
    require_cuda_feature()?;
    required_exact_env("MISTRALRS_CUDA_GRAPHS", "0")?;
    if env_flag_enabled("MISTRALRS_LOOM_INFER") {
        bail!("loom_gemm_census requires the Mistral.rs baseline provider");
    }
    let source_snapshot = capture_source_snapshot()?;
    let (hardware, driver_version) = capture_hardware()?;
    let environment = capture_environment(driver_version)?;
    let output_path = output_path(&source_snapshot.root)?;
    let model_path = PathBuf::from(required_env("LOOM_MODEL_PATH")?)
        .canonicalize()
        .context("failed to resolve LOOM_MODEL_PATH")?;
    if !model_path.is_dir() {
        bail!("LOOM_MODEL_PATH must be a directory");
    }
    let model_snapshot = capture_model_snapshot(&model_path)?;
    let run_id = required_env("LOOM_GEMM_CENSUS_RUN_ID")?;

    let canonical_request = canonical_request();
    let canonical_request_json = serde_json::to_vec(&canonical_request)?;

    let device = Device::new_cuda_with_stream(0)?;
    let model = TextModelBuilder::new(
        model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("LOOM_MODEL_PATH is not UTF-8"))?,
    )
    .with_dtype(ModelDType::BF16)
    .with_device(device)
    .with_device_mapping(DeviceMapSetting::dummy())
    .with_max_num_seqs(1)
    .with_paged_attn(
        PagedAttentionMetaBuilder::default()
            .with_block_size(PAGE_SIZE)
            .with_gpu_memory(MemoryGpuConfig::ContextSize(MAX_CONTEXT_SIZE))
            .build()?,
    )
    .build()
    .await?;

    let mut census_run = begin_gemm_census_run()?;
    let messages = TextMessages::new().add_message(TextMessageRole::User, PROMPT);
    let response = model
        .send_chat_request(
            RequestBuilder::from(messages)
                .set_sampler_temperature(0.0)
                .set_sampler_max_len(MAX_OUTPUT_TOKENS)
                .set_sampler_topn_logprobs(TOP_LOGPROBS)
                .return_logprobs(true),
        )
        .await?;
    if response.choices.len() != 1 {
        bail!(
            "model returned {} choices for a single request",
            response.choices.len()
        );
    }
    if response.usage.prompt_tokens == 0 || response.usage.completion_tokens == 0 {
        bail!("model returned zero prompt or completion tokens");
    }
    let decode_forward_steps = response
        .usage
        .completion_tokens
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("completion token count underflowed"))?;
    let entries = census_run.finish()?;
    validate_entries(&entries, decode_forward_steps)?;
    verify_model_snapshot(&model_path, &model_snapshot)?;
    verify_source_snapshot(&source_snapshot)?;

    let record = CensusRecord {
        schema: SCHEMA,
        run_id,
        source: Source {
            producer: "mistralrs/loom_gemm_census",
            repository: source_snapshot.repository,
            commit: source_snapshot.commit,
            worktree_clean: true,
            cargo_lock_sha256: source_snapshot.cargo_lock_sha256,
            binary_sha256: source_snapshot.binary_sha256,
            loom_schema_commit: LOOM_SCHEMA_COMMIT,
        },
        hardware,
        environment,
        model: Model {
            name: MODEL_NAME,
            weights_sha256: model_snapshot.weights_sha256,
            config_sha256: model_snapshot.config_sha256,
            tokenizer_sha256: model_snapshot.tokenizer_sha256,
            tensor_parallel_size: 1,
        },
        workload: Workload {
            scheduler: "single_request",
            request_count: 1,
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            prefill_forward_steps: 1,
            decode_forward_steps,
            canonical_request,
            canonical_request_sha256: sha256_bytes(&canonical_request_json),
            cuda_graph_enabled: false,
        },
        capture: Capture {
            boundary: "resolved_dense_linear_before_backend_selection",
            count_semantics: "successful_linear_dispatches",
            sample_rate: 1,
            complete: true,
            timed: false,
            mistral_gemv_enabled: true,
        },
        entries,
        accepted_claims: ACCEPTED_CLAIMS,
        excluded_claims: EXCLUDED_CLAIMS,
    };
    write_record(&output_path, &record)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, sync::atomic::Ordering};

    use super::{
        canonical_request, compute_cap_to_cuda_arch, expected_sites, hash_model_snapshot,
        hex_bytes, parse_env_flag, publish_bytes, resolve_nvcc_path, sha256_bytes,
        verify_model_snapshot, BUILD_CUDA_ARCH, BUILD_CUDA_COMPUTE_CAP, BUILD_NVCC_PATH,
        BUILD_NVCC_VERSION_HEX, BUILD_RUSTC_VERSION, TEMP_COUNTER,
    };

    fn test_dir(label: &str) -> PathBuf {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "loom-gemm-census-test-{}-{label}-{counter}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn canonical_request_matches_the_pinned_python_encoding() {
        let request = serde_json::to_vec(&canonical_request()).unwrap();
        assert_eq!(
            std::str::from_utf8(&request).unwrap(),
            r#"{"messages":[{"content":"Reply with one short sentence that explains what a CUDA kernel does.","role":"user"}],"sampler":{"max_output_tokens":8,"return_logprobs":true,"temperature":0.0,"top_logprobs":5}}"#
        );
        assert_eq!(
            sha256_bytes(&request),
            "ae3bc2183cc6d5af45c63f70b7429eb7120b68b9484c7c6acce9e079573eb1c7"
        );
    }

    #[test]
    fn qwen_site_list_has_six_linears_per_layer_and_lm_head() {
        let sites = expected_sites();
        assert_eq!(sites.len(), 169);
        assert!(sites.contains("model.layers.0.self_attn.q_proj"));
        assert!(sites.contains("model.layers.27.mlp.down_proj"));
        assert!(sites.contains("lm_head"));
    }

    #[test]
    fn build_provenance_is_embedded() {
        assert!(!BUILD_RUSTC_VERSION.is_empty());
        assert_eq!(BUILD_NVCC_PATH, "test-only");
        assert_eq!(BUILD_NVCC_VERSION_HEX, "test-only");
        assert_eq!(BUILD_CUDA_COMPUTE_CAP, "test-only");
        assert_eq!(BUILD_CUDA_ARCH, "test-only");
    }

    #[test]
    fn compute_capability_maps_to_the_schema_arch() {
        assert_eq!(compute_cap_to_cuda_arch("90").unwrap(), "sm_90a");
        assert!(compute_cap_to_cuda_arch("89").is_err());
        assert_eq!(hex_bytes(b"nvcc\n13.1"), "6e7663630a31332e31");
    }

    #[test]
    fn nvcc_path_must_be_absolute_and_resolve_to_a_file() {
        let directory = test_dir("nvcc");
        let nvcc = directory.join("nvcc");
        fs::write(&nvcc, b"test nvcc").unwrap();

        assert_eq!(
            resolve_nvcc_path(nvcc.to_str().unwrap()).unwrap(),
            nvcc.canonicalize().unwrap()
        );
        assert!(resolve_nvcc_path("relative/nvcc").is_err());
        assert!(resolve_nvcc_path(directory.to_str().unwrap()).is_err());

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn publish_is_complete_and_does_not_clobber() {
        let directory = test_dir("publish");
        let output = directory.join("record.jsonl");

        publish_bytes(&output, b"first\n").unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"first\n");
        assert!(publish_bytes(&output, b"second\n").is_err());
        assert_eq!(fs::read(&output).unwrap(), b"first\n");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn model_snapshot_detects_a_file_change() {
        let directory = test_dir("model");
        fs::write(directory.join("model.safetensors"), b"weights").unwrap();
        fs::write(directory.join("config.json"), b"config").unwrap();
        fs::write(directory.join("tokenizer.json"), b"tokenizer").unwrap();
        let snapshot = hash_model_snapshot(&directory).unwrap();

        fs::write(directory.join("tokenizer.json"), b"changed").unwrap();
        assert!(verify_model_snapshot(&directory, &snapshot).is_err());

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn loom_provider_flag_accepts_all_enabled_spellings() {
        for value in ["1", "true", "TRUE", "yes", "YES", "on", "ON"] {
            assert!(parse_env_flag(value));
        }
        for value in ["0", "false", "no", "off", "unexpected"] {
            assert!(!parse_env_flag(value));
        }
    }
}
