use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{CudaStream, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, DType, Error, InplaceOp1, Layout, Result, Storage, Tensor,
};
use half::bf16;
use oxide_cuda_core::CudaContext;
use oxide_infer::{Bf16PagedBatchDecodeSpec, PagedKvLayout};
use oxide_infer_cuda::attention::{
    Bf16PagedBatchDecodeArgs, Bf16PagedBatchDecodePlan, DecodeProvider,
    TrustedBf16PagedBatchDecodeArgs,
};
use oxide_infer_cuda::interop::{
    EngineAlgorithm, EngineCommand, EngineCommandCompletion, EngineCommandCompletionError,
    EngineCommandFailure, EngineEnqueueCause, EngineExecutionTrace, EngineExternalBindings,
    EngineInteropQueue, EngineMetadataValidation, EngineOperator, EngineStreamHandoff,
    ExternalCudaStream, StreamOrderedEngineAuthority,
};
use oxide_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

const HEAD_DIM: usize = 128;
const PAGE_SIZE: usize = 16;
const COMMAND_CAPACITY: usize = 3;
const MAX_IN_FLIGHT: usize = 128;
const BINDING_COUNT: usize = 9;
const OXIDE_PROFILE_ENV: &str = "MISTRALRS_OXIDE_PROFILE";
const OXIDE_DEVICE_PROFILE_ENV: &str = "MISTRALRS_OXIDE_DEVICE_PROFILE";
const OXIDE_DIRECT_STREAM_ENV: &str = "MISTRALRS_OXIDE_DIRECT_STREAM";

static OXIDE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static OXIDE_DEVICE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static OXIDE_DIRECT_STREAM_ENABLED: OnceLock<bool> = OnceLock::new();

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RuntimeKey {
    context: usize,
    stream: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PlanKey {
    batch_size: usize,
    max_num_pages: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
}

#[derive(Clone, Copy)]
enum AdapterMetadataValidation {
    DeviceChecked,
    TrustedByCaller,
}

struct CandleStreamLease {
    _stream: Arc<CudaStream>,
}

struct CandleStreamAuthority {
    stream: Arc<CudaStream>,
}

// SAFETY: this private authority exists only while the caller upholds the
// runtime's exclusive model-runner submission contract.
unsafe impl StreamOrderedEngineAuthority for CandleStreamAuthority {
    fn submission_stream(&self) -> oxide_cuda_core::sys::CUstream {
        oxide_stream(&self.stream)
    }
}

struct OxideRuntime {
    key: RuntimeKey,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    provider: DecodeProvider,
    queue: EngineInteropQueue,
    plans: HashMap<PlanKey, Bf16PagedBatchDecodePlan>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EnqueueProfile {
    preparation_host_nanoseconds: u64,
    allocation_host_nanoseconds: u64,
    guard_host_nanoseconds: u64,
    binding_host_nanoseconds: u64,
    interop_host_nanoseconds: u64,
    engine_total_host_nanoseconds: u64,
    engine_setup_host_nanoseconds: u64,
    engine_pre_handoff_host_nanoseconds: u64,
    engine_provider_host_nanoseconds: u64,
    engine_provider_preflight_host_nanoseconds: u64,
    engine_provider_metadata_host_nanoseconds: u64,
    engine_provider_attention_host_nanoseconds: u64,
    engine_status_readback_host_nanoseconds: u64,
    engine_post_handoff_host_nanoseconds: u64,
}

struct ProfiledCompletion {
    completion: EngineCommandCompletion,
    profile: EnqueueProfile,
}

impl OxideRuntime {
    fn new(
        key: RuntimeKey,
        ordinal: usize,
        stream: Arc<CudaStream>,
        direct_stream: bool,
    ) -> Result<Self> {
        if key.stream <= 2 {
            return Err(oxide_error(
                "Oxide paged decode requires an ordinary CUDA stream",
            ));
        }
        let context = CudaContext::new(ordinal).map_err(|error| {
            oxide_external_error("failed to retain the CUDA primary context", error)
        })?;
        if context.cu_ctx() as usize != key.context {
            return Err(oxide_error(
                "Candle and Oxide resolved different CUDA contexts",
            ));
        }
        let lease = Arc::new(CandleStreamLease {
            _stream: Arc::clone(&stream),
        });
        // SAFETY: `stream` is retained by both this runtime and `lease`. The
        // public unsafe call contract supplies stream submission exclusivity.
        let external = unsafe {
            ExternalCudaStream::from_raw_parts(oxide_stream(&stream), Arc::clone(&context), lease)
        }
        .map_err(|error| oxide_external_error("failed to import the Candle stream", error))?;
        let provider = DecodeProvider::load(&context)
            .map_err(|error| oxide_external_error("failed to load Oxide decode kernels", error))?;
        if direct_stream && oxide_device_profile_enabled() {
            return Err(oxide_error(
                "Oxide direct stream and device profiling cannot be enabled together",
            ));
        }
        let mut queue = if direct_stream {
            EngineInteropQueue::new_direct(external, COMMAND_CAPACITY, MAX_IN_FLIGHT)
        } else if oxide_device_profile_enabled() {
            EngineInteropQueue::new_device_profiled(external, COMMAND_CAPACITY, MAX_IN_FLIGHT)
        } else {
            EngineInteropQueue::new(external, COMMAND_CAPACITY, MAX_IN_FLIGHT)
        }
        .map_err(|error| oxide_external_error("failed to create the Oxide queue", error))?;
        queue.set_host_profiling_enabled(oxide_profile_enabled());
        Ok(Self {
            key,
            stream,
            context,
            provider,
            queue,
            plans: HashMap::new(),
        })
    }

    fn plan(
        &mut self,
        key: PlanKey,
        spec: Bf16PagedBatchDecodeSpec,
    ) -> Result<Bf16PagedBatchDecodePlan> {
        if let Some(plan) = self.plans.get(&key) {
            return Ok(plan.clone());
        }
        let plan = self.provider.plan_bf16_paged_batch(spec).map_err(|error| {
            oxide_external_error("failed to create the Oxide decode plan", error)
        })?;
        self.plans.insert(key, plan.clone());
        Ok(plan)
    }

    fn enqueue(
        &mut self,
        plan: &Bf16PagedBatchDecodePlan,
        tensors: &DecodeTensors<'_>,
        pointers: DecodePointers,
        metadata_validation: AdapterMetadataValidation,
    ) -> Result<ProfiledCompletion> {
        let binding_started = oxide_profile_enabled().then(Instant::now);
        let mut bindings = self
            .queue
            .bindings(BINDING_COUNT)
            .map_err(|error| oxide_external_error("failed to allocate Oxide bindings", error))?;
        let context = Arc::clone(&self.context);

        // SAFETY: pointer, extent, access, and lifetime facts were checked by
        // the Candle storage guards held across this enqueue.
        let query = unsafe {
            ReadDeviceRegion::<bf16>::from_external_parts(
                pointers.query,
                tensors.spec.query_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.query),
            )
        }
        .map_err(|error| oxide_external_error("invalid query region", error))?;
        // SAFETY: see the query region construction above.
        let key_pages = unsafe {
            ReadDeviceRegion::<bf16>::from_external_parts(
                pointers.key_pages,
                tensors.spec.kv_pages_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.key_cache),
            )
        }
        .map_err(|error| oxide_external_error("invalid key-cache region", error))?;
        // SAFETY: see the query region construction above.
        let value_pages = unsafe {
            ReadDeviceRegion::<bf16>::from_external_parts(
                pointers.value_pages,
                tensors.spec.kv_pages_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.value_cache),
            )
        }
        .map_err(|error| oxide_external_error("invalid value-cache region", error))?;
        // SAFETY: see the query region construction above.
        let page_indptr = unsafe {
            ReadDeviceRegion::<i32>::from_external_parts(
                pointers.page_indptr,
                tensors.spec.page_indptr_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.page_indptr),
            )
        }
        .map_err(|error| oxide_external_error("invalid page-indptr region", error))?;
        // SAFETY: only the logical CSR prefix is exposed to Oxide. The full
        // contiguous Tensor remains retained by its independent lease.
        let page_indices = unsafe {
            ReadDeviceRegion::<i32>::from_external_range(
                pointers.page_indices,
                tensors.page_indices.elem_count(),
                0..tensors.logical_page_count,
                Arc::clone(&context),
                tensor_lease(tensors.page_indices),
            )
        }
        .map_err(|error| oxide_external_error("invalid page-indices region", error))?;
        // SAFETY: see the query region construction above.
        let last_page_len = unsafe {
            ReadDeviceRegion::<i32>::from_external_parts(
                pointers.last_page_len,
                tensors.spec.last_page_len_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.last_page_len),
            )
        }
        .map_err(|error| oxide_external_error("invalid last-page-len region", error))?;
        // SAFETY: the nested Candle in-place guards transfer exclusive write
        // access until the post-event wait is queued.
        let metadata_status = unsafe {
            ReadWriteDeviceRegion::<i32>::from_external_parts(
                pointers.metadata_status,
                plan.metadata_status_required_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.metadata_status),
            )
        }
        .map_err(|error| oxide_external_error("invalid metadata-status region", error))?;
        // SAFETY: see the metadata-status region construction above.
        let output = unsafe {
            ReadWriteDeviceRegion::<bf16>::from_external_parts(
                pointers.output,
                tensors.spec.output_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.output),
            )
        }
        .map_err(|error| oxide_external_error("invalid output region", error))?;
        // SAFETY: see the metadata-status region construction above.
        let lse = unsafe {
            ReadWriteDeviceRegion::<f32>::from_external_parts(
                pointers.lse,
                tensors.spec.lse_numel(),
                context,
                tensor_lease(tensors.lse),
            )
        }
        .map_err(|error| oxide_external_error("invalid LSE region", error))?;

        let query = bindings
            .bind_read_region(query)
            .map_err(|error| oxide_external_error("failed to bind query", error))?;
        let key_pages = bindings
            .bind_read_region(key_pages)
            .map_err(|error| oxide_external_error("failed to bind key cache", error))?;
        let value_pages = bindings
            .bind_read_region(value_pages)
            .map_err(|error| oxide_external_error("failed to bind value cache", error))?;
        let page_indptr = bindings
            .bind_read_region(page_indptr)
            .map_err(|error| oxide_external_error("failed to bind page indptr", error))?;
        let page_indices = bindings
            .bind_read_region(page_indices)
            .map_err(|error| oxide_external_error("failed to bind page indices", error))?;
        let last_page_len = bindings
            .bind_read_region(last_page_len)
            .map_err(|error| oxide_external_error("failed to bind last page length", error))?;
        let metadata_status = bindings
            .bind_read_write_region(metadata_status)
            .map_err(|error| oxide_external_error("failed to bind metadata status", error))?;
        let output = bindings
            .bind_read_write_region(output)
            .map_err(|error| oxide_external_error("failed to bind output", error))?;
        let lse = bindings
            .bind_read_write_region(lse)
            .map_err(|error| oxide_external_error("failed to bind LSE", error))?;
        let args = Bf16PagedBatchDecodeArgs::new(
            query,
            key_pages,
            value_pages,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output.write(),
            lse.write(),
        );
        let authority = CandleStreamAuthority {
            stream: Arc::clone(&self.stream),
        };
        // SAFETY: every slot has an independent Tensor lease and the public
        // unsafe call contract grants this authority for the exact spans.
        let external =
            unsafe { EngineExternalBindings::assume_engine_authority(bindings, authority) }
                .map_err(|error| oxide_external_error("failed to couple Candle bindings", error))?;
        let binding_host_nanoseconds = profile_elapsed_nanoseconds(binding_started);
        let interop_started = oxide_profile_enabled().then(Instant::now);
        let command = match metadata_validation {
            AdapterMetadataValidation::DeviceChecked => {
                EngineCommand::Bf16PagedBatchDecode { plan, args }
            }
            AdapterMetadataValidation::TrustedByCaller => {
                // SAFETY: this private variant is selected only by
                // enqueue_trusted_paged_decode, whose caller owns the proof.
                let args = unsafe {
                    TrustedBf16PagedBatchDecodeArgs::assume_metadata_valid(tensors.spec, args)
                };
                EngineCommand::Bf16PagedBatchDecodeTrustedMetadata { plan, args }
            }
        };
        let submission = self
            .queue
            .enqueue(command, external)
            .map_err(|error| {
                if matches!(
                    error.cause(),
                    EngineEnqueueCause::InFlightCapacityExceeded
                ) {
                    oxide_error(format!(
                        "Oxide decode has {MAX_IN_FLIGHT} commands in flight; drain completions before submitting more"
                    ))
                } else {
                    oxide_external_error("Oxide decode enqueue failed", error)
                }
            })?;
        let (completion, authority) = submission.into_parts();
        drop(authority);
        let engine_profile = completion.trace().host_profile().unwrap_or_default();
        Ok(ProfiledCompletion {
            completion,
            profile: EnqueueProfile {
                binding_host_nanoseconds,
                interop_host_nanoseconds: profile_elapsed_nanoseconds(interop_started),
                engine_total_host_nanoseconds: engine_profile.total_host_nanoseconds(),
                engine_setup_host_nanoseconds: engine_profile.setup_host_nanoseconds(),
                engine_pre_handoff_host_nanoseconds: engine_profile.pre_handoff_host_nanoseconds(),
                engine_provider_host_nanoseconds: engine_profile.provider_host_nanoseconds(),
                engine_provider_preflight_host_nanoseconds: engine_profile
                    .provider_preflight_host_nanoseconds(),
                engine_provider_metadata_host_nanoseconds: engine_profile
                    .provider_metadata_host_nanoseconds(),
                engine_provider_attention_host_nanoseconds: engine_profile
                    .provider_attention_host_nanoseconds(),
                engine_status_readback_host_nanoseconds: engine_profile
                    .status_readback_host_nanoseconds(),
                engine_post_handoff_host_nanoseconds: engine_profile
                    .post_handoff_host_nanoseconds(),
                ..EnqueueProfile::default()
            },
        })
    }
}

struct PendingDecode {
    completion: EngineCommandCompletion,
}

/// Read-only evidence from the single-stream Oxide decode runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OxidePagedDecodeStats {
    submitted: u64,
    completed: u64,
    failed: u64,
    last_operator: Option<EngineOperator>,
    last_layout: Option<PagedKvLayout>,
    last_algorithm: Option<EngineAlgorithm>,
    last_metadata_validation: Option<EngineMetadataValidation>,
    last_stream_handoff: Option<EngineStreamHandoff>,
    adapter_zero_copy: bool,
    external_regions: usize,
    adapter_device_to_device_copies: usize,
    profile_enabled: bool,
    enqueue_host_nanoseconds: u64,
    preparation_host_nanoseconds: u64,
    allocation_host_nanoseconds: u64,
    guard_host_nanoseconds: u64,
    binding_host_nanoseconds: u64,
    interop_host_nanoseconds: u64,
    engine_total_host_nanoseconds: u64,
    engine_setup_host_nanoseconds: u64,
    engine_pre_handoff_host_nanoseconds: u64,
    engine_provider_host_nanoseconds: u64,
    engine_provider_preflight_host_nanoseconds: u64,
    engine_provider_metadata_host_nanoseconds: u64,
    engine_provider_attention_host_nanoseconds: u64,
    engine_status_readback_host_nanoseconds: u64,
    engine_post_handoff_host_nanoseconds: u64,
    device_profile_enabled: bool,
    device_profiled_completions: u64,
    engine_total_device_nanoseconds: u64,
    engine_pre_handoff_device_nanoseconds: u64,
    engine_provider_device_nanoseconds: u64,
    engine_post_handoff_device_nanoseconds: u64,
    drain_host_nanoseconds: u64,
    drain_calls: u64,
}

impl OxidePagedDecodeStats {
    pub const fn submitted(self) -> u64 {
        self.submitted
    }

    /// Returns the number of settled commands, including failed commands.
    pub const fn completed(self) -> u64 {
        self.completed
    }

    pub const fn failed(self) -> u64 {
        self.failed
    }

    pub const fn last_operator(self) -> Option<EngineOperator> {
        self.last_operator
    }

    pub const fn last_layout(self) -> Option<PagedKvLayout> {
        self.last_layout
    }

    pub const fn last_algorithm(self) -> Option<EngineAlgorithm> {
        self.last_algorithm
    }

    pub const fn last_metadata_validation(self) -> Option<EngineMetadataValidation> {
        self.last_metadata_validation
    }

    pub const fn last_stream_handoff(self) -> Option<EngineStreamHandoff> {
        self.last_stream_handoff
    }

    pub const fn metadata_trusted_by_adapter(self) -> bool {
        matches!(
            self.last_metadata_validation,
            Some(EngineMetadataValidation::TrustedByAdapter)
        )
    }

    pub const fn adapter_zero_copy(self) -> bool {
        self.adapter_zero_copy
    }

    pub const fn external_regions(self) -> usize {
        self.external_regions
    }

    pub const fn adapter_device_to_device_copies(self) -> usize {
        self.adapter_device_to_device_copies
    }

    pub const fn profile_enabled(self) -> bool {
        self.profile_enabled
    }

    pub const fn enqueue_host_nanoseconds(self) -> u64 {
        self.enqueue_host_nanoseconds
    }

    pub const fn preparation_host_nanoseconds(self) -> u64 {
        self.preparation_host_nanoseconds
    }

    pub const fn allocation_host_nanoseconds(self) -> u64 {
        self.allocation_host_nanoseconds
    }

    pub const fn guard_host_nanoseconds(self) -> u64 {
        self.guard_host_nanoseconds
    }

    pub const fn binding_host_nanoseconds(self) -> u64 {
        self.binding_host_nanoseconds
    }

    pub const fn interop_host_nanoseconds(self) -> u64 {
        self.interop_host_nanoseconds
    }

    pub const fn engine_total_host_nanoseconds(self) -> u64 {
        self.engine_total_host_nanoseconds
    }

    pub const fn engine_setup_host_nanoseconds(self) -> u64 {
        self.engine_setup_host_nanoseconds
    }

    pub const fn engine_pre_handoff_host_nanoseconds(self) -> u64 {
        self.engine_pre_handoff_host_nanoseconds
    }

    pub const fn engine_provider_host_nanoseconds(self) -> u64 {
        self.engine_provider_host_nanoseconds
    }

    pub const fn engine_provider_preflight_host_nanoseconds(self) -> u64 {
        self.engine_provider_preflight_host_nanoseconds
    }

    pub const fn engine_provider_metadata_host_nanoseconds(self) -> u64 {
        self.engine_provider_metadata_host_nanoseconds
    }

    pub const fn engine_provider_attention_host_nanoseconds(self) -> u64 {
        self.engine_provider_attention_host_nanoseconds
    }

    pub const fn engine_status_readback_host_nanoseconds(self) -> u64 {
        self.engine_status_readback_host_nanoseconds
    }

    pub const fn engine_post_handoff_host_nanoseconds(self) -> u64 {
        self.engine_post_handoff_host_nanoseconds
    }

    pub const fn device_profile_enabled(self) -> bool {
        self.device_profile_enabled
    }

    pub const fn device_profiled_completions(self) -> u64 {
        self.device_profiled_completions
    }

    pub const fn engine_total_device_nanoseconds(self) -> u64 {
        self.engine_total_device_nanoseconds
    }

    pub const fn engine_pre_handoff_device_nanoseconds(self) -> u64 {
        self.engine_pre_handoff_device_nanoseconds
    }

    pub const fn engine_provider_device_nanoseconds(self) -> u64 {
        self.engine_provider_device_nanoseconds
    }

    pub const fn engine_post_handoff_device_nanoseconds(self) -> u64 {
        self.engine_post_handoff_device_nanoseconds
    }

    pub const fn drain_host_nanoseconds(self) -> u64 {
        self.drain_host_nanoseconds
    }

    pub const fn drain_calls(self) -> u64 {
        self.drain_calls
    }

    fn record_submission(&mut self, trace: &EngineExecutionTrace) {
        self.submitted = self.submitted.saturating_add(1);
        self.last_operator = Some(trace.operator());
        self.last_layout = trace.paged_kv_layout();
        self.last_algorithm = Some(trace.algorithm());
        self.last_metadata_validation = trace.metadata_validation();
        self.last_stream_handoff = Some(trace.stream_handoff());
        self.adapter_zero_copy = trace.is_adapter_zero_copy();
        self.external_regions = trace.memory().external_regions();
        self.adapter_device_to_device_copies = trace.adapter_device_to_device_copies();
    }

    fn record_completion(&mut self, trace: Option<&EngineExecutionTrace>, failed: bool) {
        self.completed = self.completed.saturating_add(1);
        if failed {
            self.failed = self.failed.saturating_add(1);
        }
        if let Some(profile) = trace.and_then(EngineExecutionTrace::device_profile) {
            self.device_profile_enabled = true;
            self.device_profiled_completions = self.device_profiled_completions.saturating_add(1);
            self.engine_total_device_nanoseconds = self
                .engine_total_device_nanoseconds
                .saturating_add(profile.total_device_nanoseconds());
            self.engine_pre_handoff_device_nanoseconds = self
                .engine_pre_handoff_device_nanoseconds
                .saturating_add(profile.pre_handoff_device_nanoseconds());
            self.engine_provider_device_nanoseconds = self
                .engine_provider_device_nanoseconds
                .saturating_add(profile.provider_device_nanoseconds());
            self.engine_post_handoff_device_nanoseconds = self
                .engine_post_handoff_device_nanoseconds
                .saturating_add(profile.post_handoff_device_nanoseconds());
        }
    }

    fn record_enqueue_profile(&mut self, duration: Duration, profile: EnqueueProfile) {
        self.profile_enabled = true;
        self.enqueue_host_nanoseconds = self
            .enqueue_host_nanoseconds
            .saturating_add(duration_nanoseconds(duration));
        self.preparation_host_nanoseconds = self
            .preparation_host_nanoseconds
            .saturating_add(profile.preparation_host_nanoseconds);
        self.allocation_host_nanoseconds = self
            .allocation_host_nanoseconds
            .saturating_add(profile.allocation_host_nanoseconds);
        self.guard_host_nanoseconds = self
            .guard_host_nanoseconds
            .saturating_add(profile.guard_host_nanoseconds);
        self.binding_host_nanoseconds = self
            .binding_host_nanoseconds
            .saturating_add(profile.binding_host_nanoseconds);
        self.interop_host_nanoseconds = self
            .interop_host_nanoseconds
            .saturating_add(profile.interop_host_nanoseconds);
        self.engine_total_host_nanoseconds = self
            .engine_total_host_nanoseconds
            .saturating_add(profile.engine_total_host_nanoseconds);
        self.engine_setup_host_nanoseconds = self
            .engine_setup_host_nanoseconds
            .saturating_add(profile.engine_setup_host_nanoseconds);
        self.engine_pre_handoff_host_nanoseconds = self
            .engine_pre_handoff_host_nanoseconds
            .saturating_add(profile.engine_pre_handoff_host_nanoseconds);
        self.engine_provider_host_nanoseconds = self
            .engine_provider_host_nanoseconds
            .saturating_add(profile.engine_provider_host_nanoseconds);
        self.engine_provider_preflight_host_nanoseconds = self
            .engine_provider_preflight_host_nanoseconds
            .saturating_add(profile.engine_provider_preflight_host_nanoseconds);
        self.engine_provider_metadata_host_nanoseconds = self
            .engine_provider_metadata_host_nanoseconds
            .saturating_add(profile.engine_provider_metadata_host_nanoseconds);
        self.engine_provider_attention_host_nanoseconds = self
            .engine_provider_attention_host_nanoseconds
            .saturating_add(profile.engine_provider_attention_host_nanoseconds);
        self.engine_status_readback_host_nanoseconds = self
            .engine_status_readback_host_nanoseconds
            .saturating_add(profile.engine_status_readback_host_nanoseconds);
        self.engine_post_handoff_host_nanoseconds = self
            .engine_post_handoff_host_nanoseconds
            .saturating_add(profile.engine_post_handoff_host_nanoseconds);
    }

    fn record_drain_profile(&mut self, duration: Duration) {
        self.profile_enabled = true;
        self.drain_host_nanoseconds = self
            .drain_host_nanoseconds
            .saturating_add(duration_nanoseconds(duration));
        self.drain_calls = self.drain_calls.saturating_add(1);
    }
}

/// An error returned while draining queued Oxide decode completions.
#[derive(Debug, thiserror::Error)]
pub enum OxidePagedDecodeDrainError {
    #[error("Oxide decode runtime is poisoned after {drained} completions settled")]
    RuntimePoisoned { drained: usize },
    #[error(
        "Oxide decode completion at FIFO position {failed_position} failed after {drained} completions settled: {source}"
    )]
    Completion {
        drained: usize,
        failed_position: usize,
        #[source]
        source: EngineCommandCompletionError,
    },
}

impl OxidePagedDecodeDrainError {
    pub const fn drained(&self) -> usize {
        match self {
            Self::RuntimePoisoned { drained } | Self::Completion { drained, .. } => *drained,
        }
    }

    /// Returns the one-based position of the first failed completion.
    pub const fn failed_position(&self) -> Option<usize> {
        match self {
            Self::RuntimePoisoned { .. } => None,
            Self::Completion {
                failed_position, ..
            } => Some(*failed_position),
        }
    }

    pub const fn cause(&self) -> Option<&EngineCommandFailure> {
        match self {
            Self::RuntimePoisoned { .. } => None,
            Self::Completion { source, .. } => Some(source.cause()),
        }
    }

    pub const fn trace(&self) -> Option<&EngineExecutionTrace> {
        match self {
            Self::RuntimePoisoned { .. } => None,
            Self::Completion { source, .. } => Some(source.trace()),
        }
    }
}

#[derive(Default)]
struct OxidePagedDecodeRuntimeState {
    runtime: Option<OxideRuntime>,
    pending: VecDeque<PendingDecode>,
    stats: OxidePagedDecodeStats,
}

impl OxidePagedDecodeRuntimeState {
    fn ensure_runtime(
        &mut self,
        key: RuntimeKey,
        ordinal: usize,
        stream: Arc<CudaStream>,
        direct_stream: bool,
    ) -> Result<&mut OxideRuntime> {
        if let Some(runtime) = self.runtime.as_ref() {
            if runtime.key != key {
                return Err(oxide_error(
                    "Oxide paged decode runtime is already bound to another CUDA context or stream",
                ));
            }
        }
        if self.runtime.is_none() {
            self.runtime = Some(OxideRuntime::new(key, ordinal, stream, direct_stream)?);
        }
        self.runtime
            .as_mut()
            .ok_or_else(|| oxide_error("Oxide decode runtime initialization did not complete"))
    }
}

/// A model-owned Oxide paged-decode runtime bound lazily to one CUDA stream.
pub struct OxidePagedDecodeRuntime {
    direct_stream: bool,
    lifecycle: Mutex<()>,
    state: Mutex<OxidePagedDecodeRuntimeState>,
}

impl Default for OxidePagedDecodeRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl OxidePagedDecodeRuntime {
    /// Creates an unbound runtime. The first enqueue binds its CUDA context and stream.
    pub fn new() -> Self {
        Self::new_with_direct_stream(oxide_direct_stream_enabled())
    }

    /// Creates an unbound runtime that submits directly on the engine stream.
    pub fn new_direct() -> Self {
        Self::new_with_direct_stream(true)
    }

    fn new_with_direct_stream(direct_stream: bool) -> Self {
        Self {
            direct_stream,
            lifecycle: Mutex::new(()),
            state: Mutex::new(OxidePagedDecodeRuntimeState::default()),
        }
    }

    fn lock_lifecycle(&self) -> Result<MutexGuard<'_, ()>> {
        self.lifecycle
            .lock()
            .map_err(|_| oxide_error("Oxide decode runtime lifecycle is poisoned"))
    }

    fn lock_lifecycle_for_drain(
        &self,
        drained: usize,
    ) -> std::result::Result<MutexGuard<'_, ()>, OxidePagedDecodeDrainError> {
        self.lifecycle
            .lock()
            .map_err(|_| OxidePagedDecodeDrainError::RuntimePoisoned { drained })
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, OxidePagedDecodeRuntimeState>> {
        self.state
            .lock()
            .map_err(|_| oxide_error("Oxide decode runtime is poisoned"))
    }

    fn lock_state_for_drain(
        &self,
        drained: usize,
    ) -> std::result::Result<MutexGuard<'_, OxidePagedDecodeRuntimeState>, OxidePagedDecodeDrainError>
    {
        self.state
            .lock()
            .map_err(|_| OxidePagedDecodeDrainError::RuntimePoisoned { drained })
    }
}

struct DecodeTensors<'a> {
    query: &'a Tensor,
    key_cache: &'a Tensor,
    value_cache: &'a Tensor,
    page_indptr: &'a Tensor,
    page_indices: &'a Tensor,
    last_page_len: &'a Tensor,
    metadata_status: &'a Tensor,
    output: &'a Tensor,
    lse: &'a Tensor,
    logical_page_count: usize,
    spec: Bf16PagedBatchDecodeSpec,
}

#[derive(Clone, Copy)]
struct DecodePointers {
    query: u64,
    key_pages: u64,
    value_pages: u64,
    page_indptr: u64,
    page_indices: u64,
    last_page_len: u64,
    metadata_status: u64,
    output: u64,
    lse: u64,
}

struct DecodeLaunch<'a> {
    runtime: &'a OxidePagedDecodeRuntime,
    key: RuntimeKey,
    plan: Bf16PagedBatchDecodePlan,
    tensors: DecodeTensors<'a>,
    metadata_validation: AdapterMetadataValidation,
    completion: RefCell<Option<ProfiledCompletion>>,
}

struct OutputInplace<'a> {
    launch: &'a DecodeLaunch<'a>,
}

impl InplaceOp1 for OutputInplace<'_> {
    fn name(&self) -> &'static str {
        "oxide-paged-decode-output"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(oxide_error("Oxide paged decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let stream = require_runtime_stream(storage, self.launch.key, "output")?;
        let slice = storage.as_cuda_slice_mut::<bf16>()?;
        let (base, output_guard) = slice.device_ptr_mut(&stream);
        let output = offset_pointer::<bf16>(base, layout.start_offset())?;
        self.launch.tensors.lse.inplace_op1(&LseInplace {
            launch: self.launch,
            output,
        })?;
        drop(output_guard);
        Ok(())
    }
}

struct LseInplace<'a> {
    launch: &'a DecodeLaunch<'a>,
    output: u64,
}

impl InplaceOp1 for LseInplace<'_> {
    fn name(&self) -> &'static str {
        "oxide-paged-decode-lse"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(oxide_error("Oxide paged decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let stream = require_runtime_stream(storage, self.launch.key, "LSE")?;
        let slice = storage.as_cuda_slice_mut::<f32>()?;
        let (base, lse_guard) = slice.device_ptr_mut(&stream);
        let lse = offset_pointer::<f32>(base, layout.start_offset())?;
        self.launch
            .tensors
            .metadata_status
            .inplace_op1(&StatusInplace {
                launch: self.launch,
                output: self.output,
                lse,
            })?;
        drop(lse_guard);
        Ok(())
    }
}

struct StatusInplace<'a> {
    launch: &'a DecodeLaunch<'a>,
    output: u64,
    lse: u64,
}

impl InplaceOp1 for StatusInplace<'_> {
    fn name(&self) -> &'static str {
        "oxide-paged-decode-status"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(oxide_error("Oxide paged decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let stream = require_runtime_stream(storage, self.launch.key, "metadata status")?;
        let slice = storage.as_cuda_slice_mut::<i32>()?;
        let (base, status_guard) = slice.device_ptr_mut(&stream);
        let metadata_status = offset_pointer::<i32>(base, layout.start_offset())?;
        let completion = enqueue_with_read_guards(
            self.launch,
            DecodePointers {
                query: 0,
                key_pages: 0,
                value_pages: 0,
                page_indptr: 0,
                page_indices: 0,
                last_page_len: 0,
                metadata_status,
                output: self.output,
                lse: self.lse,
            },
            &stream,
        )?;
        drop(status_guard);
        *self.launch.completion.borrow_mut() = Some(completion);
        Ok(())
    }
}

fn enqueue_with_read_guards(
    launch: &DecodeLaunch<'_>,
    mut pointers: DecodePointers,
    stream: &CudaStream,
) -> Result<ProfiledCompletion> {
    let guard_started = oxide_profile_enabled().then(Instant::now);
    let tensors = &launch.tensors;
    let (query_storage, query_layout) = tensors.query.storage_and_layout();
    let (key_storage, key_layout) = tensors.key_cache.storage_and_layout();
    let (value_storage, value_layout) = tensors.value_cache.storage_and_layout();
    let (indptr_storage, indptr_layout) = tensors.page_indptr.storage_and_layout();
    let (indices_storage, indices_layout) = tensors.page_indices.storage_and_layout();
    let (last_storage, last_layout) = tensors.last_page_len.storage_and_layout();
    let query_storage = require_cuda_storage(&query_storage, launch.key, "query")?;
    let key_storage = require_cuda_storage(&key_storage, launch.key, "key cache")?;
    let value_storage = require_cuda_storage(&value_storage, launch.key, "value cache")?;
    let indptr_storage = require_cuda_storage(&indptr_storage, launch.key, "page indptr")?;
    let indices_storage = require_cuda_storage(&indices_storage, launch.key, "page indices")?;
    let last_storage = require_cuda_storage(&last_storage, launch.key, "last page length")?;

    let (query, query_guard) = super::slice_ptr_on_stream(
        query_storage.as_cuda_slice::<bf16>()?,
        query_layout.start_offset(),
        stream,
    );
    let (key_pages, key_guard) = super::slice_ptr_on_stream(
        key_storage.as_cuda_slice::<bf16>()?,
        key_layout.start_offset(),
        stream,
    );
    let (value_pages, value_guard) = super::slice_ptr_on_stream(
        value_storage.as_cuda_slice::<bf16>()?,
        value_layout.start_offset(),
        stream,
    );
    let (page_indptr, indptr_guard) = super::slice_ptr_on_stream(
        indptr_storage.as_cuda_slice::<i32>()?,
        indptr_layout.start_offset(),
        stream,
    );
    let (page_indices, indices_guard) = super::slice_ptr_on_stream(
        indices_storage.as_cuda_slice::<i32>()?,
        indices_layout.start_offset(),
        stream,
    );
    let (last_page_len, last_guard) = super::slice_ptr_on_stream(
        last_storage.as_cuda_slice::<i32>()?,
        last_layout.start_offset(),
        stream,
    );
    pointers.query = query;
    pointers.key_pages = key_pages;
    pointers.value_pages = value_pages;
    pointers.page_indptr = page_indptr;
    pointers.page_indices = page_indices;
    pointers.last_page_len = last_page_len;

    let profiled_completion = {
        let mut state = launch.runtime.lock_state()?;
        let runtime = state
            .runtime
            .as_mut()
            .filter(|runtime| runtime.key == launch.key)
            .ok_or_else(|| oxide_error("Oxide decode runtime binding changed before enqueue"))?;
        let guard_host_nanoseconds = profile_elapsed_nanoseconds(guard_started);
        let mut profiled_completion =
            runtime.enqueue(&launch.plan, tensors, pointers, launch.metadata_validation)?;
        profiled_completion.profile.guard_host_nanoseconds = guard_host_nanoseconds;
        profiled_completion
    };
    drop(last_guard);
    drop(indices_guard);
    drop(indptr_guard);
    drop(value_guard);
    drop(key_guard);
    drop(query_guard);
    Ok(profiled_completion)
}

impl OxidePagedDecodeRuntime {
    /// Enqueues one BF16, D128, page-size-16 HND paged decode.
    ///
    /// `logical_page_count` is the CSR terminal value. Oxide reads only this
    /// prefix of a padded `paged_kv_indices` tensor.
    ///
    /// # Safety
    ///
    /// During this call, the model runner must own exclusive submission
    /// authority for the ordinary CUDA stream and every writable span. It must
    /// not use tensor aliases concurrently. Event tracking must remain enabled.
    /// Before `drain` returns, later access must use the same stream or Candle's
    /// tracked pointer APIs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_paged_decode(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        paged_kv_indptr: &Tensor,
        paged_kv_indices: &Tensor,
        logical_page_count: usize,
        paged_kv_last_page_len: &Tensor,
    ) -> Result<Tensor> {
        self.enqueue_paged_decode_impl(
            query,
            key_cache,
            value_cache,
            paged_kv_indptr,
            paged_kv_indices,
            logical_page_count,
            paged_kv_last_page_len,
            AdapterMetadataValidation::DeviceChecked,
        )
    }

    /// Enqueues after the caller has established paged-metadata validity.
    ///
    /// This skips Oxide's device metadata validator and status readback.
    ///
    /// # Safety
    ///
    /// In addition to the stream and aliasing requirements of
    /// [`Self::enqueue_paged_decode`], the caller must guarantee that the CSR
    /// metadata satisfies the Oxide paged-decode contract and cannot be
    /// mutated through command completion. In particular, page indices must
    /// refer to physical pages in `key_cache` and `value_cache`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_trusted_paged_decode(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        paged_kv_indptr: &Tensor,
        paged_kv_indices: &Tensor,
        logical_page_count: usize,
        paged_kv_last_page_len: &Tensor,
    ) -> Result<Tensor> {
        self.enqueue_paged_decode_impl(
            query,
            key_cache,
            value_cache,
            paged_kv_indptr,
            paged_kv_indices,
            logical_page_count,
            paged_kv_last_page_len,
            AdapterMetadataValidation::TrustedByCaller,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_paged_decode_impl(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        paged_kv_indptr: &Tensor,
        paged_kv_indices: &Tensor,
        logical_page_count: usize,
        paged_kv_last_page_len: &Tensor,
        metadata_validation: AdapterMetadataValidation,
    ) -> Result<Tensor> {
        let profile_started = oxide_profile_enabled().then(Instant::now);
        let preparation_started = oxide_profile_enabled().then(Instant::now);
        validate_inputs(
            query,
            key_cache,
            value_cache,
            paged_kv_indptr,
            paged_kv_indices,
            logical_page_count,
            paged_kv_last_page_len,
        )?;
        let _lifecycle = self.lock_lifecycle()?;
        let device = query.device().as_cuda_device()?;
        if !device.is_event_tracking() {
            return Err(oxide_error(
                "Oxide paged decode requires CUDA event tracking",
            ));
        }
        let stream = device.cuda_stream();
        if stream.capture_status().map_err(|error| {
            oxide_external_error("failed to query CUDA stream capture status", error)
        })? != candle_core::cuda::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(oxide_error(
                "Oxide paged decode does not support active CUDA graph capture",
            ));
        }
        let key = runtime_key(&stream);
        if key.stream <= 2 {
            return Err(oxide_error(
                "Oxide paged decode requires Device::new_cuda_with_stream",
            ));
        }
        let ordinal = stream.context().ordinal();
        let (batch_size, num_query_heads, head_dim) = query.dims3()?;
        let (max_num_pages, num_kv_heads, page_size, _) = key_cache.dims4()?;
        let spec = Bf16PagedBatchDecodeSpec::new(
            batch_size,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            head_dim,
            page_size,
            PagedKvLayout::Hnd,
        )
        .map_err(|error| oxide_external_error("invalid Oxide paged-decode contract", error))?;
        let plan_key = PlanKey {
            batch_size,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
        };
        let plan = self
            .lock_state()?
            .ensure_runtime(key, ordinal, Arc::clone(&stream), self.direct_stream)?
            .plan(plan_key, spec)?;
        let preparation_host_nanoseconds = profile_elapsed_nanoseconds(preparation_started);
        let allocation_started = oxide_profile_enabled().then(Instant::now);
        let output = unsafe {
            Tensor::empty(
                (batch_size, num_query_heads, HEAD_DIM),
                DType::BF16,
                query.device(),
            )?
        };
        let lse =
            unsafe { Tensor::empty((batch_size, num_query_heads), DType::F32, query.device())? };
        let metadata_status = unsafe {
            Tensor::empty(
                plan.metadata_status_required_numel(),
                DType::I32,
                query.device(),
            )?
        };
        let allocation_host_nanoseconds = profile_elapsed_nanoseconds(allocation_started);
        let launch = DecodeLaunch {
            runtime: self,
            key,
            plan,
            tensors: DecodeTensors {
                query,
                key_cache,
                value_cache,
                page_indptr: paged_kv_indptr,
                page_indices: paged_kv_indices,
                last_page_len: paged_kv_last_page_len,
                metadata_status: &metadata_status,
                output: &output,
                lse: &lse,
                logical_page_count,
                spec,
            },
            metadata_validation,
            completion: RefCell::new(None),
        };
        output.inplace_op1(&OutputInplace { launch: &launch })?;
        let mut profiled_completion = launch
            .completion
            .into_inner()
            .ok_or_else(|| oxide_error("Oxide decode returned without a completion"))?;
        profiled_completion.profile.preparation_host_nanoseconds = preparation_host_nanoseconds;
        profiled_completion.profile.allocation_host_nanoseconds = allocation_host_nanoseconds;
        let mut state = self.lock_state()?;
        state
            .stats
            .record_submission(profiled_completion.completion.trace());
        state.pending.push_back(PendingDecode {
            completion: profiled_completion.completion,
        });
        if let Some(started) = profile_started {
            state
                .stats
                .record_enqueue_profile(started.elapsed(), profiled_completion.profile);
        }
        Ok(output)
    }

    /// Waits for all queued decode commands in submission order.
    pub fn drain(&self) -> std::result::Result<usize, OxidePagedDecodeDrainError> {
        let profile_started = oxide_profile_enabled().then(Instant::now);
        let result = self.drain_inner();
        if let Some(started) = profile_started {
            if let Ok(mut state) = self.state.lock() {
                state.stats.record_drain_profile(started.elapsed());
            }
        }
        result
    }

    fn drain_inner(&self) -> std::result::Result<usize, OxidePagedDecodeDrainError> {
        let mut drained = 0;
        let mut first_error = None;
        let _lifecycle = self.lock_lifecycle_for_drain(drained)?;
        loop {
            let pending = self.lock_state_for_drain(drained)?.pending.pop_front();
            let Some(pending) = pending else {
                return match first_error {
                    Some((failed_position, source)) => {
                        Err(OxidePagedDecodeDrainError::Completion {
                            drained,
                            failed_position,
                            source,
                        })
                    }
                    None => Ok(drained),
                };
            };
            let result = pending.completion.wait();
            drained += 1;
            self.lock_state_for_drain(drained)?
                .stats
                .record_completion(result.as_ref().ok(), result.is_err());
            if let Err(source) = result {
                if first_error.is_none() {
                    first_error = Some((drained, source));
                }
            }
        }
    }

    /// Returns a snapshot of provider-hit and completion evidence.
    pub fn stats(&self) -> Result<OxidePagedDecodeStats> {
        Ok(self.lock_state()?.stats)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    page_indptr: &Tensor,
    page_indices: &Tensor,
    logical_page_count: usize,
    last_page_len: &Tensor,
) -> Result<()> {
    require_dtype(query, DType::BF16, "query")?;
    require_dtype(key_cache, DType::BF16, "key cache")?;
    require_dtype(value_cache, DType::BF16, "value cache")?;
    require_dtype(page_indptr, DType::I32, "page indptr")?;
    require_dtype(page_indices, DType::I32, "page indices")?;
    require_dtype(last_page_len, DType::I32, "last page length")?;
    for (name, tensor) in [
        ("query", query),
        ("key cache", key_cache),
        ("value cache", value_cache),
        ("page indptr", page_indptr),
        ("page indices", page_indices),
        ("last page length", last_page_len),
    ] {
        if !tensor.layout().is_contiguous() {
            return Err(oxide_error(format!(
                "Oxide paged decode requires contiguous {name}"
            )));
        }
        if tensor.device().location() != query.device().location() {
            return Err(oxide_error(format!(
                "Oxide paged decode requires {name} on the query device"
            )));
        }
    }
    let (batch_size, _, head_dim) = query.dims3()?;
    let key_shape = key_cache.dims4()?;
    if value_cache.dims4()? != key_shape {
        return Err(oxide_error("Oxide paged decode cache shapes do not match"));
    }
    if head_dim != HEAD_DIM || key_shape.2 != PAGE_SIZE || key_shape.3 != HEAD_DIM {
        return Err(oxide_error(
            "Oxide paged decode requires BF16/D128/page16/HND cache tensors",
        ));
    }
    if page_indptr.dims1()? != batch_size + 1 || last_page_len.dims1()? != batch_size {
        return Err(oxide_error(
            "Oxide paged decode received invalid CSR metadata shapes",
        ));
    }
    let physical_page_indices = page_indices.dims1()?;
    if logical_page_count < batch_size || logical_page_count > physical_page_indices {
        return Err(oxide_error(format!(
            "logical page count {logical_page_count} is outside {batch_size}..={physical_page_indices}"
        )));
    }
    Ok(())
}

fn require_dtype(tensor: &Tensor, expected: DType, name: &str) -> Result<()> {
    if tensor.dtype() == expected {
        Ok(())
    } else {
        Err(oxide_error(format!(
            "Oxide paged decode requires {name} dtype {expected:?}, got {:?}",
            tensor.dtype()
        )))
    }
}

fn require_cuda_storage<'a>(
    storage: &'a Storage,
    key: RuntimeKey,
    name: &str,
) -> Result<&'a CudaStorage> {
    let Storage::Cuda(storage) = storage else {
        return Err(oxide_error(format!(
            "Oxide paged decode requires CUDA {name} storage"
        )));
    };
    require_runtime_stream(storage, key, name)?;
    Ok(storage)
}

fn require_runtime_stream(
    storage: &CudaStorage,
    key: RuntimeKey,
    name: &str,
) -> Result<Arc<CudaStream>> {
    let stream = storage.device().cuda_stream();
    if runtime_key(&stream) != key {
        return Err(oxide_error(format!(
            "Oxide paged decode requires {name} on one CUDA context and stream"
        )));
    }
    Ok(stream)
}

fn runtime_key(stream: &CudaStream) -> RuntimeKey {
    RuntimeKey {
        context: stream.context().cu_ctx() as usize,
        stream: stream.cu_stream() as usize,
    }
}

fn oxide_stream(stream: &CudaStream) -> oxide_cuda_core::sys::CUstream {
    stream.cu_stream().cast()
}

fn tensor_lease(tensor: &Tensor) -> Arc<dyn Any + Send + Sync> {
    Arc::new(tensor.clone())
}

fn offset_pointer<T>(base: u64, offset: usize) -> Result<u64> {
    let bytes = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| oxide_error("CUDA tensor offset overflow"))?;
    base.checked_add(bytes as u64)
        .ok_or_else(|| oxide_error("CUDA device pointer overflow"))
}

fn oxide_external_error(context: &str, error: impl std::fmt::Display) -> Error {
    oxide_error(format!("{context}: {error}"))
}

fn oxide_profile_enabled() -> bool {
    *OXIDE_PROFILE_ENABLED.get_or_init(|| std::env::var(OXIDE_PROFILE_ENV).as_deref() == Ok("1"))
}

fn oxide_device_profile_enabled() -> bool {
    *OXIDE_DEVICE_PROFILE_ENABLED
        .get_or_init(|| std::env::var(OXIDE_DEVICE_PROFILE_ENV).as_deref() == Ok("1"))
}

fn oxide_direct_stream_enabled() -> bool {
    *OXIDE_DIRECT_STREAM_ENABLED
        .get_or_init(|| std::env::var(OXIDE_DIRECT_STREAM_ENV).as_deref() == Ok("1"))
}

fn duration_nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn profile_elapsed_nanoseconds(started: Option<Instant>) -> u64 {
    started
        .map(|started| started.elapsed())
        .map(duration_nanoseconds)
        .unwrap_or_default()
}

fn oxide_error(message: impl Into<String>) -> Error {
    Error::Msg(message.into()).bt()
}
