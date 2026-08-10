use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{CudaStream, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, DType, Error, InplaceOp1, Layout, Result, Storage, Tensor,
};
use half::bf16;
use loom_cuda_core::CudaContext;
use loom_infer::{Bf16PagedBatchDecodeSpec, PagedKvLayout};
use loom_infer_cuda::attention::{
    Bf16PagedBatchDecodeArgs, Bf16PagedBatchDecodePlan, DecodeProvider,
};
use loom_infer_cuda::interop::{
    EngineAlgorithm, EngineCommand, EngineCommandCompletion, EngineEnqueueCause,
    EngineExecutionTrace, EngineExternalBindings, EngineInteropQueue, EngineOperator,
    ExternalCudaStream, StreamOrderedEngineAuthority,
};
use loom_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const HEAD_DIM: usize = 128;
const PAGE_SIZE: usize = 16;
const COMMAND_CAPACITY: usize = 3;
const MAX_IN_FLIGHT: usize = 128;
const BINDING_COUNT: usize = 9;

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

struct CandleStreamLease {
    _stream: Arc<CudaStream>,
}

struct CandleStreamAuthority {
    stream: Arc<CudaStream>,
}

// SAFETY: this private authority is created only while the caller of
// `loom_paged_decode` upholds its exclusive model-runner submission contract.
unsafe impl StreamOrderedEngineAuthority for CandleStreamAuthority {
    fn submission_stream(&self) -> loom_cuda_core::sys::CUstream {
        loom_stream(&self.stream)
    }
}

struct LoomRuntime {
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    provider: DecodeProvider,
    queue: EngineInteropQueue,
    plans: HashMap<PlanKey, Bf16PagedBatchDecodePlan>,
}

impl LoomRuntime {
    fn new(key: RuntimeKey, ordinal: usize, stream: Arc<CudaStream>) -> Result<Self> {
        if key.stream <= 2 {
            return Err(loom_error(
                "Loom paged decode requires an ordinary CUDA stream",
            ));
        }
        let context = CudaContext::new(ordinal).map_err(|error| {
            loom_external_error("failed to retain the CUDA primary context", error)
        })?;
        if context.cu_ctx() as usize != key.context {
            return Err(loom_error(
                "Candle and Loom resolved different CUDA contexts",
            ));
        }
        let lease = Arc::new(CandleStreamLease {
            _stream: Arc::clone(&stream),
        });
        // SAFETY: `stream` is retained by both this runtime and `lease`. The
        // public unsafe call contract supplies stream submission exclusivity.
        let external = unsafe {
            ExternalCudaStream::from_raw_parts(loom_stream(&stream), Arc::clone(&context), lease)
        }
        .map_err(|error| loom_external_error("failed to import the Candle stream", error))?;
        let provider = DecodeProvider::load(&context)
            .map_err(|error| loom_external_error("failed to load Loom decode kernels", error))?;
        let queue = EngineInteropQueue::new(external, COMMAND_CAPACITY, MAX_IN_FLIGHT)
            .map_err(|error| loom_external_error("failed to create the Loom queue", error))?;
        Ok(Self {
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
        let plan = self
            .provider
            .plan_bf16_paged_batch(spec)
            .map_err(|error| loom_external_error("failed to create the Loom decode plan", error))?;
        self.plans.insert(key, plan.clone());
        Ok(plan)
    }

    fn enqueue(
        &mut self,
        plan: &Bf16PagedBatchDecodePlan,
        tensors: &DecodeTensors<'_>,
        pointers: DecodePointers,
    ) -> Result<EngineCommandCompletion> {
        let mut bindings = self
            .queue
            .bindings(BINDING_COUNT)
            .map_err(|error| loom_external_error("failed to allocate Loom bindings", error))?;
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
        .map_err(|error| loom_external_error("invalid query region", error))?;
        // SAFETY: see the query region construction above.
        let key_pages = unsafe {
            ReadDeviceRegion::<bf16>::from_external_parts(
                pointers.key_pages,
                tensors.spec.kv_pages_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.key_cache),
            )
        }
        .map_err(|error| loom_external_error("invalid key-cache region", error))?;
        // SAFETY: see the query region construction above.
        let value_pages = unsafe {
            ReadDeviceRegion::<bf16>::from_external_parts(
                pointers.value_pages,
                tensors.spec.kv_pages_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.value_cache),
            )
        }
        .map_err(|error| loom_external_error("invalid value-cache region", error))?;
        // SAFETY: see the query region construction above.
        let page_indptr = unsafe {
            ReadDeviceRegion::<i32>::from_external_parts(
                pointers.page_indptr,
                tensors.spec.page_indptr_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.page_indptr),
            )
        }
        .map_err(|error| loom_external_error("invalid page-indptr region", error))?;
        // SAFETY: only the logical CSR prefix is exposed to Loom. The full
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
        .map_err(|error| loom_external_error("invalid page-indices region", error))?;
        // SAFETY: see the query region construction above.
        let last_page_len = unsafe {
            ReadDeviceRegion::<i32>::from_external_parts(
                pointers.last_page_len,
                tensors.spec.last_page_len_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.last_page_len),
            )
        }
        .map_err(|error| loom_external_error("invalid last-page-len region", error))?;
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
        .map_err(|error| loom_external_error("invalid metadata-status region", error))?;
        // SAFETY: see the metadata-status region construction above.
        let output = unsafe {
            ReadWriteDeviceRegion::<bf16>::from_external_parts(
                pointers.output,
                tensors.spec.output_numel(),
                Arc::clone(&context),
                tensor_lease(tensors.output),
            )
        }
        .map_err(|error| loom_external_error("invalid output region", error))?;
        // SAFETY: see the metadata-status region construction above.
        let lse = unsafe {
            ReadWriteDeviceRegion::<f32>::from_external_parts(
                pointers.lse,
                tensors.spec.lse_numel(),
                context,
                tensor_lease(tensors.lse),
            )
        }
        .map_err(|error| loom_external_error("invalid LSE region", error))?;

        let query = bindings
            .bind_read_region(query)
            .map_err(|error| loom_external_error("failed to bind query", error))?;
        let key_pages = bindings
            .bind_read_region(key_pages)
            .map_err(|error| loom_external_error("failed to bind key cache", error))?;
        let value_pages = bindings
            .bind_read_region(value_pages)
            .map_err(|error| loom_external_error("failed to bind value cache", error))?;
        let page_indptr = bindings
            .bind_read_region(page_indptr)
            .map_err(|error| loom_external_error("failed to bind page indptr", error))?;
        let page_indices = bindings
            .bind_read_region(page_indices)
            .map_err(|error| loom_external_error("failed to bind page indices", error))?;
        let last_page_len = bindings
            .bind_read_region(last_page_len)
            .map_err(|error| loom_external_error("failed to bind last page length", error))?;
        let metadata_status = bindings
            .bind_read_write_region(metadata_status)
            .map_err(|error| loom_external_error("failed to bind metadata status", error))?;
        let output = bindings
            .bind_read_write_region(output)
            .map_err(|error| loom_external_error("failed to bind output", error))?;
        let lse = bindings
            .bind_read_write_region(lse)
            .map_err(|error| loom_external_error("failed to bind LSE", error))?;
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
                .map_err(|error| loom_external_error("failed to couple Candle bindings", error))?;
        let submission = self
            .queue
            .enqueue(EngineCommand::Bf16PagedBatchDecode { plan, args }, external)
            .map_err(|error| {
                if matches!(
                    error.cause(),
                    EngineEnqueueCause::InFlightCapacityExceeded
                ) {
                    loom_error(format!(
                        "Loom decode has {MAX_IN_FLIGHT} commands in flight; drain completions before submitting more"
                    ))
                } else {
                    loom_external_error("Loom decode enqueue failed", error)
                }
            })?;
        let (completion, authority) = submission.into_parts();
        drop(authority);
        Ok(completion)
    }
}

struct PendingDecode {
    completion: EngineCommandCompletion,
}

/// Read-only evidence from the single-stream Loom decode runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoomPagedDecodeStats {
    submitted: u64,
    completed: u64,
    failed: u64,
    last_operator: Option<EngineOperator>,
    last_layout: Option<PagedKvLayout>,
    last_algorithm: Option<EngineAlgorithm>,
    adapter_zero_copy: bool,
    external_regions: usize,
    adapter_device_to_device_copies: usize,
}

impl LoomPagedDecodeStats {
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

    pub const fn adapter_zero_copy(self) -> bool {
        self.adapter_zero_copy
    }

    pub const fn external_regions(self) -> usize {
        self.external_regions
    }

    pub const fn adapter_device_to_device_copies(self) -> usize {
        self.adapter_device_to_device_copies
    }

    fn record_submission(&mut self, trace: &EngineExecutionTrace) {
        self.submitted = self.submitted.saturating_add(1);
        self.last_operator = Some(trace.operator());
        self.last_layout = trace.paged_kv_layout();
        self.last_algorithm = Some(trace.algorithm());
        self.adapter_zero_copy = trace.is_adapter_zero_copy();
        self.external_regions = trace.memory().external_regions();
        self.adapter_device_to_device_copies = trace.adapter_device_to_device_copies();
    }

    fn record_completion(&mut self, failed: bool) {
        self.completed = self.completed.saturating_add(1);
        if failed {
            self.failed = self.failed.saturating_add(1);
        }
    }
}

#[derive(Default)]
struct RuntimeRegistry {
    runtimes: HashMap<RuntimeKey, LoomRuntime>,
    pending: VecDeque<PendingDecode>,
    stats: LoomPagedDecodeStats,
}

impl RuntimeRegistry {
    fn ensure_runtime(
        &mut self,
        key: RuntimeKey,
        ordinal: usize,
        stream: Arc<CudaStream>,
    ) -> Result<()> {
        if self.runtimes.contains_key(&key) {
            return Ok(());
        }
        if !self.runtimes.is_empty() {
            return Err(loom_error(
                "Loom paged decode currently supports one CUDA device and one ordinary stream",
            ));
        }
        let runtime = LoomRuntime::new(key, ordinal, stream)?;
        self.runtimes.insert(key, runtime);
        Ok(())
    }
}

static RUNTIMES: OnceLock<Mutex<RuntimeRegistry>> = OnceLock::new();

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
    key: RuntimeKey,
    plan: Bf16PagedBatchDecodePlan,
    tensors: DecodeTensors<'a>,
    completion: RefCell<Option<EngineCommandCompletion>>,
}

struct OutputInplace<'a> {
    launch: &'a DecodeLaunch<'a>,
}

impl InplaceOp1 for OutputInplace<'_> {
    fn name(&self) -> &'static str {
        "loom-paged-decode-output"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(loom_error("Loom paged decode requires CUDA storage"))
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
        "loom-paged-decode-lse"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(loom_error("Loom paged decode requires CUDA storage"))
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
        "loom-paged-decode-status"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(loom_error("Loom paged decode requires CUDA storage"))
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
) -> Result<EngineCommandCompletion> {
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

    let completion = {
        let mut registry = lock_registry()?;
        let runtime = registry
            .runtimes
            .get_mut(&launch.key)
            .ok_or_else(|| loom_error("Loom runtime disappeared before enqueue"))?;
        runtime.enqueue(&launch.plan, tensors, pointers)?
    };
    drop(last_guard);
    drop(indices_guard);
    drop(indptr_guard);
    drop(value_guard);
    drop(key_guard);
    drop(query_guard);
    Ok(completion)
}

/// Enqueues one BF16, D128, page-size-16 HND paged decode on Loom.
///
/// `logical_page_count` is the CSR terminal value. Only that prefix of a
/// padded `paged_kv_indices` tensor is exposed to Loom.
///
/// # Safety
///
/// During this call, the model runner must own exclusive submission authority
/// for the ordinary CUDA stream and exclusive access to every writable span.
/// It must not use aliases of these tensors concurrently. Event tracking must
/// remain enabled. Until the completion drain, later access must use the same
/// stream or Candle's tracked pointer APIs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn loom_paged_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    logical_page_count: usize,
    paged_kv_last_page_len: &Tensor,
) -> Result<Tensor> {
    validate_inputs(
        query,
        key_cache,
        value_cache,
        paged_kv_indptr,
        paged_kv_indices,
        logical_page_count,
        paged_kv_last_page_len,
    )?;
    let device = query.device().as_cuda_device()?;
    if !device.is_event_tracking() {
        return Err(loom_error("Loom paged decode requires CUDA event tracking"));
    }
    let stream = device.cuda_stream();
    if stream.capture_status().map_err(|error| {
        loom_external_error("failed to query CUDA stream capture status", error)
    })? != candle_core::cuda::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err(loom_error(
            "Loom paged decode does not support active CUDA graph capture",
        ));
    }
    let key = runtime_key(&stream);
    if key.stream <= 2 {
        return Err(loom_error(
            "Loom paged decode requires Device::new_cuda_with_stream",
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
    .map_err(|error| loom_external_error("invalid Loom paged-decode contract", error))?;
    let plan_key = PlanKey {
        batch_size,
        max_num_pages,
        num_query_heads,
        num_kv_heads,
    };
    let plan = {
        let mut registry = lock_registry()?;
        registry.ensure_runtime(key, ordinal, Arc::clone(&stream))?;
        registry
            .runtimes
            .get_mut(&key)
            .expect("the runtime was inserted above")
            .plan(plan_key, spec)?
    };
    let output = unsafe {
        Tensor::empty(
            (batch_size, num_query_heads, HEAD_DIM),
            DType::BF16,
            query.device(),
        )?
    };
    let lse = unsafe { Tensor::empty((batch_size, num_query_heads), DType::F32, query.device())? };
    let metadata_status = unsafe {
        Tensor::empty(
            plan.metadata_status_required_numel(),
            DType::I32,
            query.device(),
        )?
    };
    let launch = DecodeLaunch {
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
        completion: RefCell::new(None),
    };
    output.inplace_op1(&OutputInplace { launch: &launch })?;
    let completion = launch
        .completion
        .into_inner()
        .ok_or_else(|| loom_error("Loom decode returned without a completion"))?;
    let mut registry = lock_registry()?;
    registry.stats.record_submission(completion.trace());
    registry.pending.push_back(PendingDecode { completion });
    Ok(output)
}

/// Waits for all queued Loom decode commands in submission order.
pub fn drain_loom_paged_decode_completions() -> Result<usize> {
    let mut drained = 0;
    let mut first_error = None;
    loop {
        let pending = {
            let mut registry = lock_registry()?;
            registry.pending.pop_front()
        };
        let Some(pending) = pending else {
            return match first_error {
                Some(error) => Err(error),
                None => Ok(drained),
            };
        };
        let result = pending.completion.wait();
        lock_registry()?.stats.record_completion(result.is_err());
        if let Err(error) = result {
            if first_error.is_none() {
                first_error = Some(loom_external_error("Loom decode completion failed", error));
            }
        }
        drained += 1;
    }
}

/// Returns a snapshot of provider-hit and completion evidence.
pub fn loom_paged_decode_stats() -> Result<LoomPagedDecodeStats> {
    Ok(lock_registry()?.stats)
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
            return Err(loom_error(format!(
                "Loom paged decode requires contiguous {name}"
            )));
        }
        if tensor.device().location() != query.device().location() {
            return Err(loom_error(format!(
                "Loom paged decode requires {name} on the query device"
            )));
        }
    }
    let (batch_size, _, head_dim) = query.dims3()?;
    let key_shape = key_cache.dims4()?;
    if value_cache.dims4()? != key_shape {
        return Err(loom_error("Loom paged decode cache shapes do not match"));
    }
    if head_dim != HEAD_DIM || key_shape.2 != PAGE_SIZE || key_shape.3 != HEAD_DIM {
        return Err(loom_error(
            "Loom paged decode requires BF16/D128/page16/HND cache tensors",
        ));
    }
    if page_indptr.dims1()? != batch_size + 1 || last_page_len.dims1()? != batch_size {
        return Err(loom_error(
            "Loom paged decode received invalid CSR metadata shapes",
        ));
    }
    let physical_page_indices = page_indices.dims1()?;
    if logical_page_count < batch_size || logical_page_count > physical_page_indices {
        return Err(loom_error(format!(
            "logical page count {logical_page_count} is outside {batch_size}..={physical_page_indices}"
        )));
    }
    Ok(())
}

fn require_dtype(tensor: &Tensor, expected: DType, name: &str) -> Result<()> {
    if tensor.dtype() == expected {
        Ok(())
    } else {
        Err(loom_error(format!(
            "Loom paged decode requires {name} dtype {expected:?}, got {:?}",
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
        return Err(loom_error(format!(
            "Loom paged decode requires CUDA {name} storage"
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
        return Err(loom_error(format!(
            "Loom paged decode requires {name} on one CUDA context and stream"
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

fn loom_stream(stream: &CudaStream) -> loom_cuda_core::sys::CUstream {
    stream.cu_stream().cast()
}

fn tensor_lease(tensor: &Tensor) -> Arc<dyn Any + Send + Sync> {
    Arc::new(tensor.clone())
}

fn offset_pointer<T>(base: u64, offset: usize) -> Result<u64> {
    let bytes = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| loom_error("CUDA tensor offset overflow"))?;
    base.checked_add(bytes as u64)
        .ok_or_else(|| loom_error("CUDA device pointer overflow"))
}

fn registry() -> &'static Mutex<RuntimeRegistry> {
    RUNTIMES.get_or_init(|| Mutex::new(RuntimeRegistry::default()))
}

fn lock_registry() -> Result<MutexGuard<'static, RuntimeRegistry>> {
    registry()
        .lock()
        .map_err(|_| loom_error("Loom runtime registry is poisoned"))
}

fn loom_external_error(context: &str, error: impl std::fmt::Display) -> Error {
    loom_error(format!("{context}: {error}"))
}

fn loom_error(message: impl Into<String>) -> Error {
    Error::Msg(message.into()).bt()
}
