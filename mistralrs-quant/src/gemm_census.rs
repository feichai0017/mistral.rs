use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, DeviceLocation, Result, Tensor};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusObservedPath {
    MistralCudaGemv,
    CandleCudaFlattenedMatmul,
    CandleMatmul,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusDeviceKind {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusDType {
    Bf16,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusActivationLayout {
    RowMajorContiguous,
    LastDimensionContiguous,
    Strided,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmCensusWeightLayout {
    RowMajorContiguousNk,
    LastDimensionContiguousNk,
    StridedNk,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GemmCensusEntry {
    pub phase: GemmCensusPhase,
    pub site: String,
    pub observed_path: GemmCensusObservedPath,
    pub device_kind: GemmCensusDeviceKind,
    pub device_ordinal: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_shape: Vec<usize>,
    pub a_stride: Vec<usize>,
    pub a_offset_elements: usize,
    pub weight_shape: Vec<usize>,
    pub weight_stride: Vec<usize>,
    pub weight_offset_elements: usize,
    pub a_dtype: GemmCensusDType,
    pub weight_dtype: GemmCensusDType,
    pub a_layout: GemmCensusActivationLayout,
    pub weight_layout: GemmCensusWeightLayout,
    pub transpose_a: bool,
    pub transpose_weight: bool,
    pub post_ops: Vec<String>,
    pub host_calls: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GemmCensusMetadata {
    phase: GemmCensusPhase,
    site: String,
    device_kind: GemmCensusDeviceKind,
    device_ordinal: usize,
    m: usize,
    n: usize,
    k: usize,
    a_shape: Vec<usize>,
    a_stride: Vec<usize>,
    a_offset_elements: usize,
    weight_shape: Vec<usize>,
    weight_stride: Vec<usize>,
    weight_offset_elements: usize,
    a_dtype: GemmCensusDType,
    weight_dtype: GemmCensusDType,
    a_layout: GemmCensusActivationLayout,
    weight_layout: GemmCensusWeightLayout,
    transpose_a: bool,
    transpose_weight: bool,
    post_ops: Vec<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GemmCensusEntryKey {
    metadata: GemmCensusMetadata,
    observed_path: GemmCensusObservedPath,
}

impl GemmCensusEntryKey {
    fn into_entry(self, host_calls: u64) -> GemmCensusEntry {
        let metadata = self.metadata;
        GemmCensusEntry {
            phase: metadata.phase,
            site: metadata.site,
            observed_path: self.observed_path,
            device_kind: metadata.device_kind,
            device_ordinal: metadata.device_ordinal,
            m: metadata.m,
            n: metadata.n,
            k: metadata.k,
            a_shape: metadata.a_shape,
            a_stride: metadata.a_stride,
            a_offset_elements: metadata.a_offset_elements,
            weight_shape: metadata.weight_shape,
            weight_stride: metadata.weight_stride,
            weight_offset_elements: metadata.weight_offset_elements,
            a_dtype: metadata.a_dtype,
            weight_dtype: metadata.weight_dtype,
            a_layout: metadata.a_layout,
            weight_layout: metadata.weight_layout,
            transpose_a: metadata.transpose_a,
            transpose_weight: metadata.transpose_weight,
            post_ops: metadata.post_ops,
            host_calls,
        }
    }
}

pub(crate) struct PendingGemmCensus {
    metadata: GemmCensusMetadata,
    ordinal: usize,
    run_id: u64,
}

#[derive(Debug)]
enum GemmCensusScopeKind {
    Qwen2Layer { layer_idx: usize },
    Qwen2LmHead,
}

impl GemmCensusScopeKind {
    fn expected_dispatches(&self) -> usize {
        match self {
            Self::Qwen2Layer { .. } => 6,
            Self::Qwen2LmHead => 1,
        }
    }

    fn site(&self, ordinal: usize) -> Result<String> {
        match self {
            Self::Qwen2Layer { layer_idx } => {
                let suffix = match ordinal {
                    0 => "self_attn.q_proj",
                    1 => "self_attn.k_proj",
                    2 => "self_attn.v_proj",
                    3 => "self_attn.o_proj",
                    4 => "mlp.merged_gate_up",
                    5 => "mlp.down_proj",
                    _ => {
                        candle_core::bail!(
                            "Qwen2 layer {layer_idx} recorded more than six dense linear dispatches"
                        )
                    }
                };
                Ok(format!("model.layers.{layer_idx}.{suffix}"))
            }
            Self::Qwen2LmHead if ordinal == 0 => Ok("lm_head".to_string()),
            Self::Qwen2LmHead => {
                candle_core::bail!("Qwen2 lm_head recorded more than one dense linear dispatch")
            }
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Qwen2Layer { layer_idx } => format!("Qwen2 layer {layer_idx}"),
            Self::Qwen2LmHead => "Qwen2 lm_head".to_string(),
        }
    }
}

#[derive(Debug)]
struct GemmCensusScope {
    kind: GemmCensusScopeKind,
    phase: GemmCensusPhase,
    dispatches: usize,
    run_id: u64,
}

thread_local! {
    static ACTIVE_SCOPE: RefCell<Option<GemmCensusScope>> = const { RefCell::new(None) };
}

struct ActiveRun {
    id: u64,
    owner_alive: bool,
    failed: bool,
    active_scope: bool,
    entries: BTreeMap<GemmCensusEntryKey, u64>,
}

#[derive(Default)]
struct CollectorState {
    next_run_id: u64,
    active_run: Option<ActiveRun>,
}

static COLLECTOR: OnceLock<Mutex<CollectorState>> = OnceLock::new();
#[cfg(test)]
pub(crate) static GEMM_CENSUS_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn active_entry_count() -> Result<usize> {
    let collector = collector()
        .lock()
        .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
    Ok(collector
        .active_run
        .as_ref()
        .map_or(0, |run| run.entries.len()))
}

fn collector() -> &'static Mutex<CollectorState> {
    COLLECTOR.get_or_init(|| Mutex::new(CollectorState::default()))
}

fn fail_run(run_id: u64, scope_ended: bool) -> Result<()> {
    let mut collector = collector()
        .lock()
        .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
    let remove_abandoned =
        if let Some(run) = collector.active_run.as_mut().filter(|run| run.id == run_id) {
            run.failed = true;
            run.entries.clear();
            if scope_ended {
                run.active_scope = false;
            }
            !run.owner_alive && !run.active_scope
        } else {
            false
        };
    if remove_abandoned {
        collector.active_run = None;
    }
    Ok(())
}

pub struct GemmCensusRun {
    run_id: u64,
    finished: bool,
}

pub fn begin_gemm_census_run() -> Result<GemmCensusRun> {
    let mut collector = collector()
        .lock()
        .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
    if collector.active_run.is_some() {
        candle_core::bail!("GEMM census run is already active")
    }
    let run_id = collector
        .next_run_id
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("GEMM census run ID overflowed"))?;
    collector.next_run_id = run_id;
    collector.active_run = Some(ActiveRun {
        id: run_id,
        owner_alive: true,
        failed: false,
        active_scope: false,
        entries: BTreeMap::new(),
    });
    Ok(GemmCensusRun {
        run_id,
        finished: false,
    })
}

impl GemmCensusRun {
    pub fn finish(&mut self) -> Result<Vec<GemmCensusEntry>> {
        if self.finished {
            candle_core::bail!("GEMM census run is already finished")
        }
        let mut collector = collector()
            .lock()
            .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
        let run = collector.active_run.as_ref().ok_or_else(|| {
            candle_core::Error::msg("GEMM census run ended before its guard finished")
        })?;
        if run.id != self.run_id {
            candle_core::bail!("GEMM census run guard does not own the active run")
        }
        if run.active_scope {
            candle_core::bail!("GEMM census run cannot finish while a scope is active")
        }
        if run.failed {
            collector.active_run = None;
            self.finished = true;
            candle_core::bail!("GEMM census run failed")
        }
        let run = collector
            .active_run
            .take()
            .expect("active run checked above");
        self.finished = true;
        Ok(run
            .entries
            .into_iter()
            .map(|(key, host_calls)| key.into_entry(host_calls))
            .collect())
    }
}

impl Drop for GemmCensusRun {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Ok(mut collector) = collector().lock() {
            if let Some(run) = collector
                .active_run
                .as_mut()
                .filter(|run| run.id == self.run_id)
            {
                run.failed = true;
                run.entries.clear();
                run.owner_alive = false;
                if !run.active_scope {
                    collector.active_run = None;
                }
            }
        }
        self.finished = true;
    }
}

struct GemmCensusScopeGuard {
    run_id: u64,
    finished: bool,
}

impl GemmCensusScopeGuard {
    fn enter(kind: GemmCensusScopeKind, phase: GemmCensusPhase) -> Result<Self> {
        if let Some(run_id) =
            ACTIVE_SCOPE.with(|slot| slot.borrow().as_ref().map(|scope| scope.run_id))
        {
            fail_run(run_id, false)?;
            candle_core::bail!("GEMM census scope is already active on this thread")
        }

        let run_id = {
            let mut collector = collector()
                .lock()
                .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
            let run = collector.active_run.as_mut().ok_or_else(|| {
                candle_core::Error::msg("GEMM census scope requires an active run")
            })?;
            if run.failed {
                candle_core::bail!("GEMM census run has failed")
            }
            if run.active_scope {
                run.failed = true;
                run.entries.clear();
                candle_core::bail!("GEMM census permits one active scope")
            }
            run.active_scope = true;
            run.id
        };

        ACTIVE_SCOPE.with(|slot| {
            *slot.borrow_mut() = Some(GemmCensusScope {
                kind,
                phase,
                dispatches: 0,
                run_id,
            });
        });
        Ok(Self {
            run_id,
            finished: false,
        })
    }

    fn finish(mut self) -> Result<()> {
        let scope = ACTIVE_SCOPE
            .with(|slot| slot.borrow_mut().take())
            .ok_or_else(|| {
                candle_core::Error::msg("GEMM census scope ended without active state")
            })?;
        self.finished = true;
        if scope.run_id != self.run_id {
            fail_run(self.run_id, true)?;
            candle_core::bail!("GEMM census scope guard does not own the active scope")
        }
        let expected = scope.kind.expected_dispatches();
        if scope.dispatches != expected {
            fail_run(self.run_id, true)?;
            candle_core::bail!(
                "{} recorded {} dense linear dispatches, expected {expected}",
                scope.kind.label(),
                scope.dispatches
            )
        }
        let mut collector = collector()
            .lock()
            .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
        let run = collector.active_run.as_mut().ok_or_else(|| {
            candle_core::Error::msg("GEMM census run ended before its scope finished")
        })?;
        if run.id != self.run_id {
            candle_core::bail!("GEMM census scope belongs to an inactive run")
        }
        if !run.active_scope {
            run.failed = true;
            run.entries.clear();
            candle_core::bail!("GEMM census scope state changed before completion")
        }
        let (run_failed, owner_abandoned) = {
            run.active_scope = false;
            (run.failed, !run.owner_alive)
        };
        if owner_abandoned {
            collector.active_run = None;
        }
        if run_failed {
            candle_core::bail!("GEMM census run failed during scope execution")
        }
        Ok(())
    }
}

impl Drop for GemmCensusScopeGuard {
    fn drop(&mut self) {
        if !self.finished {
            ACTIVE_SCOPE.with(|slot| {
                slot.borrow_mut().take();
            });
            let _ = fail_run(self.run_id, true);
        }
    }
}

fn with_scope<T>(
    kind: GemmCensusScopeKind,
    phase: GemmCensusPhase,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let guard = GemmCensusScopeGuard::enter(kind, phase)?;
    match f() {
        Ok(value) => {
            guard.finish()?;
            Ok(value)
        }
        Err(error) => Err(error),
    }
}

pub fn with_qwen2_layer_scope<T>(
    phase: GemmCensusPhase,
    layer_idx: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_scope(GemmCensusScopeKind::Qwen2Layer { layer_idx }, phase, f)
}

pub fn with_qwen2_lm_head_scope<T>(
    phase: GemmCensusPhase,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_scope(GemmCensusScopeKind::Qwen2LmHead, phase, f)
}

fn dtype(dtype: DType, tensor_name: &str) -> Result<GemmCensusDType> {
    match dtype {
        DType::BF16 => Ok(GemmCensusDType::Bf16),
        DType::F16 => Ok(GemmCensusDType::F16),
        DType::F32 => Ok(GemmCensusDType::F32),
        _ => candle_core::bail!(
            "GEMM census supports BF16, F16, and F32, {tensor_name} uses {dtype:?}"
        ),
    }
}

fn device(location: DeviceLocation) -> (GemmCensusDeviceKind, usize) {
    match location {
        DeviceLocation::Cpu => (GemmCensusDeviceKind::Cpu, 0),
        DeviceLocation::Cuda { gpu_id } => (GemmCensusDeviceKind::Cuda, gpu_id),
        DeviceLocation::Metal { gpu_id } => (GemmCensusDeviceKind::Metal, gpu_id),
    }
}

fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1;
    for (&dimension, &actual_stride) in shape.iter().rev().zip(stride.iter().rev()) {
        if dimension != 1 && actual_stride != expected {
            return false;
        }
        expected *= dimension;
    }
    true
}

fn activation_layout(shape: &[usize], stride: &[usize]) -> GemmCensusActivationLayout {
    if is_contiguous(shape, stride) {
        GemmCensusActivationLayout::RowMajorContiguous
    } else if stride.last() == Some(&1) {
        GemmCensusActivationLayout::LastDimensionContiguous
    } else {
        GemmCensusActivationLayout::Strided
    }
}

fn weight_layout(shape: &[usize], stride: &[usize]) -> GemmCensusWeightLayout {
    if is_contiguous(shape, stride) {
        GemmCensusWeightLayout::RowMajorContiguousNk
    } else if stride.last() == Some(&1) {
        GemmCensusWeightLayout::LastDimensionContiguousNk
    } else {
        GemmCensusWeightLayout::StridedNk
    }
}

pub(crate) fn prepare_linear(
    activation: &Tensor,
    weight: &Tensor,
    has_bias: bool,
) -> Result<Option<PendingGemmCensus>> {
    let scope_event = ACTIVE_SCOPE.with(
        |slot| -> Result<Option<(GemmCensusPhase, String, usize, u64)>> {
            let slot = slot.borrow();
            let Some(scope) = slot.as_ref() else {
                return Ok(None);
            };
            Ok(Some((
                scope.phase,
                scope.kind.site(scope.dispatches)?,
                scope.dispatches,
                scope.run_id,
            )))
        },
    )?;
    let Some((phase, site, ordinal, run_id)) = scope_event else {
        let mut collector = collector()
            .lock()
            .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
        if let Some(run) = collector.active_run.as_mut() {
            run.failed = true;
            run.entries.clear();
            candle_core::bail!("GEMM census linear dispatch requires an active scope")
        }
        return Ok(None);
    };

    {
        let mut collector = collector()
            .lock()
            .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
        let run = collector.active_run.as_mut().ok_or_else(|| {
            candle_core::Error::msg("GEMM census run ended before linear dispatch")
        })?;
        if run.id != run_id || !run.active_scope || run.failed {
            if run.id == run_id {
                run.failed = true;
                run.entries.clear();
            }
            candle_core::bail!("GEMM census run is not ready for linear dispatch")
        }
    }

    let a_shape = activation.dims().to_vec();
    let a_stride = activation.layout().stride().to_vec();
    let (&k, leading_shape) = a_shape
        .split_last()
        .ok_or_else(|| candle_core::Error::msg("GEMM census activation must have rank one"))?;
    let m = leading_shape
        .iter()
        .try_fold(1usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
        .ok_or_else(|| candle_core::Error::msg("GEMM census activation size overflowed"))?;
    let (n, weight_k) = weight.dims2()?;
    if m == 0 || n == 0 || k == 0 {
        candle_core::bail!("GEMM census dimensions must be greater than zero")
    }
    if weight_k != k {
        candle_core::bail!("GEMM census activation K {k} differs from weight K {weight_k}")
    }
    if activation.device().location() != weight.device().location() {
        candle_core::bail!("GEMM census activation and weight use different devices")
    }
    for (name, left, right) in [("activation", m, k), ("weight", n, k), ("output", m, n)] {
        left.checked_mul(right).ok_or_else(|| {
            candle_core::Error::msg(format!("GEMM census {name} size overflowed"))
        })?;
    }
    let weight_shape = weight.dims().to_vec();
    let weight_stride = weight.layout().stride().to_vec();
    let (device_kind, device_ordinal) = device(activation.device().location());
    let metadata = GemmCensusMetadata {
        phase,
        site,
        device_kind,
        device_ordinal,
        m,
        n,
        k,
        a_layout: activation_layout(&a_shape, &a_stride),
        weight_layout: weight_layout(&weight_shape, &weight_stride),
        a_shape,
        a_stride,
        weight_shape,
        weight_stride,
        a_dtype: dtype(activation.dtype(), "activation")?,
        weight_dtype: dtype(weight.dtype(), "weight")?,
        transpose_a: false,
        transpose_weight: true,
        post_ops: has_bias.then(|| "bias".to_string()).into_iter().collect(),
        a_offset_elements: activation.layout().start_offset(),
        weight_offset_elements: weight.layout().start_offset(),
    };
    Ok(Some(PendingGemmCensus {
        metadata,
        ordinal,
        run_id,
    }))
}

impl PendingGemmCensus {
    pub(crate) fn commit(self, observed_path: GemmCensusObservedPath) -> Result<()> {
        let Self {
            metadata,
            ordinal,
            run_id,
        } = self;
        ACTIVE_SCOPE.with(|slot| -> Result<()> {
            let mut slot = slot.borrow_mut();
            let scope = slot.as_mut().ok_or_else(|| {
                candle_core::Error::msg("GEMM census dispatch completed without an active scope")
            })?;
            if scope.run_id != run_id
                || scope.dispatches != ordinal
                || scope.phase != metadata.phase
                || scope.kind.site(scope.dispatches)?.as_str() != metadata.site.as_str()
            {
                candle_core::bail!("GEMM census scope changed before dispatch completion")
            }
            scope.dispatches += 1;
            Ok(())
        })?;

        let key = GemmCensusEntryKey {
            metadata,
            observed_path,
        };
        let mut collector = collector()
            .lock()
            .map_err(|_| candle_core::Error::msg("GEMM census collector lock is poisoned"))?;
        let run = collector.active_run.as_mut().ok_or_else(|| {
            candle_core::Error::msg("GEMM census run ended before dispatch completion")
        })?;
        if run.id != run_id {
            candle_core::bail!("GEMM census dispatch belongs to an inactive run")
        }
        if run.failed || !run.active_scope {
            run.failed = true;
            run.entries.clear();
            candle_core::bail!("GEMM census run is not ready for dispatch completion")
        }
        let host_calls = run.entries.get(&key).copied().unwrap_or(0);
        let host_calls = host_calls
            .checked_add(1)
            .ok_or_else(|| candle_core::Error::msg("GEMM census host call count overflowed"))?;
        run.entries.insert(key, host_calls);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        active_entry_count, begin_gemm_census_run, prepare_linear, with_qwen2_layer_scope,
        with_qwen2_lm_head_scope, GemmCensusObservedPath, GemmCensusPhase, GEMM_CENSUS_TEST_LOCK,
    };

    #[test]
    fn qwen2_layer_scope_assigns_six_fixed_sites() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;
        let activation = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;

        with_qwen2_layer_scope(GemmCensusPhase::Decode, 3, || {
            for _ in 0..6 {
                prepare_linear(&activation, &weight, false)?
                    .expect("active Qwen2 scope must create a census ticket")
                    .commit(GemmCensusObservedPath::CandleMatmul)?;
            }
            Ok(())
        })?;

        let entries = run.finish()?;
        let sites = entries
            .iter()
            .map(|entry| entry.site.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            sites,
            [
                "model.layers.3.mlp.down_proj",
                "model.layers.3.mlp.merged_gate_up",
                "model.layers.3.self_attn.k_proj",
                "model.layers.3.self_attn.o_proj",
                "model.layers.3.self_attn.q_proj",
                "model.layers.3.self_attn.v_proj",
            ]
        );
        assert!(entries.iter().all(|entry| entry.host_calls == 1));
        Ok(())
    }

    #[test]
    fn qwen2_layer_scope_rejects_incomplete_dispatches() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;
        let error = with_qwen2_layer_scope(GemmCensusPhase::Prefill, 0, || Ok(()))
            .expect_err("empty Qwen2 layer scope must fail");
        assert!(error.to_string().contains("expected 6"));
        assert!(run.finish().is_err());
        Ok(())
    }

    #[test]
    fn pending_ticket_counts_only_after_successful_commit() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;
        let activation = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;

        with_qwen2_lm_head_scope(GemmCensusPhase::Decode, || {
            let pending = prepare_linear(&activation, &weight, false)?
                .expect("active Qwen2 scope must create a census ticket");
            assert_eq!(active_entry_count()?, 0);
            pending.commit(GemmCensusObservedPath::CandleMatmul)?;
            assert_eq!(active_entry_count()?, 1);
            Ok(())
        })?;

        assert_eq!(run.finish()?.len(), 1);
        Ok(())
    }

    #[test]
    fn qwen2_lm_head_scope_aggregates_successful_dispatches() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;
        let activation = Tensor::zeros((2, 4), DType::BF16, &Device::Cpu)?;
        let weight = Tensor::zeros((8, 4), DType::BF16, &Device::Cpu)?;

        for _ in 0..2 {
            with_qwen2_lm_head_scope(GemmCensusPhase::Prefill, || {
                let pending = prepare_linear(&activation, &weight, false)?
                    .expect("active Qwen2 scope must create a census ticket");
                pending.commit(GemmCensusObservedPath::CandleMatmul)
            })?;
        }

        let entries = run.finish()?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].site, "lm_head");
        assert_eq!(entries[0].m, 2);
        assert_eq!(entries[0].n, 8);
        assert_eq!(entries[0].k, 4);
        assert_eq!(entries[0].host_calls, 2);
        Ok(())
    }

    #[test]
    fn concurrent_begin_does_not_replace_the_active_run() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;

        assert!(begin_gemm_census_run().is_err());
        assert!(run.finish()?.is_empty());
        Ok(())
    }

    #[test]
    fn abandoned_run_clears_entries_and_releases_ownership() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let activation = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;

        {
            let _run = begin_gemm_census_run()?;
            with_qwen2_lm_head_scope(GemmCensusPhase::Decode, || {
                prepare_linear(&activation, &weight, false)?
                    .expect("active Qwen2 scope must create a census ticket")
                    .commit(GemmCensusObservedPath::CandleMatmul)
            })?;
            assert_eq!(active_entry_count()?, 1);
        }

        let mut replacement = begin_gemm_census_run()?;
        assert!(replacement.finish()?.is_empty());
        Ok(())
    }

    #[test]
    fn abandoned_run_retains_ownership_until_its_active_scope_exits() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let run = begin_gemm_census_run()?;
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let worker = std::thread::spawn(move || -> candle_core::Result<()> {
            with_qwen2_lm_head_scope(GemmCensusPhase::Decode, || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });

        entered_rx.recv().unwrap();
        drop(run);
        assert!(begin_gemm_census_run().is_err());

        release_tx.send(()).unwrap();
        assert!(worker.join().unwrap().is_err());

        let mut replacement = begin_gemm_census_run()?;
        assert!(replacement.finish()?.is_empty());
        Ok(())
    }

    #[test]
    fn partial_scope_failure_clears_the_run() -> candle_core::Result<()> {
        let _test = GEMM_CENSUS_TEST_LOCK.lock().unwrap();
        let mut run = begin_gemm_census_run()?;
        let activation = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;

        let error = with_qwen2_layer_scope(GemmCensusPhase::Prefill, 0, || {
            prepare_linear(&activation, &weight, false)?
                .expect("active Qwen2 scope must create a census ticket")
                .commit(GemmCensusObservedPath::CandleMatmul)
        })
        .expect_err("partial Qwen2 layer scope must fail");

        assert!(error.to_string().contains("expected 6"));
        assert_eq!(active_entry_count()?, 0);
        assert!(run.finish().is_err());
        let mut replacement = begin_gemm_census_run()?;
        assert!(replacement.finish()?.is_empty());
        Ok(())
    }
}
