#![deny(unsafe_code)]

use candle_core::{Device, Tensor};
use half::bf16;
use mistralrs_paged_attn::{OxidePagedDecodeRuntime, OxidePagedDecodeStats};
use oxide_infer::{
    paged_batch_decode_bf16_reference, Bf16PagedBatchDecodeSpec, ContractError, PagedKvLayout,
};
use oxide_infer_cuda::interop::{
    EngineAlgorithm, EngineCommandFailure, EngineMetadataValidation, EngineOperator,
};
use std::error::Error;

const BATCH_SIZE: usize = 2;
const MAX_NUM_PAGES: usize = 4;
const QUERY_HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const PAGE_SIZE: usize = 16;
const LOGICAL_PAGE_COUNT: usize = 4;
const INVALID_PAGE_POSITION: usize = 1;
const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda_with_stream(0)?;
    let runtime = OxidePagedDecodeRuntime::new();
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
    let page_indptr_host = [0_i32, 2, 4];
    let valid_page_indices_host = [0_i32, 3, 1, 2];
    let invalid_page_indices_host = [0_i32, MAX_NUM_PAGES as i32, 1, 2];
    let last_page_len_host = [9_i32, 16];
    let mut expected_output = vec![bf16::ZERO; spec.output_numel()];
    let mut expected_lse = vec![0.0_f32; spec.lse_numel()];
    paged_batch_decode_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &page_indptr_host,
        &valid_page_indices_host,
        &last_page_len_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Tensor::from_vec(query_host, (BATCH_SIZE, QUERY_HEADS, HEAD_DIM), &device)?;
    let key_cache = Tensor::from_vec(
        key_host,
        (MAX_NUM_PAGES, KV_HEADS, PAGE_SIZE, HEAD_DIM),
        &device,
    )?;
    let value_cache = Tensor::from_vec(
        value_host,
        (MAX_NUM_PAGES, KV_HEADS, PAGE_SIZE, HEAD_DIM),
        &device,
    )?;
    let page_indptr = Tensor::from_vec(page_indptr_host.to_vec(), BATCH_SIZE + 1, &device)?;
    let valid_page_indices = Tensor::from_vec(
        valid_page_indices_host.to_vec(),
        LOGICAL_PAGE_COUNT,
        &device,
    )?;
    let invalid_page_indices = Tensor::from_vec(
        invalid_page_indices_host.to_vec(),
        LOGICAL_PAGE_COUNT,
        &device,
    )?;
    let last_page_len = Tensor::from_vec(last_page_len_host.to_vec(), BATCH_SIZE, &device)?;
    let baseline = runtime.stats()?;

    let valid_before = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    let _invalid = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &invalid_page_indices,
        &last_page_len,
    )?;
    let valid_after = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    let valid_tail = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    assert_stats_delta(runtime.stats()?, baseline, 4, 0, 0)?;

    let error = runtime
        .drain()
        .expect_err("invalid CSR metadata must fail the adapter drain");
    let failed_trace = error
        .trace()
        .ok_or("adapter drain error did not include an execution trace")?;
    if error.drained() != 4
        || error.failed_position() != Some(2)
        || !matches!(
            error.cause(),
            Some(EngineCommandFailure::DeviceRejected(
                ContractError::PageIndexOutOfRange {
                    position: INVALID_PAGE_POSITION,
                    index,
                    max_num_pages: MAX_NUM_PAGES,
                }
            )) if *index == MAX_NUM_PAGES as i32
        )
        || failed_trace.operator() != EngineOperator::Bf16PagedBatchDecode
        || failed_trace.paged_kv_layout() != Some(PagedKvLayout::Hnd)
        || failed_trace.algorithm() != EngineAlgorithm::PagedBatchDecodeTokenParallel8
        || failed_trace.memory().external_regions() != 9
        || failed_trace.adapter_device_to_device_copies() != 0
        || !failed_trace.is_adapter_zero_copy()
    {
        return Err(format!("adapter drain returned the wrong error: {error}").into());
    }
    let before_max_abs = compare_output(&valid_before, &expected_output, "valid before error")?;
    let after_max_abs = compare_output(&valid_after, &expected_output, "valid after error")?;
    let tail_max_abs = compare_output(&valid_tail, &expected_output, "valid FIFO tail")?;
    assert_stats_delta(runtime.stats()?, baseline, 4, 4, 1)?;

    let reused = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    assert_stats_delta(runtime.stats()?, baseline, 5, 4, 1)?;
    if runtime.drain()? != 1 {
        return Err("adapter reuse drain did not settle one completion".into());
    }
    let reuse_max_abs = compare_output(&reused, &expected_output, "valid reuse")?;
    let stats = runtime.stats()?;
    assert_stats_delta(stats, baseline, 5, 5, 1)?;
    if runtime.drain()? != 0 {
        return Err("adapter drain left a queued completion".into());
    }

    let concurrent_first = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    let concurrent_second = enqueue(
        &runtime,
        &query,
        &key_cache,
        &value_cache,
        &page_indptr,
        &valid_page_indices,
        &last_page_len,
    )?;
    assert_stats_delta(runtime.stats()?, baseline, 7, 5, 1)?;
    let mut concurrent_drains = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            runtime
                .drain()
                .expect("the first concurrent valid drain must succeed")
        });
        let second = scope.spawn(|| {
            runtime
                .drain()
                .expect("the second concurrent valid drain must succeed")
        });
        [
            first.join().expect("the first drainer must not panic"),
            second.join().expect("the second drainer must not panic"),
        ]
    });
    concurrent_drains.sort_unstable();
    if concurrent_drains != [0, 2] {
        return Err(
            format!("concurrent drainers split the FIFO: observed {concurrent_drains:?}").into(),
        );
    }
    let concurrent_first_max_abs = compare_output(
        &concurrent_first,
        &expected_output,
        "first concurrent drain",
    )?;
    let concurrent_second_max_abs = compare_output(
        &concurrent_second,
        &expected_output,
        "second concurrent drain",
    )?;
    assert_stats_delta(runtime.stats()?, baseline, 7, 7, 1)?;
    if runtime.drain()? != 0 {
        return Err("concurrent drainers left a queued completion".into());
    }

    println!(
        "gate=oxide_adapter status=pass sequence=valid,invalid,valid,valid,drain,valid,drain \
         submitted_delta=7 completed_delta=7 failed_delta=1 typed_page_error=true \
         fifo_failed_position=2 same_runtime_reuse=true layout=HND gqa_group=6 \
         algorithm=PagedBatchDecodeTokenParallel8 adapter_zero_copy=true \
         adapter_d2d_copies=0 concurrent_drains=0,2 valid_before_max_abs={before_max_abs:.9e} \
         valid_after_max_abs={after_max_abs:.9e} valid_tail_max_abs={tail_max_abs:.9e} \
         reuse_max_abs={reuse_max_abs:.9e} \
         concurrent_first_max_abs={concurrent_first_max_abs:.9e} \
         concurrent_second_max_abs={concurrent_second_max_abs:.9e}"
    );
    Ok(())
}

#[allow(unsafe_code)]
fn enqueue(
    runtime: &OxidePagedDecodeRuntime,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    page_indptr: &Tensor,
    page_indices: &Tensor,
    last_page_len: &Tensor,
) -> candle_core::Result<Tensor> {
    // SAFETY: the gate uses one thread, one device, and one ordinary stream.
    unsafe {
        runtime.enqueue_paged_decode(
            query,
            key_cache,
            value_cache,
            page_indptr,
            page_indices,
            LOGICAL_PAGE_COUNT,
            last_page_len,
        )
    }
}

fn compare_output(output: &Tensor, expected: &[bf16], name: &str) -> Result<f32, Box<dyn Error>> {
    let actual = output.flatten_all()?.to_vec1::<bf16>()?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{name} output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    let mut max_abs = 0.0_f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!("{name} output contains a non-finite value at {index}").into());
        }
        max_abs = max_abs.max((actual - expected).abs());
    }
    if max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "{name} output max abs {max_abs:.9e} exceeds {OUTPUT_MAX_ABS_LIMIT:.9e}"
        )
        .into());
    }
    Ok(max_abs)
}

fn assert_stats_delta(
    actual: OxidePagedDecodeStats,
    baseline: OxidePagedDecodeStats,
    submitted: u64,
    completed: u64,
    failed: u64,
) -> Result<(), Box<dyn Error>> {
    let observed = (
        actual.submitted().checked_sub(baseline.submitted()),
        actual.completed().checked_sub(baseline.completed()),
        actual.failed().checked_sub(baseline.failed()),
    );
    if observed != (Some(submitted), Some(completed), Some(failed))
        || actual.last_operator() != Some(EngineOperator::Bf16PagedBatchDecode)
        || actual.last_layout() != Some(PagedKvLayout::Hnd)
        || actual.last_algorithm() != Some(EngineAlgorithm::PagedBatchDecodeTokenParallel8)
        || actual.last_metadata_validation() != Some(EngineMetadataValidation::DeviceChecked)
        || !actual.adapter_zero_copy()
        || actual.external_regions() != 9
        || actual.adapter_device_to_device_copies() != 0
    {
        return Err(format!(
            "adapter stats mismatch: expected deltas {submitted}/{completed}/{failed}, got {actual:?}"
        )
        .into());
    }
    Ok(())
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
