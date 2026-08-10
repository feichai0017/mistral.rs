use candle_core::{DType, Result};

use super::attention_backend::AttentionBackendKind;
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::attention::flash_backend_supports_sdpa;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{self, FlashInferDecodePlan, FlashInferDecodePlanInput};

#[allow(dead_code)]
pub(crate) struct PrefixPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub has_sinks: bool,
    pub has_custom_mask: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub has_softcap: bool,
    pub has_sliding_window: bool,
    pub query_layout_is_dense: bool,
    pub block_size: usize,
    pub attention_backend: AttentionBackendKind,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefixPrefillPlan {
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    FlashAttentionPaged,
    GatherSdpa,
}

impl PrefixPrefillPlan {
    pub fn choose(input: PrefixPrefillPlanInput) -> Self {
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        let _ = (
            input.device_is_cuda,
            input.dtype,
            input.has_sinks,
            input.has_custom_mask,
            input.causality_known,
            input.head_size,
            input.has_softcap,
            input.has_sliding_window,
            input.query_layout_is_dense,
            input.block_size,
            input.attention_backend,
        );

        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if input.device_is_cuda
            && matches!(input.dtype, DType::F16 | DType::BF16)
            && !input.has_sinks
            && !input.has_custom_mask
            && input.causality_known
            && input.query_layout_is_dense
            && paged_flash_attention_supports(
                input.head_size,
                input.block_size,
                input.has_softcap,
                input.has_sliding_window,
            )
            && matches!(input.attention_backend, AttentionBackendKind::FlashInfer)
        {
            return Self::FlashAttentionPaged;
        }

        Self::GatherSdpa
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn paged_flash_attention_supports(
    head_size: usize,
    block_size: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    flash_backend_supports_sdpa(head_size, has_softcap, has_sliding_window)
        && block_size.is_multiple_of(32)
}

#[allow(dead_code)]
pub(crate) struct DecodePlanInput {
    pub attention_backend: AttentionBackendKind,
    pub dtype: DType,
    pub query_len: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
    pub softmax_scale: f32,
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_sliding_window: bool,
    pub has_softcap: bool,
    pub has_custom_mask: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodePlan {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    FlashInfer(FlashInferDecodePlan),
    Loom,
    GatherSdpa,
    PagedAttention,
}

impl DecodePlan {
    pub(crate) fn requires_host_context_lengths(
        attention_backend: AttentionBackendKind,
        head_size: usize,
    ) -> bool {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            match attention_backend {
                AttentionBackendKind::FlashInfer | AttentionBackendKind::Standard => {
                    head_size > FlashInferDecodePlan::head_size_limit(attention_backend)
                }
                AttentionBackendKind::Loom => true,
            }
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        {
            let _ = head_size;
            matches!(
                attention_backend,
                AttentionBackendKind::FlashInfer | AttentionBackendKind::Loom
            )
        }
    }

    pub fn choose(input: DecodePlanInput) -> Result<Self> {
        if input.attention_backend == AttentionBackendKind::Loom {
            let expected_softmax_scale = 1.0 / 128.0_f32.sqrt();
            if input.dtype != DType::BF16
                || input.query_len != 1
                || input.query_heads == 0
                || input.kv_heads == 0
                || !input.query_heads.is_multiple_of(input.kv_heads)
                || input.head_size != 128
                || input.block_size != 16
                || input.softmax_scale.to_bits() != expected_softmax_scale.to_bits()
                || input.has_alibi
                || input.has_sinks
                || input.has_sliding_window
                || input.has_softcap
                || input.has_custom_mask
            {
                candle_core::bail!(
                    "Loom paged decode requires one query token, BF16, head_size=128, block_size=16, default softmax scale, nonzero query/kv heads with query_heads divisible by kv_heads, and no alibi, sinks, sliding window, softcap, or custom mask; got dtype={:?}, query_len={}, query_heads={}, kv_heads={}, head_size={}, block_size={}, softmax_scale={}, expected_softmax_scale={}, alibi={}, sinks={}, sliding_window={}, softcap={}, custom_mask={}",
                    input.dtype,
                    input.query_len,
                    input.query_heads,
                    input.kv_heads,
                    input.head_size,
                    input.block_size,
                    input.softmax_scale,
                    expected_softmax_scale,
                    input.has_alibi,
                    input.has_sinks,
                    input.has_sliding_window,
                    input.has_softcap,
                    input.has_custom_mask,
                );
            }
            return Ok(Self::Loom);
        }
        if input.has_custom_mask {
            return Ok(Self::GatherSdpa);
        }
        if Self::requires_host_context_lengths(input.attention_backend, input.head_size) {
            return Ok(Self::GatherSdpa);
        }
        match input.attention_backend {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            AttentionBackendKind::FlashInfer => {
                flashinfer::decode_plan(FlashInferDecodePlanInput {
                    head_size: input.head_size,
                    has_alibi: input.has_alibi,
                    has_sinks: input.has_sinks,
                })
                .map(Self::FlashInfer)
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            AttentionBackendKind::FlashInfer => Ok(Self::GatherSdpa),
            AttentionBackendKind::Standard if input.has_sliding_window => Ok(Self::GatherSdpa),
            AttentionBackendKind::Standard => Ok(Self::PagedAttention),
            AttentionBackendKind::Loom => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix_plan(
        head_size: usize,
        has_softcap: bool,
        has_sliding_window: bool,
    ) -> PrefixPrefillPlan {
        PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::F16,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size,
            has_softcap,
            has_sliding_window,
            query_layout_is_dense: true,
            block_size: 32,
            attention_backend: AttentionBackendKind::FlashInfer,
        })
    }

    #[test]
    fn paged_prefix_rejects_disabled_large_head_features() {
        assert!(matches!(
            prefix_plan(320, true, false),
            PrefixPrefillPlan::GatherSdpa
        ));
        assert!(matches!(
            prefix_plan(320, false, true),
            PrefixPrefillPlan::GatherSdpa
        ));
    }

    #[test]
    fn standard_sliding_decode_uses_exact_gather_path() {
        let plan = DecodePlan::choose(DecodePlanInput {
            attention_backend: AttentionBackendKind::Standard,
            dtype: DType::BF16,
            query_len: 1,
            query_heads: 8,
            kv_heads: 2,
            head_size: 128,
            block_size: 16,
            softmax_scale: 1.0 / (128f32).sqrt(),
            has_alibi: false,
            has_sinks: false,
            has_sliding_window: true,
            has_softcap: false,
            has_custom_mask: false,
        })
        .unwrap();

        assert!(matches!(plan, DecodePlan::GatherSdpa));
    }

    #[test]
    fn standard_full_decode_keeps_paged_kernel() {
        let plan = DecodePlan::choose(DecodePlanInput {
            attention_backend: AttentionBackendKind::Standard,
            dtype: DType::BF16,
            query_len: 1,
            query_heads: 8,
            kv_heads: 2,
            head_size: 128,
            block_size: 16,
            softmax_scale: 1.0 / (128f32).sqrt(),
            has_alibi: false,
            has_sinks: false,
            has_sliding_window: false,
            has_softcap: false,
            has_custom_mask: false,
        })
        .unwrap();

        assert!(matches!(plan, DecodePlan::PagedAttention));
    }

    fn loom_decode_input() -> DecodePlanInput {
        DecodePlanInput {
            attention_backend: AttentionBackendKind::Loom,
            dtype: DType::BF16,
            query_len: 1,
            query_heads: 12,
            kv_heads: 2,
            head_size: 128,
            block_size: 16,
            softmax_scale: 1.0 / (128f32).sqrt(),
            has_alibi: false,
            has_sinks: false,
            has_sliding_window: false,
            has_softcap: false,
            has_custom_mask: false,
        }
    }

    #[test]
    fn loom_decode_accepts_only_the_admitted_contract() {
        assert!(matches!(
            DecodePlan::choose(loom_decode_input()).unwrap(),
            DecodePlan::Loom
        ));

        let unsupported = [
            {
                let mut input = loom_decode_input();
                input.dtype = DType::F16;
                input
            },
            {
                let mut input = loom_decode_input();
                input.query_len = 2;
                input
            },
            {
                let mut input = loom_decode_input();
                input.query_heads = 0;
                input
            },
            {
                let mut input = loom_decode_input();
                input.kv_heads = 0;
                input
            },
            {
                let mut input = loom_decode_input();
                input.query_heads = 13;
                input
            },
            {
                let mut input = loom_decode_input();
                input.head_size = 64;
                input
            },
            {
                let mut input = loom_decode_input();
                input.block_size = 32;
                input
            },
            {
                let mut input = loom_decode_input();
                input.softmax_scale = 0.125;
                input
            },
            {
                let mut input = loom_decode_input();
                input.has_alibi = true;
                input
            },
            {
                let mut input = loom_decode_input();
                input.has_sinks = true;
                input
            },
            {
                let mut input = loom_decode_input();
                input.has_sliding_window = true;
                input
            },
            {
                let mut input = loom_decode_input();
                input.has_softcap = true;
                input
            },
            {
                let mut input = loom_decode_input();
                input.has_custom_mask = true;
                input
            },
        ];
        for input in unsupported {
            assert!(DecodePlan::choose(input).is_err());
        }
    }
}
