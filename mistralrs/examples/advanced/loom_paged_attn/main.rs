//! Run one local Qwen2.5 model through the Loom paged-decode provider.
//!
//! ```text
//! MISTRALRS_LOOM_INFER=1 LOOM_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct \
//! cargo +nightly-2026-04-03 oxide run --bin loom_paged_attn \
//!   --features loom-infer --arch sm_90
//! ```
//!
//! Set `LOOM_BASELINE=1` instead of `MISTRALRS_LOOM_INFER=1` to run the
//! same deterministic request through the default Mistral.rs provider.

use anyhow::{bail, Result};
use candle_core::Device;
use mistralrs::{
    DeviceMapSetting, MemoryGpuConfig, ModelDType, PagedAttentionMetaBuilder, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};

const DEFAULT_MODEL_PATH: &str = "/workspace/models/qwen2.5-1.5b-instruct";
const MAX_OUTPUT_TOKENS: usize = 8;
const TOP_LOGPROBS: usize = 5;

#[tokio::main]
async fn main() -> Result<()> {
    let loom_enabled = std::env::var("MISTRALRS_LOOM_INFER").as_deref() == Ok("1");
    let baseline_enabled = std::env::var("LOOM_BASELINE").as_deref() == Ok("1");
    let validation_mode = match (loom_enabled, baseline_enabled) {
        (true, false) => "loom",
        (false, true) => "baseline",
        _ => bail!("set exactly one of MISTRALRS_LOOM_INFER=1 or LOOM_BASELINE=1"),
    };

    let model_path =
        std::env::var("LOOM_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    let device = Device::new_cuda_with_stream(0)?;
    let model = TextModelBuilder::new(model_path)
        .with_dtype(ModelDType::BF16)
        .with_device(device)
        .with_device_mapping(DeviceMapSetting::dummy())
        .with_max_num_seqs(1)
        .with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(16)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(4096))
                .build()?,
        )
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Reply with one short sentence that explains what a CUDA kernel does.",
    );
    let response = model
        .send_chat_request(
            RequestBuilder::from(messages)
                .set_sampler_temperature(0.0)
                .set_sampler_max_len(MAX_OUTPUT_TOKENS)
                .set_sampler_topn_logprobs(TOP_LOGPROBS)
                .return_logprobs(true),
        )
        .await?;
    let choice = response
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("model returned no choices"))?;
    let stats = if loom_enabled {
        let stats = model
            .loom_paged_decode_stats()
            .await?
            .ok_or_else(|| anyhow::anyhow!("model has no Loom paged-decode runtime"))?;
        if stats.submitted() == 0
            || stats.completed() != stats.submitted()
            || stats.failed() != 0
            || !stats.adapter_zero_copy()
            || stats.external_regions() != 9
            || stats.adapter_device_to_device_copies() != 0
        {
            bail!("Loom provider-hit validation failed: {stats:?}");
        }
        Some(stats)
    } else {
        None
    };

    println!("validation_mode={validation_mode}");
    println!(
        "model_response={}",
        choice.message.content.as_deref().unwrap_or("")
    );
    println!(
        "model_usage=prompt:{} completion:{}",
        response.usage.prompt_tokens, response.usage.completion_tokens
    );
    println!(
        "token_logprobs={}",
        serde_json::to_string(&choice.logprobs)?
    );
    if let Some(stats) = stats {
        println!("loom_provider={stats:?}");
    }
    Ok(())
}
