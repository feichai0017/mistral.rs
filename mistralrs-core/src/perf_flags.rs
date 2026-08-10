use std::sync::OnceLock;

const CUDA_GRAPHS_ENV: &str = "MISTRALRS_CUDA_GRAPHS";
const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";
const LOOM_INFER_ENV: &str = "MISTRALRS_LOOM_INFER";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
static LOOM_INFER_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true) && !loom_infer_enabled())
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}

pub(crate) fn loom_infer_enabled() -> bool {
    cfg!(all(feature = "loom-infer", target_family = "unix"))
        && *LOOM_INFER_ENABLED.get_or_init(|| env_flag(LOOM_INFER_ENV, false))
}
