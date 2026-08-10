#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionBackendKind {
    Standard,
    FlashInfer,
    Loom,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerSpec {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

pub trait AttentionBackend {
    fn kind(&self) -> AttentionBackendKind;
    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool;
}
