use candle_core::{Device, Result, Tensor};
use candle_nn::{VarBuilder};
use candle_transformers::generation::LogitsProcessor;

use crate::csm_full_impl::FullModel;
use crate::csm_quantized_impl::QuantizedCsmModel;

#[derive(serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    #[serde(rename = "llama-1B")]
    Llama1B,
    #[serde(rename = "llama-100M")]
    Llama100M,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub audio_num_codebooks: usize,
    pub audio_vocab_size: usize,
    pub backbone_flavor: Flavor,
    pub decoder_flavor: Flavor,
    pub text_vocab_size: usize,
}

pub trait Csm {
    fn clear_kv_cache(&mut self);

    fn generate_frame(
        &mut self,
        tokens: &Tensor,
        tokens_mask: &Tensor,
        input_pos: usize,
        lp: &mut LogitsProcessor,
    ) -> Result<Vec<u32>>;

    fn audio_tokens_and_mask(&self, frame: Vec<u32>) -> Result<(Tensor, Tensor)>;

    fn text_tokens_and_mask(&self, ids: &[u32]) -> Result<(Tensor, Tensor)>;

    fn config(&self) -> &Config;

    fn device(&self) -> &Device;
}

pub enum CsmModelWrapper {
    Full(FullModel),
    Quantized(QuantizedCsmModel),
}

impl CsmModelWrapper {
    pub fn new_full(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(CsmModelWrapper::Full(FullModel::new(config, vb)?))
    }

    pub fn new_quantized<P: AsRef<std::path::Path>>(
        config: &Config,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        Ok(CsmModelWrapper::Quantized(QuantizedCsmModel::from_gguf(
            config, path, device,
        )?))
    }
}

impl Csm for CsmModelWrapper {
    fn clear_kv_cache(&mut self) {
        match self {
            CsmModelWrapper::Full(m) => m.clear_kv_cache(),
            CsmModelWrapper::Quantized(m) => m.clear_kv_cache(),
        }
    }

    fn generate_frame(
        &mut self,
        tokens: &Tensor,
        tokens_mask: &Tensor,
        input_pos: usize,
        lp: &mut LogitsProcessor,
    ) -> Result<Vec<u32>> {
        match self {
            CsmModelWrapper::Full(m) => m.generate_frame(tokens, tokens_mask, input_pos, lp),
            CsmModelWrapper::Quantized(m) => m.generate_frame(tokens, tokens_mask, input_pos, lp),
        }
    }

    fn audio_tokens_and_mask(&self, frame: Vec<u32>) -> Result<(Tensor, Tensor)> {
        match self {
            CsmModelWrapper::Full(m) => m.audio_tokens_and_mask(frame),
            CsmModelWrapper::Quantized(m) => m.audio_tokens_and_mask(frame),
        }
    }

    fn text_tokens_and_mask(&self, ids: &[u32]) -> Result<(Tensor, Tensor)> {
        match self {
            CsmModelWrapper::Full(m) => m.text_tokens_and_mask(ids),
            CsmModelWrapper::Quantized(m) => m.text_tokens_and_mask(ids),
        }
    }

    fn config(&self) -> &Config {
        match self {
            CsmModelWrapper::Full(m) => m.config(),
            CsmModelWrapper::Quantized(m) => m.config(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            CsmModelWrapper::Full(m) => m.device(),
            CsmModelWrapper::Quantized(m) => m.device(),
        }
    }
}