use anyhow::{anyhow, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use futures_util::Stream;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use moshi::mimi;
use rand::Rng;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::pin::Pin;
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod csm_full_impl;
use crate::csm_full_impl::WeightMapFlavor;
mod csm_quantized_impl;

mod model;
use crate::model::{Csm, CsmModelWrapper};

pub struct Generator {
    pub model: CsmModelWrapper,
    pub audio_tokenizer: mimi::Mimi,
    text_tokenizer: Tokenizer,
    device: Device,
    max_seq_len: usize,
}

pub struct GeneratorArgs {
    pub quantized: bool,
    pub quantized_weights: Option<PathBuf>,
    pub model_id: Option<String>,
    pub tokenizer_id: Option<String>,
    pub index_file: Option<String>,
    pub device: Device,
}

impl Generator {
    pub async fn new(args: GeneratorArgs) -> Result<Self> {
        let device = args.device;
        let csm_dtype = match &device {
            Device::Cuda(_) => {
                if device.supports_bf16() {
                    log::info!("CUDA device supports bf16, using DType::BF16.");
                    DType::BF16
                } else {
                    log::info!("CUDA device does not support bf16, using DType::F16.");
                    DType::F16
                }
            }
            _ => DType::F32,
        };
        log::info!("Using device: {:?} for generation", device);

        let api = Api::new()?;

        log::info!("Loading text tokenizer...");
        let tokenizer_id = args
            .tokenizer_id
            .unwrap_or_else(|| "unsloth/Llama-3.2-1B".to_string());
        let tokenizer_repo = api.repo(Repo::new(tokenizer_id, RepoType::Model));
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").await?;
        let text_tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

        let config = model::Config {
            backbone_flavor: model::Flavor::Llama1B,
            decoder_flavor: model::Flavor::Llama100M,
            text_vocab_size: 128256,
            audio_vocab_size: 2051,
            audio_num_codebooks: 32,
        };

        log::info!("Loading CSM model...");
        let start_model_load = std::time::Instant::now();
        let mut model = if args.quantized {
            let qw_path = args.quantized_weights.ok_or_else(|| {
                anyhow!("--quantized-weights must be specified for quantized model")
            })?;
            log::info!("Loading QUANTIZED model from {:?}", qw_path);
            CsmModelWrapper::new_quantized(&config, &qw_path, &device)?
        } else {
            log::info!("Loading FULL PRECISION model with dtype {:?}", csm_dtype);
            let model_id = args.model_id.unwrap_or_else(|| "sesame/csm-1b".to_string());
            let model_repo = api.repo(Repo::new(model_id, RepoType::Model));

            let mut safetensors_paths: Vec<PathBuf> = Vec::new();
            let index_file: String = args.index_file.clone().unwrap_or_else(|| "model.safetensors.index.json".to_string());
            match model_repo.get(&index_file).await {
                Ok(index_path) => {
                    log::info!("Found {:?}, loading sharded weights.", args.index_file);
                    let index_content = fs::read_to_string(&index_path)?;
                    let json: serde_json::Value = serde_json::from_str(&index_content)?;
                    let weight_map = json["weight_map"]
                        .as_object()
                        .ok_or_else(|| anyhow!("Invalid 'weight_map' in index.json"))?;
                    let unique_files: HashSet<&str> =
                        weight_map.values().filter_map(|v| v.as_str()).collect();
                    for filename in unique_files {
                        let safetensor_file = model_repo.get(filename).await?;
                        safetensors_paths.push(safetensor_file);
                    }
                }
                Err(_) => {
                    log::info!("Could not find index file '{:?}'. Assuming single-file model and trying 'model.safetensors'.", args.index_file);
                    let model_path = model_repo.get("model.safetensors").await?;
                    safetensors_paths.push(model_path);
                }
            }

            log::info!("Loading weights from: {:?}", safetensors_paths);
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensors_paths, csm_dtype, &device)?
            };

            let flavor = {
                if vb.contains_tensor("embed_text_tokens.weight"){
                    log::info!("Detected 'Transformers' weight naming convention.");
                    WeightMapFlavor::Transformers
                } else {
                    log::info!("Assuming 'Sesame' weight naming convention.");
                    WeightMapFlavor::Sesame
                }
            };

            CsmModelWrapper::new_full(&config, vb, flavor)?
        };
        log::info!(
            "Loaded CSM model in {:.2}s.",
            start_model_load.elapsed().as_secs_f64()
        );

        model.clear_kv_cache();

        let mimi_dtype = match &model {
            CsmModelWrapper::Full(m) => m.backbone.dtype,
            CsmModelWrapper::Quantized(_) => DType::F32,
        };

        log::info!(
            "Loading mimi audio tokenizer weights with dtype {:?}...",
            mimi_dtype
        );
        let start_mimi_load = std::time::Instant::now();
        let repo = api.repo(Repo::new(
            "kyutai/moshiko-pytorch-bf16".to_string(),
            RepoType::Model,
        ));
        let mimi_weights_path = repo
            .get("tokenizer-e351c8d8-checkpoint125.safetensors")
            .await?;

        let num_codebooks_for_decode = model.config().audio_num_codebooks / 2;
        let mimi_cfg = mimi::Config::v0_1(Some(num_codebooks_for_decode));

        let vb_mimi =
            unsafe { VarBuilder::from_mmaped_safetensors(&[mimi_weights_path], mimi_dtype, &device)? };
        let audio_tokenizer = mimi::Mimi::new(mimi_cfg, vb_mimi)?;
        log::info!(
            "Loaded mimi audio tokenizer in {:.2}s",
            start_mimi_load.elapsed().as_secs_f64()
        );

        Ok(Self {
            max_seq_len: 2048,
            model,
            audio_tokenizer,
            text_tokenizer,
            device: device.clone(),
        })
    }

    fn tokenize_text(&self, text: &str, speaker_id: u32, template: Option<&str>) -> Result<(Tensor, Tensor)> {
        let formatted_text = if let Some(template) = template {
            template
                .replace("{speaker_id}", &speaker_id.to_string())
                .replace("{text}", text)
        } else {
            format!("<|begin_of_text|>[{speaker_id}]{text}<|end_of_text|>")
        };
        let text_tokens_encoded = self
            .text_tokenizer
            .encode(formatted_text, true)
            .map_err(Error::msg)?;
        let ids: &[u32] = text_tokens_encoded.get_ids();
        Ok(self.model.text_tokens_and_mask(ids)?)
    }
    pub fn generate_stream<'a>(
        &'a mut self,
        text: &'a str,
        speaker_id: u32,
        max_audio_len_ms: f32,
        temperature: f64,
        top_k: usize,
        buffer_size: usize,
        tokenizer_template: Option<String>,
    ) -> Pin<Box<dyn Stream<Item = Result<Tensor>> + Send + 'a>> {
        use async_stream::stream;

        let stream = stream! {
            let mut lp = LogitsProcessor::from_sampling(
                rand::thread_rng().gen(),
                candle_transformers::generation::Sampling::TopK { k: top_k, temperature },
            );

            self.model.clear_kv_cache();

            let (prompt_tokens, prompt_mask) = match self.tokenize_text(text, speaker_id, tokenizer_template.as_deref()) {
                Ok(t) => t,
                Err(e) => { yield Err(e); return; }
            };

            let mut frame_buffer: VecDeque<Vec<u32>> = VecDeque::new();

            let max_gen_len = (max_audio_len_ms / 80.0) as usize;
            log::info!("Starting generation for up to {} frames...", max_gen_len);

            let mut current_tokens = prompt_tokens;
            let mut current_mask = prompt_mask;
            let mut current_pos = 0;

            for i in 0..max_gen_len {
                log::info!("generating frame {:?} (max: {:?})", i + 1, max_gen_len);
                let seq_len = current_tokens.dim(1)?;
                if seq_len > self.max_seq_len {
                    let start_pos = seq_len - self.max_seq_len;
                    current_tokens = current_tokens.narrow(1, start_pos, self.max_seq_len)?;
                    current_mask = current_mask.narrow(1, start_pos, self.max_seq_len)?;
                }

                let new_frame = match self.model.generate_frame(
                    &current_tokens,
                    &current_mask,
                    current_pos,
                    &mut lp,
                ) {
                    Ok(t) => t,
                    Err(e) => { yield Err(e.into()); return; }
                };

                if new_frame.iter().all(|&x| x == 0) {
                    log::info!("Model signaled end of generation at frame {}. Stopping.", i + 1);
                    break;
                }

                current_pos += current_tokens.dim(1)?;
                let (next_tokens, next_mask) = match self.model.audio_tokens_and_mask(new_frame.clone()) {
                    Ok(t) => t,
                    Err(e) => { yield Err(e.into()); return; }
                };

                current_tokens = next_tokens;
                current_mask = next_mask;

                frame_buffer.push_back(new_frame);

                if frame_buffer.len() >= buffer_size {
                        let frames_to_decode: Vec<Vec<u32>> = frame_buffer.drain(..buffer_size).collect();
                        let audio_chunk = match self.decode_frames(frames_to_decode) {
                            Ok(chunk) => chunk,
                            Err(e) => { yield Err(e); return; }
                        };
                        yield Ok(audio_chunk.to_device(&Device::Cpu)?);
                }
            }

            if !frame_buffer.is_empty() {
                let frames_to_decode: Vec<Vec<u32>> = frame_buffer.drain(..).collect();
                let audio_chunk = match self.decode_frames(frames_to_decode) {
                    Ok(chunk) => chunk,
                    Err(e) => { yield Err(e); return; }
                };
                log::info!("Generated final audio chunk of size {:?}", audio_chunk.shape());
                yield Ok(audio_chunk.to_device(&Device::Cpu)?);
            }
        };

        Box::pin(stream)
    }

    fn decode_frames(&mut self, frames: Vec<Vec<u32>>) -> Result<Tensor> {
        let decode_start_time = std::time::Instant::now();
        if frames.is_empty() {
            return Ok(Tensor::zeros((0,), DType::F32, &self.device)?);
        }
        let num_frames = frames.len();
        let num_codebooks_to_decode = self.model.config().audio_num_codebooks / 2;

        let mut flat_frames: Vec<u32> = Vec::with_capacity(num_frames * num_codebooks_to_decode);
        for frame in frames.iter() {
            flat_frames.extend_from_slice(&frame[..num_codebooks_to_decode]);
        }

        let frames_tensor = Tensor::from_vec(
            flat_frames,
            (num_frames, num_codebooks_to_decode),
            &self.device,
        )?
        .transpose(0, 1)?
        .unsqueeze(0)?;

        self.audio_tokenizer.reset_state();
        let audio_output = self
            .audio_tokenizer
            .decode(&frames_tensor)?
            .squeeze(0)?
            .squeeze(0)?;

        log::info!(
            "Decoded {} frames in {:.2}ms",
            num_frames,
            decode_start_time.elapsed().as_secs_f64() * 1000.0
        );

        Ok(audio_output)
    }
}