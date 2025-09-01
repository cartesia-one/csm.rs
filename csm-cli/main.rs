use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use csm_rs::{
    model::{Config as CsmModelConfig, Csm, CsmModelWrapper, Flavor},
    Generator,
};
use futures_util::StreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use moshi::wav;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "Hello there, this is a test")]
    text: String,
    #[arg(long, default_value_t = 0)]
    speaker_id: u32,
    #[arg(long, default_value = "csm_output.wav")]
    output: String,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 100)]
    top_k: usize,
    #[arg(long, default_value_t = 30000.0)]
    max_audio_len_ms: f32,
    #[arg(long, default_value_t = false)]
    quantized: bool,
    #[arg(long)]
    quantized_weights: Option<PathBuf>,
}

async fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    log::info!("Using device: {:?} with dtype: {:?}", device, dtype);
    log::info!("Loading models and tokenizers...");

    let api = Api::new()?;

    let tokenizer_repo =
        api.repo(Repo::new("unsloth/Llama-3.2-1B".to_string(), RepoType::Model));
    let tokenizer_path = tokenizer_repo.get("tokenizer.json").await?;
    let text_tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

    let config = CsmModelConfig {
        backbone_flavor: Flavor::Llama1B,
        decoder_flavor: Flavor::Llama100M,
        text_vocab_size: 128256,
        audio_vocab_size: 2051,
        audio_num_codebooks: 32,
    };

    let start_model_load = std::time::Instant::now();
    let mut model = if args.quantized {
        log::info!("Loading QUANTIZED model.");
        let qw_path = args
            .quantized_weights
            .ok_or_else(|| anyhow!("--quantized-weights must be specified for quantized model"))?;
        CsmModelWrapper::new_quantized(&config, &qw_path, &device)?
    } else {
        log::info!("Loading FULL PRECISION model.");
        let model_repo = api.repo(Repo::new("sesame/csm-1b".to_string(), RepoType::Model));
        let model_path = model_repo.get("model.safetensors").await?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
        CsmModelWrapper::new_full(&config, vb)?
    };
    log::info!(
        "Loaded CSM model in {:.2}s.",
        start_model_load.elapsed().as_secs_f64()
    );

    model.clear_kv_cache();
    let start_generator_new = std::time::Instant::now();
    let mut generator = Generator::new(model, text_tokenizer).await?;
    log::info!(
        "Created generator (loaded audio tokenizer) in {:.2}s.",
        start_generator_new.elapsed().as_secs_f64()
    );
    let sample_rate = generator.audio_tokenizer.config().sample_rate as u32;

    log::info!("Generating audio for text: '{}'", args.text);
    let mut audio_stream = generator.generate_stream(
        &args.text,
        args.speaker_id,
        args.max_audio_len_ms,
        args.temperature,
        args.top_k,
    );

    let mut all_audio_chunks = vec![];
    while let Some(chunk_result) = audio_stream.next().await {
        match chunk_result {
            Ok(chunk) => all_audio_chunks.push(chunk),
            Err(e) => {
                log::error!("Error during generation: {}", e);
                bail!("Stream generation failed");
            }
        }
    }
    let final_audio = if all_audio_chunks.is_empty() {
        Tensor::zeros((0,), DType::F32, &Device::Cpu)?
    } else {
        Tensor::cat(&all_audio_chunks, 0)?
    };

    if final_audio.dim(0)? > 0 {
        let audio_f32 = final_audio.to_vec1::<f32>()?;
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        let mut file = std::fs::File::create(&args.output)?;
        wav::write_pcm_as_wav(&mut file, &audio_i16, sample_rate)?;
        log::info!("Successfully saved audio to {}", args.output);
    } else {
        log::warn!("No audio was generated.");
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {:?}", e);
    }
}