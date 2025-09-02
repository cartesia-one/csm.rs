use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use csm_rs::{Generator, GeneratorArgs};
use futures_util::StreamExt;
use moshi::wav;
use std::path::PathBuf;

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
    #[arg(long, default_value_t = 20)]
    buffer_size: usize,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, default_value_t = false)]
    quantized: bool,
    #[arg(long)]
    quantized_weights: Option<PathBuf>,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    tokenizer_id: Option<String>,
    #[arg(long)]
    tokenizer_template: Option<String>,
    #[arg(long, default_value = "model.safetensors.index.json")]
    index_file: Option<String>,
}

async fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    log::info!("Initializing generator and loading models...");
    let start_load = std::time::Instant::now();
    let gen_args = GeneratorArgs {
        quantized: args.quantized,
        quantized_weights: args.quantized_weights.clone(),
        model_id: args.model_id,
        tokenizer_id: args.tokenizer_id,
        index_file: args.index_file,
        device,
    };

    let mut generator = Generator::new(gen_args).await?;
    log::info!(
        "Initialized generator and all models in {:.2}s.",
        start_load.elapsed().as_secs_f64()
    );

    let sample_rate = generator.audio_tokenizer.config().sample_rate as u32;

    log::info!("Generating audio for text: '{}'", args.text);
    let mut audio_stream = generator.generate_stream(
        &args.text,
        args.speaker_id,
        args.max_audio_len_ms,
        args.temperature,
        args.top_k,
        args.buffer_size,
        args.tokenizer_template.clone(),
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