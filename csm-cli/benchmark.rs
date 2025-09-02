use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use csm_rs::{Generator, GeneratorArgs};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::time::Instant;

async fn generate_audio_blocking(
    generator: &mut Generator,
    text: &str,
    speaker_id: u32,
    temperature: f64,
    top_k: usize,
) -> Result<Tensor> {
    let max_len_ms = 1000.0;
    let mut stream =
        generator.generate_stream(text, speaker_id, max_len_ms, temperature, top_k);
    let mut chunks = vec![];
    while let Some(res) = stream.next().await {
        chunks.push(res?);
    }

    if chunks.is_empty() {
        Ok(Tensor::zeros((0,), DType::F32, &Device::Cpu)?)
    } else {
        Ok(Tensor::cat(&chunks, 0)?)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "Hi there, this is a test")]
    text: String,
    #[arg(long, default_value_t = 0)]
    speaker_id: u32,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 100)]
    top_k: usize,
    #[arg(short, long, default_value_t = 1)]
    warmup_runs: u32,
    #[arg(short, long, default_value_t = 5)]
    num_runs: u32,
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
    index_file: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    log::info!("--- RUNNING BENCHMARK ---");
    let start_load = std::time::Instant::now();
    let gen_args = GeneratorArgs {
        quantized: args.quantized,
        quantized_weights: args.quantized_weights.clone(),
        model_id: args.model_id,
        tokenizer_id: args.tokenizer_id,
        index_file: args.index_file,
        device: device.clone(),
    };
    let mut generator = Generator::new(gen_args).await?;
    log::info!(
        "Initialized generator and all models in {:.2}s.",
        start_load.elapsed().as_secs_f64()
    );

    let sample_rate = generator.audio_tokenizer.config().sample_rate as f64;

    log::info!("Starting {} warm-up runs...", args.warmup_runs);
    for i in 0..args.warmup_runs {
        let _ = generate_audio_blocking(
            &mut generator,
            &args.text,
            args.speaker_id,
            args.temperature,
            args.top_k,
        )
        .await?;
        log::info!("Warm-up run {}/{} complete.", i + 1, args.warmup_runs);
    }

    log::info!("Starting {} timed benchmark runs...", args.num_runs);
    let mut total_duration = std::time::Duration::new(0, 0);
    let mut total_audio_len_s = 0.0;

    for i in 0..args.num_runs {
        let start_time = Instant::now();
        let audio_tensor = generate_audio_blocking(
            &mut generator,
            &args.text,
            args.speaker_id,
            args.temperature,
            args.top_k,
        )
        .await?;
        let elapsed = start_time.elapsed();

        let audio_len_s = audio_tensor.dim(0)? as f64 / sample_rate;
        total_duration += elapsed;
        total_audio_len_s += audio_len_s;

        log::info!(
            "Run {}/{}: Generated {:.2}s of audio in {:.2}s.",
            i + 1,
            args.num_runs,
            audio_len_s,
            elapsed.as_secs_f64()
        );
    }

    println!("\n--- Benchmark Results ---");
    print_results(total_duration, total_audio_len_s, args.num_runs, &device);

    Ok(())
}

fn print_results(
    total_duration: std::time::Duration,
    total_audio_len_s: f64,
    num_runs: u32,
    device: &Device,
) {
    let avg_gen_time_s = total_duration.as_secs_f64() / num_runs as f64;
    let avg_audio_len_s = total_audio_len_s / num_runs as f64;
    if avg_audio_len_s == 0.0 {
        println!("No audio was generated, cannot calculate RTF.");
        return;
    }
    let rtf = avg_gen_time_s / avg_audio_len_s;
    let throughput = avg_audio_len_s / avg_gen_time_s;

    println!("Model: CSM-1B ({:?})", device.location());
    println!("Number of runs: {}", num_runs);
    println!("Average audio generated: {:.2} seconds", avg_audio_len_s);
    println!("Average generation time: {:.2} seconds", avg_gen_time_s);
    println!("-------------------------");
    println!("Real-Time Factor (RTF): {:.3}", rtf);
    println!("Throughput (xRealTime): {:.3}x", throughput);
    println!("-------------------------");
    println!("RTF is the time taken to generate 1s of audio. Lower is better.");
    println!("Throughput is how many seconds of audio are generated in 1s. Higher is better.");
}
