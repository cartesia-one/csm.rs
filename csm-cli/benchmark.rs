use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use csm_rs::{Generator, GeneratorArgs};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct BenchmarkResult {
    total_duration: Duration,
    ttfb: Duration,
    audio_tensor: Tensor,
}

async fn generate_audio_blocking(
    generator: &mut Generator,
    text: &str,
    speaker_id: u32,
    temperature: f64,
    top_k: usize,
    buffer_size: usize,
    tokenizer_template: Option<String>,
) -> Result<BenchmarkResult> {
    let max_len_ms = 1000.0;
    let start_time = Instant::now();
    let mut stream =
        generator.generate_stream(text, speaker_id, max_len_ms, temperature, top_k, buffer_size, tokenizer_template);

    let mut chunks = vec![];
    let mut ttfb = Duration::ZERO;

    while let Some(res) = stream.next().await {
        if chunks.is_empty() {
            ttfb = start_time.elapsed();
        }
        chunks.push(res?);
    }

    let total_duration = start_time.elapsed();

    let audio_tensor = if chunks.is_empty() {
        Tensor::zeros((0,), DType::F32, &Device::Cpu)?
    } else {
        Tensor::cat(&chunks, 0)?
    };

    Ok(BenchmarkResult {
        total_duration,
        ttfb,
        audio_tensor,
    })
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
    #[arg(long, default_value_t = 1)]
    buffer_size: usize,
    #[arg(short, long, default_value_t = 1)]
    warmup_runs: u32,
    #[arg(short, long, default_value_t = 5)]
    num_runs: u32,
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long, help = "Data type for model weights: f32, f16, bf16. Defaults to f16 on CUDA, f32 on CPU.")]
    dtype: Option<String>,

    #[arg(long, help = "Absolute path to a weight file (.safetensors or .gguf). Overrides all other model loading options.")]
    weights_path: Option<PathBuf>,
    #[arg(long, help = "The model ID from the Hugging Face Hub (e.g., 'sesame/csm-1b').")]
    model_id: Option<String>,
    #[arg(long, help = "Path to a local directory containing the model files.")]
    model_path: Option<PathBuf>,
    #[arg(long, help = "The name of a single model file to use within a --model-id or --model-path.")]
    model_file: Option<String>,
    #[arg(long, help = "The name of the index file for sharded models.")]
    index_file: Option<String>,
    #[arg(long, help = "The tokenizer ID from the Hugging Face Hub. Defaults to the --model-id if not set.")]
    tokenizer_id: Option<String>,
    #[arg(long)]
    tokenizer_template: Option<String>,
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        _ => anyhow::bail!("Unsupported dtype '{}'. Use f32, f16, or bf16.", s),
    }
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

    let dtype = args.dtype.as_deref().map(parse_dtype).transpose()?;

    log::info!("--- RUNNING BENCHMARK ---");
    let start_load = std::time::Instant::now();
    let gen_args = GeneratorArgs {
        weights_path: args.weights_path,
        model_id: args.model_id,
        model_path: args.model_path,
        model_file: args.model_file,
        index_file: args.index_file,
        tokenizer_id: args.tokenizer_id,
        device: device.clone(),
        dtype,
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
            args.buffer_size,
            args.tokenizer_template.clone(),
        )
        .await?;
        log::info!("Warm-up run {}/{} complete.", i + 1, args.warmup_runs);
    }

    log::info!("Starting {} timed benchmark runs...", args.num_runs);
    let mut total_duration = Duration::ZERO;
    let mut total_ttfb = Duration::ZERO;
    let mut total_audio_len_s = 0.0;

    for i in 0..args.num_runs {
        let result = generate_audio_blocking(
            &mut generator,
            &args.text,
            args.speaker_id,
            args.temperature,
            args.top_k,
            args.buffer_size,
            args.tokenizer_template.clone(),
        )
        .await?;

        let audio_len_s = result.audio_tensor.dim(0)? as f64 / sample_rate;
        total_duration += result.total_duration;
        total_ttfb += result.ttfb;
        total_audio_len_s += audio_len_s;

        log::info!(
            "Run {}/{}: Generated {:.2}s of audio in {:.2}s (TTFB: {:.0}ms)",
            i + 1,
            args.num_runs,
            audio_len_s,
            result.total_duration.as_secs_f64(),
            result.ttfb.as_secs_f64() * 1000.0,
        );
    }

    println!("\n--- Benchmark Results ---");
    print_results(total_duration, total_ttfb, total_audio_len_s, args.num_runs, args.buffer_size, &device);

    Ok(())
}

fn print_results(
    total_duration: Duration,
    total_ttfb: Duration,
    total_audio_len_s: f64,
    num_runs: u32,
    buffer_size: usize,
    device: &Device,
) {
    let avg_gen_time_s = total_duration.as_secs_f64() / num_runs as f64;
    let avg_audio_len_s = total_audio_len_s / num_runs as f64;
    let avg_ttfb_ms = (total_ttfb.as_secs_f64() / num_runs as f64) * 1000.0;

    if avg_audio_len_s == 0.0 {
        println!("No audio was generated, cannot calculate RTF.");
        return;
    }
    let rtf = avg_gen_time_s / avg_audio_len_s;
    let throughput = avg_audio_len_s / avg_gen_time_s;

    println!("Device: {:?}", device.location());
    println!("Number of runs: {}", num_runs);
    println!("Buffer size: {} frames", buffer_size);
    println!("Average audio generated: {:.2} seconds", avg_audio_len_s);
    println!("Average generation time: {:.2} seconds", avg_gen_time_s);
    println!("Average TTFB: {:.0}ms", avg_ttfb_ms);
    println!("-------------------------");
    println!("Real-Time Factor (RTF): {:.3}", rtf);
    println!("Throughput (xRealTime): {:.3}x", throughput);
    println!("-------------------------");
    println!("RTF is the time taken to generate 1s of audio. Lower is better.");
    println!("Throughput is how many seconds of audio are generated in 1s. Higher is better.");
    println!("TTFB is the time until the first audio chunk is ready to stream. Lower is better.");
}
