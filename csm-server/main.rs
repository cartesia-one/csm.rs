use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    serve::ListenerExt,
    Json, Router,
    body::Body
};
use bytes::Bytes;
use clap::Parser;
use csm_rs::{Generator, GeneratorArgs};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

struct AppState {
    generator: std::sync::Mutex<Generator>,
    api_key: Option<String>,
    default_buffer_size: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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

    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long, default_value_t = 10, help = "Default buffer size for streaming (number of frames to buffer before sending). Can be overridden per-request.")]
    buffer_size: usize,
}

#[derive(Deserialize, Debug)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    speaker_id: u32,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_max_audio_len_ms")]
    max_audio_len_ms: f32,
    #[serde(default)]
    buffer_size: usize,
    tokenizer_template: Option<String>
}

fn default_temperature() -> f64 { 0.7 }
fn default_top_k() -> usize { 100 }
fn default_max_audio_len_ms() -> f32 { 30000.0 }


#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn parse_dtype(s: &str) -> anyhow::Result<candle_core::DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(candle_core::DType::F32),
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        _ => anyhow::bail!("Unsupported dtype '{}'. Use f32, f16, or bf16.", s),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let device = if args.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
    };

    let dtype = args.dtype.as_deref().map(parse_dtype).transpose()?;

    log::info!("Loading model...");
    let gen_args = GeneratorArgs {
        weights_path: args.weights_path,
        model_id: args.model_id,
        model_path: args.model_path,
        model_file: args.model_file,
        index_file: args.index_file,
        tokenizer_id: args.tokenizer_id,
        device,
        dtype,
    };

    let generator = Generator::new(gen_args).await?;

    let shared_state = Arc::new(AppState {
        generator: std::sync::Mutex::new(generator),
        api_key: args.api_key.clone(),
        default_buffer_size: args.buffer_size,
    });

    let app = Router::new()
        .route("/v1/audio/speech", post(speech_handler))
        .with_state(shared_state.clone())
        .route_layer(middleware::from_fn_with_state(shared_state, auth_middleware));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    log::info!("🚀 Starting server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?
        .tap_io(|tcp_stream| {
            if let Err(err) = tcp_stream.set_nodelay(true) {
                log::warn!("failed to set TCP_NODELAY on incoming connection: {err:#}");
            }
        });
    axum::serve(listener, app).await?;

    Ok(())
}


async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(ref api_key) = state.api_key {
        let auth_header = request
            .headers()
            .get("Authorization")
            .and_then(|value| value.to_str().ok());

        if let Some(auth_value) = auth_header {
            if let Some(token) = auth_value.strip_prefix("Bearer ") {
                if token == api_key {
                    return next.run(request).await;
                }
            }
        }

        let error_response = Json(ErrorResponse {
            error: "Invalid or missing API key".to_string(),
        });
        return (StatusCode::UNAUTHORIZED, error_response).into_response();
    }
    next.run(request).await
}

async fn speech_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SpeechRequest>,
) -> impl IntoResponse {
    let buffer_size = if payload.buffer_size == 0 {
        state.default_buffer_size
    } else {
        payload.buffer_size
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);

    let state_clone = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let mut generator = state_clone.generator.lock().unwrap();

        // Send WAV header first
        let sample_rate = generator.audio_tokenizer.config().sample_rate as u32;
        let header = create_wav_header(sample_rate, 1, 16);
        if tx.blocking_send(Ok(Bytes::from(header))).is_err() {
            log::error!("Failed to send WAV header: client disconnected.");
            return;
        }

        // Use a std::sync::mpsc channel for the synchronous generation loop.
        // generate_to_channel runs blocking GPU work and sends audio tensors
        // through audio_tx as they're produced. We then forward them to the
        // async tokio channel from a second thread so streaming works in real time.
        let (audio_tx, audio_rx) = std::sync::mpsc::sync_channel::<anyhow::Result<candle_core::Tensor>>(2);

        let input = payload.input.clone();
        let tokenizer_template = payload.tokenizer_template.clone();
        let speaker_id = payload.speaker_id;
        let max_audio_len_ms = payload.max_audio_len_ms;
        let temperature = payload.temperature;
        let top_k = payload.top_k;

        // Forward audio chunks to the HTTP response stream in a separate thread,
        // so that generate_to_channel (which blocks on GPU) can proceed in parallel
        // with hyper flushing data to the client.
        let forwarder = std::thread::spawn(move || {
            for chunk_result in audio_rx {
                let bytes_result = chunk_result
                    .and_then(convert_tensor_to_bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

                if tx.blocking_send(bytes_result).is_err() {
                    log::info!("Client disconnected, stopping generation stream.");
                    break;
                }
            }
        });

        generator.generate_to_channel(
            &input,
            speaker_id,
            max_audio_len_ms,
            temperature,
            top_k,
            buffer_size,
            tokenizer_template,
            audio_tx,
        );

        // Wait for forwarder to finish sending all chunks
        let _ = forwarder.join();
    });

    let stream = ReceiverStream::new(rx);

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "audio/wav".parse().unwrap());

    (headers, Body::from_stream(stream)).into_response()
}


fn convert_tensor_to_bytes(tensor: candle_core::Tensor) -> Result<Bytes> {
    let audio_f32 = tensor.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let audio_i16: Vec<i16> = audio_f32
        .iter()
        .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            audio_i16.as_ptr() as *const u8,
            audio_i16.len() * std::mem::size_of::<i16>(),
        )
    };
    
    Ok(Bytes::copy_from_slice(byte_slice))
}


fn create_wav_header(sample_rate: u32, num_channels: u16, bits_per_sample: u16) -> Vec<u8> {
    let mut header = Vec::with_capacity(44);

    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&u32::MAX.to_le_bytes());
    header.extend_from_slice(b"WAVE");

    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&num_channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    header.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * bits_per_sample / 8;
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());

    header.extend_from_slice(b"data");
    header.extend_from_slice(&u32::MAX.to_le_bytes());

    header
}