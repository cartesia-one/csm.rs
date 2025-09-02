use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
    body::Body
};
use bytes::Bytes;
use clap::Parser;
use csm_rs::{Generator, GeneratorArgs};
use futures_util::StreamExt;
use http_body_util::StreamBody;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

struct AppState {
    generator: Mutex<Generator>,
    api_key: Option<String>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long)]
    api_key: Option<String>,
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
    #[serde(default = "default_buffer_size")]
    buffer_size: usize,
    tokenizer_template: Option<String>,

    model: Option<String>,
    voice: Option<String>,
    instructions: Option<String>,
    response_format: Option<String>,
}

fn default_temperature() -> f64 { 0.7 }
fn default_top_k() -> usize { 100 }
fn default_max_audio_len_ms() -> f32 { 30000.0 }
fn default_buffer_size() -> usize { 20 }


#[derive(Serialize)]
struct ErrorResponse {
    error: String,
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

    log::info!("Loading model...");
    let gen_args = GeneratorArgs {
        quantized: args.quantized,
        quantized_weights: args.quantized_weights.clone(),
        model_id: args.model_id.clone(),
        tokenizer_id: args.tokenizer_id.clone(),
        index_file: args.index_file.clone(),
        device,
    };

    let generator = Generator::new(gen_args).await?;

    let shared_state = Arc::new(AppState {
        generator: Mutex::new(generator),
        api_key: args.api_key.clone(),
    });

    let app = Router::new()
        .route("/v1/audio/speech", post(speech_handler))
        .with_state(shared_state.clone())
        .route_layer(middleware::from_fn_with_state(shared_state, auth_middleware));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    log::info!("🚀 Starting server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
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
    let (tx, rx) = mpsc::channel::<Result<Bytes, axum::Error>>(16);

    let sample_rate = {
        let generator = state.generator.lock().await;
        generator.audio_tokenizer.config().sample_rate as u32
    };

    let header = create_wav_header(sample_rate, 1, 16);
    if tx.send(Ok(Bytes::from(header))).await.is_err() {
        log::error!("Failed to send WAV header: client disconnected.");
        return (StatusCode::INTERNAL_SERVER_ERROR, "Client disconnected").into_response();
    }

    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        let mut generator = state_clone.generator.lock().await;

        let mut audio_stream = generator.generate_stream(
            &payload.input,
            payload.speaker_id,
            payload.max_audio_len_ms,
            payload.temperature,
            payload.top_k,
            payload.buffer_size,
            payload.tokenizer_template,
        );

        while let Some(chunk_result) = audio_stream.next().await {
            let bytes_result = chunk_result
                .and_then(convert_tensor_to_bytes)
                .map_err(|e| axum::Error::new(e));

            if tx.send(bytes_result).await.is_err() {
                log::info!("Client disconnected, stopping generation stream.");
                break;
            }
        }
    });

    let body = StreamBody::new(ReceiverStream::new(rx));
    
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "audio/wav".parse().unwrap());

    (headers, Body::from_stream(body)).into_response()
}


fn convert_tensor_to_bytes(tensor: candle_core::Tensor) -> Result<Bytes> {
    let audio_f32 = tensor.to_vec1::<f32>()?;
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