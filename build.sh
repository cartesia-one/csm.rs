#!/bin/sh

# mkl

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin benchmark --features "candle-core/mkl candle-nn/mkl candle-transformers/mkl" -- --quantized --quantized-weights ./q8.gguf

# cuda

cargo run --release --bin benchmark --features "candle-core/cuda candle-nn/cuda candle-transformers/cuda" -- --quantized --quantized-weights ./q8.gguf
