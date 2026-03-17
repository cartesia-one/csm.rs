# Multi-stage build for csm.rs with CUDA support
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    gcc-10 \
    g++-10 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC-10 as default for NVCC compatibility
ENV CC=gcc-10
ENV CXX=g++-10
ENV NVCC_CCBIN=/usr/bin/gcc-10

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy workspace files
COPY Cargo.toml ./
COPY csm-core ./csm-core
COPY csm-cli ./csm-cli
COPY csm-server ./csm-server

# Set CUDA compute capability (use 80 for A100, 86 for RTX 30xx, 89 for RTX 40xx, 90 for H100)
# This skips nvidia-smi detection which isn't available at build time
ARG CUDA_COMPUTE_CAP=90
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

# Build with CUDA support
RUN cargo build --release --features cudnn

# Runtime stage
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 csm && \
    mkdir -p /app /models && \
    chown -R csm:csm /app /models

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/server /app/server

# Switch to non-root user
USER csm

# Expose port
EXPOSE 8080

# Set default environment variables
ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
ENTRYPOINT ["/app/server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--model-id", "cartesia/azzurra-voice"]
