# rustvani — Test & Run Dockerfile
#
# Kept the original Fedora 40 base + ONNX runtime setup that builds reliably.
# Added missing test directories and a flexible entrypoint.
#
# Build:
#   docker build -t rustvani .
#
# Run tests:
#   docker run --rm -e MODE=test rustvani
#
# Run agent text demo (mount your .env with OPENAI_API_KEY):
#   docker run --rm --env-file .env -e MODE=demo rustvani
#
# Run any binary:
#   docker run --rm --env-file .env -e MODE=agent_text_demo rustvani

FROM fedora:40

ARG MODE=prod
ENV MODE=${MODE}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN dnf install -y \
    curl \
    wget \
    gcc \
    gcc-c++ \
    make \
    pkg-config \
    openssl-devel \
    ca-certificates \
    espeak-ng \
    && dnf clean all

# ---------------------------------------------------------------------------
# Rust toolchain
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ---------------------------------------------------------------------------
# ONNX Runtime (matches the ort crate's expected layout)
# ---------------------------------------------------------------------------
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
    -O /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz
ENV ORT_LIB_LOCATION=/opt/onnxruntime
ENV ORT_SKIP_DOWNLOAD=1
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

# ---------------------------------------------------------------------------
# Build environment
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy manifests
COPY Cargo.toml .
COPY Cargo.lock .

# Copy all source, tests, and assets
COPY src/ src/
COPY dhara/ dhara/
COPY tests/ tests/
COPY examples/ examples/
COPY assets/ assets/

# Model cache directory — resolved at runtime via RUSTVANI_CACHE_DIR
ENV RUSTVANI_CACHE_DIR=/app/data
RUN mkdir -p /app/data
COPY src/vad/data/silero.onnx data/silero.onnx
COPY src/vad/data/silero_vad_16k.bin data/silero_vad_16k.bin
COPY src/turn/smart_turn_weights.bin.gz data/smart_turn_weights.bin.gz

# ---------------------------------------------------------------------------
# Build release + compile tests
# ---------------------------------------------------------------------------
RUN cargo build --release --features db-postgres
RUN cargo test --no-run --features db-postgres

# ---------------------------------------------------------------------------
# Entrypoint — test / demo / any binary
# ---------------------------------------------------------------------------
RUN printf '#!/bin/bash\nset -e\n\nif [ "$MODE" = "test" ]; then\n  echo "=== rustvani test mode ==="\n  exec cargo test -- --test-threads=1 --nocapture\n\nelif [ "$MODE" = "demo" ]; then\n  echo "=== rustvani agent_text_demo ==="\n  exec cargo run --release --bin agent_text_demo\n\nelse\n  echo "=== rustvani pizza_voice_server ==="\n  exec /app/target/release/pizza_voice_server\nfi\n' > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
