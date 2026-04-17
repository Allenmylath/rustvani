FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# MODE=test  → runs cargo test
# MODE=prod  → runs websocket_server (default)
ARG MODE=prod

# --- System dependencies ---
RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Rust stable ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# --- Copy project ---
COPY Cargo.toml .
COPY src/ src/

# --- Download Silero ONNX model ---
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx" \
    -o silero_vad.onnx

# --- Download test WAV ---
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/master/tests/data/test.wav" \
    -o test.wav \
    && mkdir -p tests && cp test.wav tests/test.wav

# --- Build ---
RUN cargo build --release
RUN cargo test --no-run

# --- Make libonnxruntime findable ---
RUN find /root/.cache -name 'libonnxruntime.so*' -exec cp {} /usr/local/lib/ \; \
    && ldconfig

# --- Entrypoint based on MODE ---
RUN echo '#!/bin/bash\n\
    if [ "$MODE" = "test" ]; then\n\
    exec cargo test -- --test-output immediate\n\
    else\n\
    exec /app/target/release/websocket_server\n\
    fi' > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]