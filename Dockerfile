# rustvani — EU Volt Interview Bot Dockerfile
#
# Fedora 40 base + ONNX runtime for Silero VAD.
#
# Build:
#   docker build -t rustvani-interview .
#
# Run server (mount .env with SARVAM_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY):
#   docker run --rm --env-file .env -p 10000:10000 rustvani-interview
#
# Run any binary:
#   docker run --rm --env-file .env -e MODE=agent_text_demo rustvani-interview

FROM fedora:40

ARG MODE=prod
ENV MODE=${MODE}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN dnf install -y \
        curl wget gcc gcc-c++ make pkg-config \
        openssl-devel ca-certificates espeak-ng \
    && dnf clean all

# ---------------------------------------------------------------------------
# Rust toolchain
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ---------------------------------------------------------------------------
# ONNX Runtime (Silero VAD)
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

COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY examples/ examples/
COPY assets/ assets/

# VAD models at expected paths
COPY src/vad/data/silero.onnx silero.onnx
COPY src/vad/data/silero_vad_16k.bin data/silero_vad_16k.bin

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
RUN cargo build --release --bin interview_voice_server

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
RUN printf '#!/bin/bash\nset -e\n\ncase "$MODE" in\n  demo)\n    echo "=== rustvani agent_text_demo ==="\n    exec cargo run --release --bin agent_text_demo\n    ;;\n  *)\n    echo "=== EU Volt Interview Bot ==="\n    exec /app/target/release/interview_voice_server\n    ;;\nesac\n' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 10000
CMD ["/entrypoint.sh"]
