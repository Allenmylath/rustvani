FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
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
# Pinned to v4.0 tag — our inference API (input/state/sr) matches v4.
# The master branch now contains v5 which has a different API.
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx" \
    -o silero_vad.onnx

# --- Download official Silero test WAV ---
# Real speech recording from the Silero repo — required for meaningful
# VAD state transitions. Synthetic audio will not trigger the model.
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/master/tests/data/test.wav" \
    -o test.wav

# --- Build the VAD integration test binary ---
# ort-sys downloads libonnxruntime prebuilt during this step.
# Ubuntu 24.04 provides glibc 2.39 which satisfies ort 1.20.x __isoc23_* symbols.
RUN cargo build --release --bin vad_test

# --- Make the downloaded libonnxruntime findable at runtime ---
RUN find /root/.cache -name 'libonnxruntime.so*' -exec cp {} /usr/local/lib/ \; \
    && ldconfig

# --- Run ---
CMD ["/app/target/release/vad_test"]