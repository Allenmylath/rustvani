FROM fedora:40

# MODE=test  → runs cargo test
# MODE=prod  → runs websocket_server (default)
ARG MODE=prod

# --- System dependencies ---
# Fedora 40 has glibc 2.39 — satisfies ort __isoc23_* symbol requirements.
# Independent of Canonical's repos which have been having major outages.
RUN dnf install -y \
        curl \
        gcc \
        gcc-c++ \
        make \
        pkg-config \
        openssl-devel \
        ca-certificates \
    && dnf clean all

# --- Rust stable ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# --- Copy project ---
COPY Cargo.toml .
COPY src/ src/

# --- Download Silero ONNX model ---
# Pinned to v4.0 — inference API (input/state/sr) matches v4 only
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx" \
    -o silero_vad.onnx

# --- Download test WAV ---
RUN curl -fL \
    "https://github.com/snakers4/silero-vad/raw/master/tests/data/test.wav" \
    -o test.wav \
    && mkdir -p tests && cp test.wav tests/test.wav

# --- Build release binaries ---
# ort-sys downloads libonnxruntime prebuilt during this step.
# glibc 2.39 satisfies __isoc23_strtoll / strtol / strtoull symbols.
RUN cargo build --release

# --- Pre-compile test binaries ---
RUN cargo test --no-run

# --- Make libonnxruntime findable at runtime ---
RUN find /root/.cache -name 'libonnxruntime.so*' -exec cp {} /usr/local/lib/ \; \
    && ldconfig

# --- Entrypoint ---
RUN printf '#!/bin/bash\nif [ "$MODE" = "test" ]; then\n  exec cargo test -- --test-output immediate\nelse\n  exec /app/target/release/websocket_server\nfi\n' \
    > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
