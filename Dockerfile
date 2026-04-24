FROM fedora:40

ARG MODE=prod

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

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Pre-fetch ONNX Runtime 1.22.0 from official GitHub releases ---
# Bypasses ort-sys auto-download from cdn.pyke.io entirely.
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
    -O /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

ENV ORT_LIB_LOCATION=/opt/onnxruntime
ENV ORT_SKIP_DOWNLOAD=1

WORKDIR /app
COPY Cargo.toml .
COPY src/ src/
COPY src/vad/data/silero.onnx silero.onnx

RUN cargo build --release
RUN cargo test --no-run

# --- Make libonnxruntime findable at runtime ---
RUN cp /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/ \
    && ldconfig

RUN printf '#!/bin/bash\nif [ "$MODE" = "test" ]; then\n  exec cargo test -- --test-output immediate\nelse\n  exec /app/target/release/websocket_server\nfi\n' \
    > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
