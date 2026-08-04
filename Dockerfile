# Which binary from src/bin/ to build and run.
# Override at build time:
#   docker build --build-arg BIN=twilio_coordinator_server .
# Binaries behind non-default features also need FEATURES (see Cargo.toml).
# Pass a comma-separated list, no spaces:
#   docker build --build-arg BIN=websocket_server --build-arg FEATURES=tts-sarvam .
#   docker build --build-arg BIN=vaniwebrtc_server --build-arg FEATURES=vaniwebrtc,tts-sarvam .
# (vaniwebrtc additionally needs cmake in the builder — add it to the dnf line.)
ARG BIN=pizza_voice_server_v2
ARG FEATURES=""

# Builder
FROM fedora:40 AS builder
ARG BIN
ARG FEATURES

WORKDIR /app
COPY Cargo.toml Cargo.lock* build.rs ./
COPY src/ src/
# pizza_voice_server_v2 include!s dhara/pizza_order/functions.rs.
COPY dhara/ dhara/

RUN dnf install -y gcc gcc-c++ make pkg-config openssl-devel && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    dnf clean all
ENV PATH="/root/.cargo/bin:${PATH}"

# build.rs embeds src/vad/data/silero_vad_16k.bin into the binary, so the
# runtime image needs no model download for VAD. Copy the result to a fixed
# path so the runtime stage doesn't have to interpolate BIN again.
RUN cargo build --release --bin "${BIN}" ${FEATURES:+--features ${FEATURES}} && \
    cp "target/release/${BIN}" /app/rustvani-bot

# Runtime
FROM fedora:40

RUN dnf install -y ca-certificates openssl && dnf clean all

COPY --from=builder /app/rustvani-bot /usr/local/bin/rustvani-bot
ENV PORT=8080
EXPOSE 8080

CMD ["rustvani-bot"]
