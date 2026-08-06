//! WebSocket plumbing shared by every provider.
//!
//! Before the core existed, the handshake block, the `WsSink`/`WsStream`
//! aliases, the send task and the keepalive task were copied into each provider
//! file — four near-identical versions, differing mainly in the channel's item
//! type. There is one of each now.
//!
//! The *receive* loop lives in [`driver`](super::driver) instead: it needs the
//! turn gate, the front-end and the billing collector, so splitting it out here
//! would only move the coupling around.

use std::time::Duration;

use futures::SinkExt;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use super::provider::{Handshake, Outgoing};

pub type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    Message,
>;

pub type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
>;

impl From<Outgoing> for Message {
    fn from(o: Outgoing) -> Self {
        match o {
            Outgoing::Text(t) => Message::Text(t.into()),
            Outgoing::Binary(b) => Message::Binary(b.into()),
        }
    }
}

/// Host portion of a `ws://` / `wss://` URL, for the `Host` header.
pub(crate) fn host_of(url: &str) -> &str {
    url.trim_start_matches("wss://")
        .trim_start_matches("ws://")
        .split('/')
        .next()
        .unwrap_or_default()
}

/// Build the upgrade request: the provider's URL and headers plus the five
/// boilerplate WebSocket headers.
pub fn build_request(hs: &Handshake) -> Result<Request<()>, String> {
    let mut builder = Request::builder()
        .uri(&hs.url)
        .header("Host", host_of(&hs.url))
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header(
            "Sec-WebSocket-Key",
            tokio_tungstenite::tungstenite::handshake::client::generate_key(),
        );

    for (name, value) in &hs.headers {
        builder = builder.header(name.as_str(), value.as_str());
    }

    builder.body(()).map_err(|e| format!("request build failed: {}", e))
}

/// Connect and split the stream. The error string is already caller-facing;
/// the driver prefixes it with the provider name.
pub async fn connect(hs: &Handshake) -> Result<(WsSink, WsStream), String> {
    use futures::StreamExt;

    let request = build_request(hs)?;
    let (stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| format!("connect failed: {}", e))?;
    Ok(stream.split())
}

/// Drain the outbound channel onto the socket until the channel closes or the
/// socket errors.
pub async fn run_send_task(mut sink: WsSink, mut rx: mpsc::Receiver<Outgoing>, name: &'static str) {
    while let Some(msg) = rx.recv().await {
        if sink.send(msg.into()).await.is_err() {
            log::warn!("{}: send failed — closing send task", name);
            break;
        }
    }
    let _ = sink.close().await;
    log::debug!("{}: send task exited", name);
}

/// Periodic idle keepalive. Exits when the outbound channel closes.
pub async fn run_keepalive_task(
    tx: mpsc::Sender<Outgoing>,
    interval: Duration,
    msg: Outgoing,
    name: &'static str,
) {
    loop {
        tokio::time::sleep(interval).await;
        if tx.send(msg.clone()).await.is_err() {
            break;
        }
        log::trace!("{}: sent keepalive", name);
    }
    log::debug!("{}: keepalive task exited", name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_of_strips_scheme_and_path() {
        assert_eq!(host_of("wss://api.sarvam.ai/speech-to-text/ws?a=1"), "api.sarvam.ai");
        assert_eq!(host_of("ws://localhost:8080/stt"), "localhost:8080");
        assert_eq!(host_of("wss://api.deepgram.com"), "api.deepgram.com");
    }

    #[test]
    fn build_request_includes_provider_headers_and_upgrade() {
        let hs = Handshake::new("wss://api.sarvam.ai/speech-to-text/ws?model=saaras%3Av3")
            .header("api-subscription-key", "secret");
        let req = build_request(&hs).expect("request should build");
        let h = req.headers();

        assert_eq!(h.get("api-subscription-key").unwrap(), "secret");
        assert_eq!(h.get("Host").unwrap(), "api.sarvam.ai");
        assert_eq!(h.get("Upgrade").unwrap(), "websocket");
        assert_eq!(h.get("Sec-WebSocket-Version").unwrap(), "13");
        assert!(h.get("Sec-WebSocket-Key").is_some());
    }

    #[test]
    fn outgoing_converts_to_ws_message() {
        assert!(matches!(
            Message::from(Outgoing::Text("hi".into())),
            Message::Text(_)
        ));
        assert!(matches!(
            Message::from(Outgoing::Binary(vec![1, 2, 3])),
            Message::Binary(_)
        ));
    }
}
