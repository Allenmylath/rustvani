//! `VaniWebRTCTransport` — a peer-to-peer WebRTC transport (no SFU).
//!
//! A thin protocol wrapper around [`BaseTransport`]: it exposes `input()` /
//! `output()` `FrameProcessor`s for the pipeline and drives a single
//! `RTCPeerConnection` per connection. Audio is carried as Opus/RTP/SRTP P2P;
//! control messages (interruption / RAVI / client-VAD) ride a reliable data
//! channel. Mirrors `WebSocketTransport::run_socket`.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures::stream::{SplitSink, StreamExt};
use futures::SinkExt;
use tokio::sync::{mpsc, Mutex};

use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::api::APIBuilder;
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;
use webrtc::ice_transport::ice_candidate::{RTCIceCandidate, RTCIceCandidateInit};
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::media::Sample;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTPCodecType};
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_local::TrackLocal;
use webrtc::track::track_remote::TrackRemote;

use crate::frames::{AudioRawData, Frame, FrameDirection, FrameProcessor};
use crate::transport::base::BaseTransport;
use crate::transport::incoming::dispatch_text_message;
use crate::transport::output::OutputMessage;

use super::codec::{OpusInbound, OpusOutbound};
use super::params::VaniWebRTCParams;
use super::signaling::{munge_answer_sdp, SignalMsg};

const AUDIO_OUT_CHANNEL_CAP: usize = 150;
/// Duration of each encoded Opus sample (20 ms frame at 48 kHz).
const OPUS_SAMPLE_DURATION: Duration = Duration::from_millis(20);

type SharedWsTx = Arc<Mutex<SplitSink<WebSocket, Message>>>;

// ---------------------------------------------------------------------------
// VaniWebRTCTransport
// ---------------------------------------------------------------------------

pub struct VaniWebRTCTransport {
    base:         Arc<BaseTransport>,
    audio_out_rx: std::sync::Mutex<Option<mpsc::Receiver<OutputMessage>>>,
    params:       VaniWebRTCParams,
}

impl VaniWebRTCTransport {
    pub fn new(name: &str, params: VaniWebRTCParams) -> Self {
        let base = Arc::new(BaseTransport::new(name, params.transport.clone()));

        let (audio_out_tx, audio_out_rx) = mpsc::channel::<OutputMessage>(AUDIO_OUT_CHANNEL_CAP);
        base.set_audio_out_tx(audio_out_tx);

        Self {
            base,
            audio_out_rx: std::sync::Mutex::new(Some(audio_out_rx)),
            params,
        }
    }

    /// The input FrameProcessor — place first in the pipeline.
    pub fn input(&self) -> FrameProcessor {
        self.base.input()
    }

    /// The output FrameProcessor — place last in the pipeline.
    pub fn output(&self) -> FrameProcessor {
        self.base.output()
    }

    /// Build a fresh `RTCPeerConnection` with Opus registered and the
    /// configured ICE servers.
    async fn build_peer_connection(&self) -> webrtc::error::Result<Arc<RTCPeerConnection>> {
        let mut media = MediaEngine::default();
        media.register_default_codecs()?;

        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut media)?;

        let api = APIBuilder::new()
            .with_media_engine(media)
            .with_interceptor_registry(registry)
            .build();

        let config = RTCConfiguration {
            ice_servers: vec![RTCIceServer {
                urls: self.params.ice_servers.clone(),
                ..Default::default()
            }],
            ..Default::default()
        };

        Ok(Arc::new(api.new_peer_connection(config).await?))
    }

    /// Drive the WebRTC peer connection until it closes.
    ///
    /// `socket` carries signaling (SDP offer/answer + trickle ICE) as JSON
    /// [`SignalMsg`]. Media flows P2P over SRTP once negotiated; control
    /// messages flow over the data channel the client opens.
    pub async fn run(
        &self,
        socket: WebSocket,
        push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ) {
        let mut audio_out_rx = self
            .audio_out_rx
            .lock()
            .unwrap()
            .take()
            .expect("run called more than once on the same VaniWebRTCTransport");

        let pc = match self.build_peer_connection().await {
            Ok(pc) => pc,
            Err(e) => {
                log::error!("vaniwebrtc: failed to build peer connection: {}", e);
                return;
            }
        };

        // Split the signaling socket so the ICE callback can trickle candidates
        // while the main loop handles offers and pipeline output.
        let (ws_tx, mut ws_rx) = socket.split();
        let ws_tx: SharedWsTx = Arc::new(Mutex::new(ws_tx));

        // ---- Local audio track (bot → client) ----
        let local_track = Arc::new(TrackLocalStaticSample::new(
            RTCRtpCodecCapability {
                mime_type: MIME_TYPE_OPUS.to_owned(),
                ..Default::default()
            },
            "audio".to_owned(),
            "rustvani".to_owned(),
        ));
        match pc
            .add_track(Arc::clone(&local_track) as Arc<dyn TrackLocal + Send + Sync>)
            .await
        {
            Ok(rtp_sender) => {
                // Drain inbound RTCP so the sender keeps running.
                tokio::spawn(async move {
                    let mut rtcp_buf = vec![0u8; 1500];
                    while rtp_sender.read(&mut rtcp_buf).await.is_ok() {}
                });
            }
            Err(e) => log::error!("vaniwebrtc: add_track failed: {}", e),
        }

        // ---- Inbound audio track (client → bot) ----
        {
            let base = self.base.clone();
            let out_rate = self.params.transport.audio_in_sample_rate.unwrap_or(16_000);
            let denoiser_factory = self.params.denoiser_factory.clone();
            pc.on_track(Box::new(move |track: Arc<TrackRemote>, _recv, _trans| {
                let base = base.clone();
                let denoiser_factory = denoiser_factory.clone();
                Box::pin(async move {
                    if track.kind() != RTPCodecType::Audio {
                        return;
                    }
                    let denoiser = denoiser_factory.as_ref().map(|f| f());
                    let mut inbound = OpusInbound::new(out_rate, denoiser);
                    tokio::spawn(async move {
                        loop {
                            match track.read_rtp().await {
                                Ok((packet, _)) => {
                                    let pcm = inbound.push_rtp(&packet.payload);
                                    if !pcm.is_empty() {
                                        let data = AudioRawData::new(pcm, inbound.out_rate(), 1);
                                        base.push_audio_frame(data).await;
                                    }
                                }
                                Err(e) => {
                                    log::debug!("vaniwebrtc: inbound track ended: {}", e);
                                    break;
                                }
                            }
                        }
                    });
                })
            }));
        }

        // ---- Control data channel (client → bot text) ----
        let dc_slot: Arc<Mutex<Option<Arc<RTCDataChannel>>>> = Arc::new(Mutex::new(None));
        {
            let dc_slot = dc_slot.clone();
            let push_tx = push_tx.clone();
            pc.on_data_channel(Box::new(move |dc: Arc<RTCDataChannel>| {
                let dc_slot = dc_slot.clone();
                let push_tx = push_tx.clone();
                Box::pin(async move {
                    *dc_slot.lock().await = Some(dc.clone());
                    let push_tx = push_tx.clone();
                    dc.on_message(Box::new(move |msg: DataChannelMessage| {
                        let push_tx = push_tx.clone();
                        Box::pin(async move {
                            if msg.is_string {
                                if let Ok(text) = String::from_utf8(msg.data.to_vec()) {
                                    dispatch_text_message(&text, &push_tx).await;
                                }
                            }
                        })
                    }));
                })
            }));
        }

        // ---- Trickle local ICE candidates to the client ----
        {
            let ws_tx = ws_tx.clone();
            pc.on_ice_candidate(Box::new(move |c: Option<RTCIceCandidate>| {
                let ws_tx = ws_tx.clone();
                Box::pin(async move {
                    if let Some(c) = c {
                        if let Ok(init) = c.to_json() {
                            let msg = SignalMsg::Ice {
                                candidate:       init.candidate,
                                sdp_mid:         init.sdp_mid,
                                sdp_mline_index: init.sdp_mline_index,
                            };
                            send_signal(&ws_tx, msg).await;
                        }
                    }
                })
            }));
        }

        // ---- Main event loop ----
        let mut outbound = OpusOutbound::new();
        let audio_out_rate = self.params.transport.audio_out_sample_rate.unwrap_or(16_000);

        loop {
            tokio::select! {
                // Signaling: offer / ICE / bye from the client.
                maybe_msg = ws_rx.next() => {
                    match maybe_msg {
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<SignalMsg>(&text) {
                                Ok(SignalMsg::Offer { sdp }) => {
                                    if let Err(e) = self
                                        .handle_offer(&pc, &ws_tx, sdp)
                                        .await
                                    {
                                        log::warn!("vaniwebrtc: offer handling failed: {}", e);
                                    }
                                }
                                Ok(SignalMsg::Ice { candidate, sdp_mid, sdp_mline_index }) => {
                                    let init = RTCIceCandidateInit {
                                        candidate,
                                        sdp_mid,
                                        sdp_mline_index,
                                        username_fragment: None,
                                    };
                                    if let Err(e) = pc.add_ice_candidate(init).await {
                                        log::warn!("vaniwebrtc: add_ice_candidate failed: {}", e);
                                    }
                                }
                                Ok(SignalMsg::Bye) => break,
                                Ok(SignalMsg::Answer { .. }) => {} // server is answerer; ignore
                                Err(e) => log::warn!("vaniwebrtc: bad signaling message: {}", e),
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            log::debug!("vaniwebrtc: signaling socket closed");
                            break;
                        }
                        Some(Ok(_)) => {} // binary/ping/pong on signaling channel — ignore
                        Some(Err(e)) => {
                            log::warn!("vaniwebrtc: signaling error: {}", e);
                            break;
                        }
                    }
                }

                // Pipeline output → client.
                output_msg = audio_out_rx.recv() => {
                    match output_msg {
                        Some(OutputMessage::Audio(pcm)) => {
                            for packet in outbound.push_pcm(&pcm, audio_out_rate) {
                                let sample = Sample {
                                    data:     bytes::Bytes::from(packet),
                                    duration: OPUS_SAMPLE_DURATION,
                                    ..Default::default()
                                };
                                if local_track.write_sample(&sample).await.is_err() {
                                    log::warn!("vaniwebrtc: write_sample failed");
                                }
                            }
                        }

                        Some(OutputMessage::Text(json)) => {
                            if let Some(dc) = dc_slot.lock().await.clone() {
                                let _ = dc.send_text(json).await;
                            }
                        }

                        Some(OutputMessage::Interruption) => {
                            // Drain audio queued before the marker, then reset
                            // the encoder so stale frames don't play out.
                            while let Ok(queued) = audio_out_rx.try_recv() {
                                match queued {
                                    OutputMessage::Interruption => break,
                                    OutputMessage::Audio(_) | OutputMessage::Text(_) => {}
                                }
                            }
                            outbound.reset();
                            if let Some(dc) = dc_slot.lock().await.clone() {
                                let _ = dc.send_text(r#"{"type":"interruption"}"#).await;
                            }
                            log::debug!("vaniwebrtc: sent interruption to client");
                        }

                        None => break, // pipeline shut down
                    }
                }
            }
        }

        let _ = pc.close().await;
        let _ = push_tx
            .send((Frame::end(), FrameDirection::Downstream))
            .await;
    }

    /// Set the remote offer, create + apply an answer, and send the (Opus-tuned)
    /// answer SDP back to the client. ICE candidates trickle separately.
    async fn handle_offer(
        &self,
        pc: &Arc<RTCPeerConnection>,
        ws_tx: &SharedWsTx,
        sdp: String,
    ) -> webrtc::error::Result<()> {
        pc.set_remote_description(RTCSessionDescription::offer(sdp)?).await?;

        let answer = pc.create_answer(None).await?;
        // Apply the canonical answer locally (this kicks off ICE gathering)…
        pc.set_local_description(answer.clone()).await?;
        // …but send the client a munged copy that forces high-bitrate full-band
        // Opus. Only the fmtp line differs, so ICE/DTLS parameters still match.
        let munged = munge_answer_sdp(&answer.sdp, &self.params);
        send_signal(ws_tx, SignalMsg::Answer { sdp: munged }).await;
        Ok(())
    }
}

/// Serialize and send a signaling message over the (shared) WebSocket sink.
async fn send_signal(ws_tx: &SharedWsTx, msg: SignalMsg) {
    if let Ok(json) = serde_json::to_string(&msg) {
        let mut guard = ws_tx.lock().await;
        let _ = guard.send(Message::Text(json)).await;
    }
}
