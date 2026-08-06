//! The TurnGate — ordering, attribution, and audio gating.
//!
//! Extracted from `services/stt/sarvam.rs`, where it was developed. It is
//! provider-agnostic and now applies to every STT service built on
//! [`SttService`](super::driver::SttService).
//!
//! ── Ordering & attribution invariants ──────────────────────────────────────
//!
//! 1. AUDIO GATING (fatherhood by construction).
//!    Audio is forwarded to the provider only while the local VAD says the
//!    user is speaking, plus a pre-roll buffer (the ~pre_roll_ms of audio
//!    captured while the VAD was still confirming speech) and the denoiser
//!    tail. Between turns nothing is sent, so the provider's server-side VAD
//!    has no noise to hallucinate from: every transcript it can possibly
//!    return descends from a local-VAD-attested turn. Spurious "fatherless"
//!    transcripts are impossible by construction, not by bookkeeping.
//!
//! 2. STOP-FRAME GATING (transcript bundled onto the released stop).
//!    VADUserStoppedSpeaking is never forwarded directly. It is stashed in
//!    the gate; the WebSocket receive task releases it downstream immediately
//!    AFTER pushing the transcript it was waiting for. The transcript is
//!    bundled onto the stop frame itself so that a single frame carries both
//!    the turn boundary and the text. Separate frames cannot guarantee order
//!    across the system/data lanes, so one frame is the only structural fix.
//!
//! 3. EMISSION LINEARIZATION (the `emit` mutex).
//!    Turn frames (VadStart, Transcription, released VadStop) can originate
//!    on two tasks: the pipeline task (VadStart passthrough) and the WS
//!    receive task (transcript + released stop). All such emissions acquire
//!    the gate's async `emit` lock, and the claim/drop of the pending stop
//!    happens inside the same critical section. Every interleaving therefore
//!    collapses to one of two valid serial orders:
//!    …Transcript, VadStop, VadStart…   (turn flushed, then new turn), or
//!    …VadStart[pending dropped], Transcript…  (barge-in merge).
//!    The premature-flush and lost-text interleavings cannot occur.
//!
//! 4. EXACTLY-ONCE RELEASE (atomic claim).
//!    The stashed stop lives in a Mutex<Option<Frame>>; `take()` is the CAS.
//!    Three parties race for it — transcript arrival (release), timeout
//!    (release), barge-in VadStart (drop). Exactly one wins. Timeouts are
//!    additionally generation-guarded so a stale timer can never steal a
//!    newer turn's pending stop.
//!
//! 5. DURATION LEDGER (turn attribution + billing).
//!    The gate keeps a FIFO of (epoch, ms-of-audio-sent). Providers that
//!    report a per-transcript audio duration (Sarvam's
//!    `metrics.audio_duration`) let us identify which turn fathered a
//!    transcript: since the server consumes the stream in order, consuming
//!    that duration from the ledger head names the epoch, with a small
//!    tolerance for server-side silence trimming. The server-reported
//!    duration is also the billing source of truth. Providers that report
//!    nothing fall through to consuming the oldest closed turn whole.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::error::Result;
use crate::frames::{Frame, FrameDirection, FrameProcessor, TranscriptionData};

/// Server-side VAD may trim leading/trailing silence, so the audio duration
/// reported per transcript can be slightly less than what we sent. Ledger
/// consumption treats remainders at or below this as fully consumed.
pub(crate) const LEDGER_TOLERANCE_MS: f64 = 120.0;

pub(crate) struct LedgerEntry {
    pub(crate) epoch: u64,
    pub(crate) ms: f64,
}

pub(crate) struct GateInner {
    /// Turn counter. Bumped on every VADUserStartedSpeaking.
    pub(crate) epoch: u64,
    /// Local VAD state, as seen by this gate.
    pub(crate) speaking: bool,
    /// The stashed VADUserStoppedSpeaking frame, waiting for its transcript.
    pub(crate) pending_stop: Option<Frame>,
    pub(crate) pending_epoch: u64,
    /// Generation counter guarding release timeouts. Bumped on every stash,
    /// claim, and drop, so a stale timer always finds a mismatch.
    pub(crate) timeout_gen: u64,
    /// Ring buffer of recent denoised samples captured while NOT speaking;
    /// drained and sent as pre-roll on VadStart.
    pub(crate) pre_roll: VecDeque<i16>,
    pub(crate) pre_roll_cap: usize,
    /// FIFO of (epoch, ms of audio sent) for closed turns.
    pub(crate) ledger: VecDeque<LedgerEntry>,
    /// Audio ms sent so far for the currently open turn.
    pub(crate) current_sent_ms: f64,
}

/// Outcome of processing one final-transcript event through the gate.
pub struct TranscriptOutcome {
    pub released_stop: bool,
    pub father_epoch: Option<u64>,
    pub billed_ms: f64,
}

pub struct TurnGate {
    /// Linearizes downstream emission of turn frames (VadStart, Transcript,
    /// released VadStop) across the pipeline task and the WS receive task.
    /// The pending-stop claim/drop always happens inside this critical
    /// section, which is what makes the emission order provably serial.
    emit: Mutex<()>,
    pub(crate) inner: std::sync::Mutex<GateInner>,
    sample_rate: u32,
}

impl TurnGate {
    pub fn new(sample_rate: u32, pre_roll_ms: u32) -> Arc<Self> {
        let cap = (sample_rate as u64 * pre_roll_ms as u64 / 1000) as usize;
        Arc::new(Self {
            emit: Mutex::new(()),
            inner: std::sync::Mutex::new(GateInner {
                epoch: 0,
                speaking: false,
                pending_stop: None,
                pending_epoch: 0,
                timeout_gen: 0,
                pre_roll: VecDeque::with_capacity(cap),
                pre_roll_cap: cap,
                ledger: VecDeque::new(),
                current_sent_ms: 0.0,
            }),
            sample_rate,
        })
    }

    pub fn ms_of(&self, samples: usize) -> f64 {
        samples as f64 * 1000.0 / self.sample_rate as f64
    }

    /// Local VAD start. Atomically drops any pending stop (barge-in: the
    /// user resumed, so that turn boundary is stale), bumps the epoch,
    /// forwards the VadStart frame downstream under the emit lock, and
    /// returns the pre-roll samples the caller should send to the provider.
    pub async fn on_vad_start(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<Vec<i16>> {
        let _emit = self.emit.lock().await;

        let (dropped, epoch, pre_roll) = {
            let mut s = self.inner.lock().unwrap();
            let dropped = s.pending_stop.take().is_some();
            if dropped {
                // Disarm the timer that was guarding the dropped stop.
                s.timeout_gen = s.timeout_gen.wrapping_add(1);
            }
            s.epoch += 1;
            s.speaking = true;
            let pre: Vec<i16> = s.pre_roll.drain(..).collect();
            s.current_sent_ms = self.ms_of(pre.len());
            (dropped, s.epoch, pre)
        };

        if dropped {
            log::info!(
                "TurnGate: barge-in — pending VadStop dropped (atomic take), \
                 turn continues as epoch {}",
                epoch
            );
        } else {
            log::debug!("TurnGate: VadStart — opening epoch {}", epoch);
        }

        processor.push_frame(frame, direction).await?;
        Ok(pre_roll)
    }

    /// Local VAD stop. Finalizes the open turn's ledger entry, stashes the
    /// stop frame (NOT forwarded), and returns the timeout generation the
    /// caller should arm a release timer with. `tail_ms` is the duration of
    /// the denoiser tail the caller already sent for this turn.
    pub fn on_vad_stop(&self, frame: Frame, tail_ms: f64) -> u64 {
        let mut s = self.inner.lock().unwrap();
        if s.pending_stop.is_some() {
            // Should be unreachable (transport CAS prevents double-stop),
            // but never silently lose a frame boundary.
            log::warn!("TurnGate: replacing an unreleased pending VadStop");
        }
        s.speaking = false;
        s.current_sent_ms += tail_ms;
        let epoch = s.epoch;
        let ms = s.current_sent_ms;
        s.ledger.push_back(LedgerEntry { epoch, ms });
        s.current_sent_ms = 0.0;
        s.pending_stop = Some(frame);
        s.pending_epoch = epoch;
        s.timeout_gen = s.timeout_gen.wrapping_add(1);
        log::debug!(
            "TurnGate: VadStop gated for epoch {} ({:.0}ms sent), awaiting transcript",
            epoch, ms
        );
        s.timeout_gen
    }

    /// Audio admission. While speaking: account the duration and tell the
    /// caller to send. While quiet: buffer into the pre-roll ring instead.
    /// When `gated` is false (legacy continuous streaming), always send but
    /// still account the duration against the current epoch.
    pub fn admit_audio(&self, samples: &[i16], gated: bool) -> bool {
        let mut s = self.inner.lock().unwrap();
        if s.speaking || !gated {
            s.current_sent_ms += self.ms_of(samples.len());
            return true;
        }
        for &v in samples {
            if s.pre_roll.len() == s.pre_roll_cap {
                s.pre_roll.pop_front();
            }
            s.pre_roll.push_back(v);
        }
        false
    }

    /// A final transcript arrived on the receive task. If a pending stop is
    /// present, the transcript (if any) is bundled onto it and the single
    /// combined frame is released downstream. If there is no pending stop,
    /// the transcript is pushed as a standalone data frame (mid-turn partial).
    ///
    /// `server_ms` is the provider-reported audio duration in milliseconds
    /// when available; it drives ledger attribution and is the preferred
    /// billing duration.
    pub async fn on_transcript(
        &self,
        processor: &FrameProcessor,
        data: Option<TranscriptionData>,
        server_ms: Option<f64>,
    ) -> Result<TranscriptOutcome> {
        let _emit = self.emit.lock().await;

        let (stop, father, consumed_ms) = {
            let mut s = self.inner.lock().unwrap();
            let stop = s.pending_stop.take();
            if stop.is_some() {
                // Disarm the release timer for this stash.
                s.timeout_gen = s.timeout_gen.wrapping_add(1);
            }
            let (father, consumed) = consume_ledger(&mut s, server_ms);
            (stop, father, consumed)
        };

        let released = stop.is_some();
        match stop {
            Some(stop_frame) => {
                // Closing transcript rides the stop — single frame, no lane race.
                let stop_frame = match data {
                    Some(td) => stop_frame.with_vad_stop_transcript(td),
                    None => stop_frame, // empty answer still closes the turn
                };
                log::debug!(
                    "TurnGate: releasing VadStop (+transcript) for epoch {:?}",
                    father
                );
                processor.push_frame(stop_frame, FrameDirection::Downstream).await?;
            }
            None => {
                // Mid-turn transcript: turn still open, standalone data frame is fine.
                if let Some(td) = data {
                    processor
                        .push_frame(Frame::transcription(td), FrameDirection::Downstream)
                        .await?;
                }
            }
        }

        Ok(TranscriptOutcome {
            released_stop: released,
            father_epoch: father,
            billed_ms: server_ms.unwrap_or(consumed_ms),
        })
    }

    /// Timeout fallback: release the pending stop if — and only if — it is
    /// still the same stash this timer was armed for (generation guard).
    pub async fn release_pending_after(
        self: Arc<Self>,
        processor: FrameProcessor,
        gen: u64,
        after: Duration,
    ) {
        tokio::time::sleep(after).await;
        let _emit = self.emit.lock().await;

        let stop = {
            let mut s = self.inner.lock().unwrap();
            if s.timeout_gen == gen {
                s.pending_stop.take()
            } else {
                None
            }
        };

        match stop {
            Some(frame) => {
                log::warn!("TurnGate: no transcript within timeout — releasing VadStop anyway");
                let _ = processor.push_frame(frame, FrameDirection::Downstream).await;
            }
            None => {
                log::debug!("TurnGate: release timer fired but lost the race — no-op");
            }
        }
    }

    /// Full reset on End/Cancel/Start. Any pending stop is dropped (the
    /// pipeline is going away or restarting; releasing it would be noise).
    pub fn reset(&self) {
        let mut s = self.inner.lock().unwrap();
        s.pending_stop = None;
        s.timeout_gen = s.timeout_gen.wrapping_add(1);
        s.speaking = false;
        s.pre_roll.clear();
        s.ledger.clear();
        s.current_sent_ms = 0.0;
    }
}

/// Consume `server_ms` of audio from the front of the ledger, returning the
/// epoch the consumption ends in (the transcript's father) and the ms
/// consumed. With `server_ms == None` (providers without duration metrics),
/// the oldest closed turn is consumed whole; if none exists, the open turn is.
pub(crate) fn consume_ledger(s: &mut GateInner, server_ms: Option<f64>) -> (Option<u64>, f64) {
    match server_ms {
        Some(ms) => {
            let mut remaining = ms;
            let mut father = None;
            while remaining > LEDGER_TOLERANCE_MS {
                match s.ledger.front_mut() {
                    Some(e) if e.ms > 0.0 => {
                        let take = e.ms.min(remaining);
                        e.ms -= take;
                        remaining -= take;
                        father = Some(e.epoch);
                        if e.ms <= LEDGER_TOLERANCE_MS {
                            s.ledger.pop_front();
                        }
                    }
                    Some(_) => {
                        s.ledger.pop_front();
                    }
                    None => {
                        // Mid-turn transcript: the server finalized part of
                        // the still-open turn. Charge the open epoch.
                        // (`remaining` is not decremented — we break out here,
                        // and the reported `ms` is what gets billed.)
                        if s.speaking {
                            let take = s.current_sent_ms.min(remaining);
                            s.current_sent_ms -= take;
                            father = Some(s.epoch);
                        }
                        break;
                    }
                }
            }
            (father, ms)
        }
        None => {
            if let Some(e) = s.ledger.pop_front() {
                (Some(e.epoch), e.ms)
            } else {
                let ms = s.current_sent_ms;
                s.current_sent_ms = 0.0;
                (s.speaking.then_some(s.epoch), ms)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_proc() -> FrameProcessor {
        FrameProcessor::new("test", Box::new(crate::frames::PassthroughHandler), false)
    }

    fn stop_frame() -> Frame {
        Frame::vad_user_stopped_speaking(0.0, 0.0)
    }

    fn start_frame() -> Frame {
        Frame::vad_user_started_speaking(0.0, 0.0)
    }

    // ---- ms_of ---------------------------------------------------------------

    #[test]
    fn ms_of_16khz_one_second() {
        let gate = TurnGate::new(16_000, 500);
        assert!((gate.ms_of(16_000) - 1000.0).abs() < 0.001);
    }

    #[test]
    fn ms_of_8khz_half_second() {
        let gate = TurnGate::new(8_000, 500);
        assert!((gate.ms_of(4_000) - 500.0).abs() < 0.001);
    }

    // ---- pre-roll ring -------------------------------------------------------

    #[test]
    fn pre_roll_ring_is_capped() {
        // 16kHz × 100ms cap = 1600 samples
        let gate = TurnGate::new(16_000, 100);
        let chunk = vec![1i16; 800];
        for _ in 0..5 {
            let sent = gate.admit_audio(&chunk, true);
            assert!(!sent, "quiet audio must be buffered, not sent");
        }
        let s = gate.inner.lock().unwrap();
        assert_eq!(s.pre_roll.len(), 1600, "ring must be capped at pre_roll_cap");
    }

    #[test]
    fn admit_audio_sends_while_speaking_and_accounts_duration() {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            s.speaking = true;
        }
        let chunk = vec![0i16; 16_000]; // 1000ms
        assert!(gate.admit_audio(&chunk, true));
        let s = gate.inner.lock().unwrap();
        assert!((s.current_sent_ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn admit_audio_ungated_always_sends() {
        let gate = TurnGate::new(16_000, 100);
        let chunk = vec![0i16; 1600];
        assert!(gate.admit_audio(&chunk, false), "ungated mode must always send");
    }

    // ---- stash / claim / drop -------------------------------------------------

    #[tokio::test]
    async fn vad_start_drops_pending_stop_and_bumps_epoch() {
        let gate = TurnGate::new(16_000, 100);
        let gen = gate.on_vad_stop(stop_frame(), 0.0);
        assert!(gate.inner.lock().unwrap().pending_stop.is_some());

        let pre = gate
            .on_vad_start(&dummy_proc(), start_frame(), FrameDirection::Downstream)
            .await
            .unwrap();
        assert!(pre.is_empty());

        let s = gate.inner.lock().unwrap();
        assert!(s.pending_stop.is_none(), "barge-in must drop the pending stop");
        assert_eq!(s.epoch, 1);
        assert_ne!(s.timeout_gen, gen, "drop must disarm the release timer");
    }

    #[tokio::test]
    async fn transcript_claims_pending_stop_exactly_once() {
        let gate = TurnGate::new(16_000, 100);
        gate.on_vad_stop(stop_frame(), 0.0);

        let o1 = gate.on_transcript(&dummy_proc(), None, Some(0.0)).await.unwrap();
        assert!(o1.released_stop, "first claim must win");

        let o2 = gate.on_transcript(&dummy_proc(), None, Some(0.0)).await.unwrap();
        assert!(!o2.released_stop, "second claim must find nothing");
    }

    #[tokio::test]
    async fn stale_timeout_generation_cannot_steal_newer_stash() {
        let gate = TurnGate::new(16_000, 100);
        let old_gen = gate.on_vad_stop(stop_frame(), 0.0);

        // Transcript claims it (and bumps the generation)...
        let o = gate.on_transcript(&dummy_proc(), None, None).await.unwrap();
        assert!(o.released_stop);

        // ...then a NEW turn is stashed.
        gate.on_vad_start(&dummy_proc(), start_frame(), FrameDirection::Downstream)
            .await
            .unwrap();
        gate.on_vad_stop(stop_frame(), 0.0);

        // The old timer fires with its stale generation: must be a no-op.
        gate.clone()
            .release_pending_after(dummy_proc(), old_gen, Duration::from_millis(0))
            .await;
        assert!(
            gate.inner.lock().unwrap().pending_stop.is_some(),
            "stale timer must not release a newer turn's pending stop"
        );
    }

    #[tokio::test]
    async fn empty_transcript_still_releases_pending_stop() {
        let gate = TurnGate::new(16_000, 100);
        gate.on_vad_stop(stop_frame(), 0.0);
        // data: None models an empty/whitespace transcript from the provider.
        let o = gate.on_transcript(&dummy_proc(), None, Some(40.0)).await.unwrap();
        assert!(o.released_stop, "empty answer must still close the turn");
        assert!(gate.inner.lock().unwrap().pending_stop.is_none());
    }

    // ---- ledger ----------------------------------------------------------------

    fn gate_with_ledger(entries: &[(u64, f64)]) -> Arc<TurnGate> {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            for &(epoch, ms) in entries {
                s.ledger.push_back(LedgerEntry { epoch, ms });
            }
        }
        gate
    }

    #[test]
    fn ledger_exact_consume_attributes_to_single_epoch() {
        let gate = gate_with_ledger(&[(3, 1840.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, billed) = consume_ledger(&mut s, Some(1840.0));
        assert_eq!(father, Some(3));
        assert!((billed - 1840.0).abs() < 0.001);
        assert!(s.ledger.is_empty(), "fully consumed entry must be popped");
    }

    #[test]
    fn ledger_tolerates_server_silence_trim() {
        // Sent 2000ms, server reports 1900ms (trimmed 100ms < tolerance):
        // the 100ms remainder must not linger and pollute the next turn.
        let gate = gate_with_ledger(&[(4, 2000.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1900.0));
        assert_eq!(father, Some(4));
        assert!(s.ledger.is_empty(), "sub-tolerance remainder must be dropped");
    }

    #[test]
    fn ledger_spanning_consume_attributes_to_last_epoch_touched() {
        let gate = gate_with_ledger(&[(5, 500.0), (6, 1500.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1000.0));
        assert_eq!(father, Some(6), "consumption ends in epoch 6");
        assert_eq!(s.ledger.len(), 1);
        assert!((s.ledger[0].ms - 1000.0).abs() < 0.001, "epoch 6 keeps its remainder");
    }

    #[test]
    fn ledger_fallback_without_metrics_consumes_oldest_turn_whole() {
        let gate = gate_with_ledger(&[(7, 800.0), (8, 600.0)]);
        let mut s = gate.inner.lock().unwrap();
        let (father, billed) = consume_ledger(&mut s, None);
        assert_eq!(father, Some(7));
        assert!((billed - 800.0).abs() < 0.001);
        assert_eq!(s.ledger.len(), 1);
    }

    #[test]
    fn ledger_mid_turn_transcript_charges_open_epoch() {
        let gate = TurnGate::new(16_000, 100);
        {
            let mut s = gate.inner.lock().unwrap();
            s.speaking = true;
            s.epoch = 9;
            s.current_sent_ms = 3000.0;
        }
        let mut s = gate.inner.lock().unwrap();
        let (father, _) = consume_ledger(&mut s, Some(1200.0));
        assert_eq!(father, Some(9));
        assert!((s.current_sent_ms - 1800.0).abs() < 0.001);
    }
}
