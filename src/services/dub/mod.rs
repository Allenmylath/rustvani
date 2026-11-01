//! Dubbing: translated transcripts spoken in their language, as a pipeline stage.
//!
//! ```text
//!   STT ─▶ TranslationService ─▶ DubbingService ─▶ …
//!                                TranslationFrame (target in the dub targets) ──▶ TTS ──▶ DubbedSpeechFrame
//! ```
//!
//! [`DubbingService`] sits after the [`TranslationService`](crate::services::translate::TranslationService).
//! Every frame passes through unchanged; each `TranslationFrame` whose target language is among
//! the current [`TranslationTargets`] is spoken by a [`TtsSynthesizer`] and comes out as one
//! `DubbedSpeechFrame` ([`DubbedSpeechData`]): the whole utterance's PCM, with the source
//! transcript's `timestamp` and `source_text` so a consumer can place it where the words were
//! said. Synthesis runs off the pipeline's tasks, one ordered queue per language.
//!
//! The pace of each utterance comes from a [`PaceHint`] the host supplies (a dubbing host that
//! fits speech into the time the original took), clamped to the provider's range.
//!
//! With a synthesizer that streams ([`TtsSynthesizer::streams`]), the frame goes out as soon as
//! the first chunk of audio is there, carrying that chunk, and the rest follows on the frame's
//! [`SpeechStream`]; a host can start playing a few hundred milliseconds after the text instead
//! of after the whole utterance.
//!
//! An utterance whose speech could not be made (the voice failed, or gave no audio) is tried once
//! more after [`RETRY_WAIT`] before it is reported `failed`. With a
//! [`SentenceLedger`](crate::services::ledger::SentenceLedger) that `failed` closes the sentence as
//! lost; a translation into a language no longer dubbed closes it as abandoned; and speech handed
//! on moves it to `Playing`, after which **whatever plays the `DubbedSpeechFrame` closes it**
//! (played, failed, or abandoned when nobody listens any more).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::Notify;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::mpsc;

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, DubEventData, DubEventKind, DubbedSpeechData, Frame, FrameDirection,
    FrameHandler, FrameInner, FrameProcessor, SystemFrame, TranslationData,
};
use crate::services::ledger::{Closed, SentenceKey, SentenceLedger, Stage};
use crate::services::translate::TranslationTargets;
use crate::services::tts::synth::{Pcm, TtsSynthesizer};

/// How long after a failed synthesis an utterance is tried again.
pub const RETRY_WAIT: Duration = Duration::from_millis(500);

/// Decides the pace of an utterance (1.0 is natural); the service clamps it to the provider's range.
pub type PaceHint = Arc<dyn Fn(&TranslationData) -> f32 + Send + Sync>;

/// The rest of a streamed utterance's audio, after the first chunk on its frame: chunks of PCM
/// in the frame's format, appended as they arrive, until [`finish`](Self::finish) (or
/// [`fail`](Self::fail), after which nothing more comes).
#[derive(Debug, Default)]
pub struct SpeechStream {
    chunks: Mutex<VecDeque<Bytes>>,
    done: AtomicBool,
    failed: AtomicBool,
    notify: Notify,
}

impl SpeechStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, chunk: Bytes) {
        if chunk.is_empty() {
            return;
        }
        self.chunks.lock().unwrap().push_back(chunk);
        self.notify.notify_waiters();
    }

    pub fn finish(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Ended early: the synthesis failed part way.
    pub fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        self.finish();
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// The next chunk, waiting for it; `None` once the utterance is over.
    pub async fn next(&self) -> Option<Bytes> {
        loop {
            let notified = self.notify.notified();
            if let Some(chunk) = self.chunks.lock().unwrap().pop_front() {
                return Some(chunk);
            }
            if self.is_done() {
                return None;
            }
            notified.await;
        }
    }

    /// Bytes waiting to be taken.
    pub fn buffered(&self) -> usize {
        self.chunks.lock().unwrap().iter().map(Bytes::len).sum()
    }
}

#[derive(Clone)]
pub struct DubConfig {
    /// Voice per language tag (as the targets name them); `default_voice` otherwise; empty means
    /// the provider's default.
    pub voices: HashMap<String, String>,
    pub default_voice: String,
    /// Sample rate asked of the provider.
    pub sample_rate: u32,
}

impl Default for DubConfig {
    fn default() -> Self {
        Self { voices: HashMap::new(), default_voice: String::new(), sample_rate: 48_000 }
    }
}

impl DubConfig {
    pub fn voice_for(&self, lang: &str) -> &str {
        self.voices.get(lang).map(String::as_str).unwrap_or(&self.default_voice)
    }
}

/// The frame handler: forwards everything, speaks translations into the dub targets.
pub struct DubbingService {
    synth: Arc<dyn TtsSynthesizer>,
    targets: TranslationTargets,
    config: DubConfig,
    pace: PaceHint,
    billing: Option<Arc<dyn BillingCollector>>,
    ledger: Option<Arc<SentenceLedger>>,
    retry_wait: Duration,
    workers: Mutex<HashMap<String, mpsc::UnboundedSender<TranslationData>>>,
}

impl DubbingService {
    pub fn new(synth: Arc<dyn TtsSynthesizer>, targets: TranslationTargets, config: DubConfig) -> Self {
        Self {
            synth,
            targets,
            config,
            pace: Arc::new(|_| 1.0),
            billing: None,
            ledger: None,
            retry_wait: RETRY_WAIT,
            workers: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_pace_hint(mut self, pace: PaceHint) -> Self {
        self.pace = pace;
        self
    }

    /// Report a [`BillingEvent::TtsUsage`] per utterance.
    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    /// Close the sentences that cannot be spoken in `ledger`, and move the spoken ones to
    /// `Playing` (see the module docs).
    pub fn with_ledger(mut self, ledger: Arc<SentenceLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// How long after a failed synthesis it is tried again ([`RETRY_WAIT`] by default).
    pub fn with_retry_wait(mut self, wait: Duration) -> Self {
        self.retry_wait = wait;
        self
    }

    pub fn targets(&self) -> TranslationTargets {
        self.targets.clone()
    }

    pub fn synthesizer(&self) -> &Arc<dyn TtsSynthesizer> {
        &self.synth
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("Dubbing", Box::new(self), false)
    }

    fn submit(&self, processor: &FrameProcessor, translation: &TranslationData) {
        if !self.targets.get().iter().any(|t| *t == translation.target_language) {
            if let Some(ledger) = &self.ledger {
                ledger.close(&SentenceKey::of_translation(translation), Closed::Abandoned("not dubbed".into()));
            }
            return;
        }
        if translation.text.trim().is_empty() {
            if let Some(ledger) = &self.ledger {
                ledger.close(&SentenceKey::of_translation(translation), Closed::Failed("empty translation".into()));
            }
            return;
        }
        let mut workers = self.workers.lock().unwrap();
        let queue = workers
            .entry(translation.target_language.clone())
            .or_insert_with(|| self.worker(processor.clone(), translation.target_language.clone()));
        let _ = queue.send(translation.clone());
    }

    fn worker(&self, processor: FrameProcessor, lang: String) -> mpsc::UnboundedSender<TranslationData> {
        let (queue, mut translations) = mpsc::unbounded_channel::<TranslationData>();
        let speaker = Speaker {
            synth: Arc::clone(&self.synth),
            voice: self.config.voice_for(&lang).to_string(),
            sample_rate: self.config.sample_rate,
            billing: self.billing.clone(),
            ledger: self.ledger.clone(),
            processor,
            lang,
        };
        let pace_hint = Arc::clone(&self.pace);
        let retry_wait = self.retry_wait;
        let (pace_min, pace_max) = speaker.synth.pace_range();
        tokio::spawn(async move {
            while let Some(translation) = translations.recv().await {
                let received = Instant::now();
                let pace = pace_hint(&translation).clamp(pace_min, pace_max);
                let error = match speaker.speak(&translation, pace, received).await {
                    Ok(()) => continue,
                    Err(e) => e,
                };
                log::warn!("Dubbing ({}): {error}; once more", speaker.lang);
                tokio::time::sleep(retry_wait).await;
                let error = match speaker.speak(&translation, pace, received).await {
                    Ok(()) => continue,
                    Err(e) => e,
                };
                log::warn!("Dubbing ({}): {error}", speaker.lang);
                speaker.tell(DubEventData::new(DubEventKind::Failed, &translation).failed(error.clone())).await;
                if let Some(ledger) = &speaker.ledger {
                    ledger.close(&SentenceKey::of_translation(&translation), Closed::Failed(format!("speech: {error}")));
                }
            }
        });
        queue
    }

    fn stop_workers(&self) {
        self.workers.lock().unwrap().clear();
    }
}

/// One language's voice, speaking its utterances for the worker.
struct Speaker {
    synth: Arc<dyn TtsSynthesizer>,
    voice: String,
    sample_rate: u32,
    billing: Option<Arc<dyn BillingCollector>>,
    ledger: Option<Arc<SentenceLedger>>,
    processor: FrameProcessor,
    lang: String,
}

impl Speaker {
    /// What became of an utterance goes downstream as a `DubEventFrame` (RAVI `user-dub`).
    async fn tell(&self, data: DubEventData) {
        let _ = self.processor.push_frame(Frame::dub_event(data), FrameDirection::Downstream).await;
    }

    /// Speaks `translation`: its frame pushed, with the rest of a streamed one following on its
    /// stream. An error when nothing was pushed (the voice failed or gave no audio): the caller
    /// may try again. Speech that stops part way is reported here, `failed`, and its stream fails.
    async fn speak(&self, translation: &TranslationData, pace: f32, received: Instant) -> std::result::Result<(), String> {
        let text = translation.text.trim();
        let lang = self.lang.as_str();
        if self.synth.streams() {
            let mut chunks = self
                .synth
                .synthesize_streaming(text, lang, &self.voice, pace, self.sample_rate)
                .await
                .map_err(|e| e.to_string())?;
            // The frame goes out with the first chunk; the rest follows on its stream.
            let first = loop {
                match chunks.chunks.recv().await {
                    Some(Ok(b)) if b.is_empty() => continue,
                    Some(Ok(b)) => break b,
                    Some(Err(e)) => return Err(e.to_string()),
                    None => return Err("no audio".to_string()),
                }
            };
            let stream = Arc::new(SpeechStream::new());
            let pcm = Pcm { samples: first, rate: chunks.rate, channels: chunks.channels };
            let data = DubbedSpeechData::new(translation, &self.voice, pace, pcm).with_stream(Arc::clone(&stream));
            self.hand_over(translation, data, received).await;
            while let Some(chunk) = chunks.chunks.recv().await {
                match chunk {
                    Ok(b) => stream.push(b),
                    Err(e) => {
                        log::warn!("Dubbing ({lang}): {e}");
                        stream.fail();
                        self.tell(DubEventData::new(DubEventKind::Failed, translation).failed(e.to_string())).await;
                        break;
                    }
                }
            }
            stream.finish();
            return Ok(());
        }
        let pcm = self
            .synth
            .synthesize(text, lang, &self.voice, pace, self.sample_rate)
            .await
            .map_err(|e| e.to_string())?;
        if pcm.samples.is_empty() {
            return Err("no audio".to_string());
        }
        let data = DubbedSpeechData::new(translation, &self.voice, pace, pcm);
        self.hand_over(translation, data, received).await;
        Ok(())
    }

    /// Bills the utterance, reports it `synthesized`, and pushes its speech (it is `Playing` now).
    async fn hand_over(&self, translation: &TranslationData, data: DubbedSpeechData, received: Instant) {
        if let Some(bc) = &self.billing {
            bc.record(BillingEvent::TtsUsage {
                session_id: bc.session_id(),
                provider: self.synth.name().to_string(),
                voice: self.voice.clone(),
                language: Some(self.lang.clone()),
                char_count: translation.text.trim().chars().count(),
                occurred_at: Utc::now(),
            });
        }
        let mut synthesized = DubEventData::new(DubEventKind::Synthesized, translation);
        synthesized.ready_ms = Some(received.elapsed().as_millis() as u64);
        self.tell(synthesized).await;
        if let Some(ledger) = &self.ledger {
            ledger.advance(&SentenceKey::of_translation(translation), Stage::Playing);
        }
        if let Err(e) = self.processor.push_frame(Frame::dubbed_speech(data), FrameDirection::Downstream).await {
            log::warn!("Dubbing: could not push dubbed speech ({}): {e}", self.lang);
        }
    }
}

#[async_trait]
impl FrameHandler for DubbingService {
    async fn on_process_frame(&self, processor: &FrameProcessor, frame: Frame, direction: FrameDirection) -> Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::Translation(t)) if t.finalized => self.submit(processor, t),
            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. })
            | FrameInner::System(SystemFrame::Stop { .. }) => self.stop_workers(),
            _ => {}
        }
        processor.push_frame(frame, direction).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PipecatError;
    use crate::frames::{PassthroughHandler, StartFrameData, TranscriptionData};
    use crate::services::tts::synth::Pcm;
    use bytes::Bytes;
    use std::time::Duration;

    /// Like `Tone`, in three chunks a while apart.
    struct Trickle;

    #[async_trait]
    impl TtsSynthesizer for Trickle {
        fn name(&self) -> &'static str { "trickle" }
        fn model(&self) -> String { "trickle:v1".into() }
        fn supports(&self, _lang: &str) -> bool { true }
        fn streams(&self) -> bool { true }
        async fn synthesize(&self, text: &str, _lang: &str, _voice: &str, _pace: f32, rate: u32) -> Result<Pcm> {
            Ok(Pcm { samples: Bytes::from(vec![0u8; text.len() * 2]), rate, channels: 1 })
        }
        async fn synthesize_streaming(&self, text: &str, _lang: &str, _voice: &str, _pace: f32, rate: u32) -> Result<crate::services::tts::synth::PcmChunks> {
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            let n = text.len();
            tokio::spawn(async move {
                for i in 1..=3u8 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    let _ = tx.send(Ok(Bytes::from(vec![i; n * 2]))).await;
                }
            });
            Ok(crate::services::tts::synth::PcmChunks { rate, channels: 1, chunks: rx })
        }
    }

    #[tokio::test]
    async fn a_streaming_voice_sends_the_frame_on_the_first_chunk_and_the_rest_after() {
        let service = DubbingService::new(Arc::new(Trickle), TranslationTargets::new(["hi"]), DubConfig::default());
        let proc = FrameProcessor::new("dub-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::DubbedSpeech(d)) = &frame.inner {
                seen.lock().unwrap().push(d.clone());
            }
        });
        proc.process_frame(Frame::translation(translation("abcd", "hi", "1")), FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        let d = &got[0];
        assert_eq!(&d.audio[..], &[1u8; 8], "the first chunk rides the frame");
        let stream = d.stream.clone().expect("the rest streams");
        assert_eq!(stream.next().await.unwrap(), Bytes::from(vec![2u8; 8]));
        assert_eq!(stream.next().await.unwrap(), Bytes::from(vec![3u8; 8]));
        assert!(stream.next().await.is_none(), "and ends");
        assert!(stream.is_done() && !stream.failed());
    }

    /// 0.1 s of audio per character at pace 1; "slow" takes a while; "fail" errors; no Urdu.
    struct Tone;

    #[async_trait]
    impl TtsSynthesizer for Tone {
        fn name(&self) -> &'static str { "tone" }
        fn model(&self) -> String { "tone:v1".into() }
        fn supports(&self, lang: &str) -> bool { lang != "ur" }
        fn pace_range(&self) -> (f32, f32) { (0.5, 2.0) }
        async fn synthesize(&self, text: &str, _lang: &str, _voice: &str, pace: f32, rate: u32) -> Result<Pcm> {
            if text.contains("slow") {
                tokio::time::sleep(Duration::from_millis(60)).await;
            }
            if text.contains("fail") {
                return Err(PipecatError::pipeline("no"));
            }
            let frames = (text.chars().count() as f32 * 0.1 / pace * rate as f32) as usize;
            Ok(Pcm { samples: Bytes::from(vec![0u8; frames * 2]), rate, channels: 1 })
        }
    }

    fn translation(text: &str, lang: &str, ts: &str) -> TranslationData {
        let source = TranscriptionData::new("src", "u", ts).with_language("ml-IN").finalized();
        TranslationData::new(&source, lang, text)
    }

    async fn harness(targets: TranslationTargets, config: DubConfig, pace: Option<PaceHint>) -> (FrameProcessor, Arc<Mutex<Vec<DubbedSpeechData>>>) {
        let mut service = DubbingService::new(Arc::new(Tone), targets, config).with_retry_wait(Duration::from_millis(5));
        if let Some(p) = pace {
            service = service.with_pace_hint(p);
        }
        let proc = FrameProcessor::new("dub-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::DubbedSpeech(d)) = &frame.inner {
                seen.lock().unwrap().push(d.clone());
            }
        });
        let _ = PassthroughHandler;
        (proc, out)
    }

    async fn wait_for(out: &Arc<Mutex<Vec<DubbedSpeechData>>>, n: usize) -> Vec<DubbedSpeechData> {
        for _ in 0..100 {
            if out.lock().unwrap().len() >= n {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        out.lock().unwrap().clone()
    }

    /// Each utterance reports what became of it: `synthesized` with how long the voice took, or
    /// `failed` with why.
    #[tokio::test]
    async fn each_utterance_says_whether_its_speech_was_made() {
        let (proc, _out) = harness(TranslationTargets::new(["hi"]), DubConfig::default(), None).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::DubEvent(e)) = &frame.inner {
                seen.lock().unwrap().push(e.clone());
            }
        });
        for (text, ts) in [("slow words", "1"), ("fail here", "2")] {
            proc.process_frame(Frame::translation(translation(text, "hi", ts)), FrameDirection::Downstream).await.unwrap();
        }
        for _ in 0..100 {
            if events.lock().unwrap().len() >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!((events[0].event, events[0].timestamp.as_str()), (DubEventKind::Synthesized, "1"));
        assert!(events[0].ready_ms.unwrap() >= 50, "{:?}", events[0].ready_ms);
        assert_eq!((events[1].event, events[1].timestamp.as_str()), (DubEventKind::Failed, "2"));
        assert!(events[1].error.is_some());
    }

    #[tokio::test]
    async fn watched_languages_are_spoken_in_order_with_their_source_and_pace() {
        let mut config = DubConfig { default_voice: "neha".into(), sample_rate: 48_000, ..Default::default() };
        config.voices.insert("ta".into(), "kavya".into());
        let pace: PaceHint = Arc::new(|t| if t.text.contains("quick") { 1.5 } else { 1.0 });
        let (proc, out) = harness(TranslationTargets::new(["hi", "ta"]), config, Some(pace)).await;
        for t in [
            translation("slow one", "hi", "1"),
            translation("quick two", "hi", "2"),
            translation("ஒன்று", "ta", "1"),
            translation("unwatched", "en", "1"),
            translation("fail", "hi", "3"),
        ] {
            proc.process_frame(Frame::translation(t), FrameDirection::Downstream).await.unwrap();
        }
        let got = wait_for(&out, 3).await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(out.lock().unwrap().len(), 3, "{got:?}");
        let hi: Vec<&DubbedSpeechData> = got.iter().filter(|d| d.target_language == "hi").collect();
        assert_eq!(hi.len(), 2);
        assert_eq!((hi[0].text.as_str(), hi[0].timestamp.as_str(), hi[0].voice.as_str()), ("slow one", "1", "neha"));
        assert_eq!(hi[0].source_text, "src");
        assert_eq!(hi[0].source_language.as_deref(), Some("ml-IN"));
        assert_eq!((hi[0].pace, hi[1].pace), (1.0, 1.5));
        // 8 chars at 0.1 s each = 0.8 s; "quick two" 9 chars at pace 1.5 = 0.6 s
        assert!((hi[0].pcm().duration_secs() - 0.8).abs() < 0.01, "{}", hi[0].pcm().duration_secs());
        assert!((hi[1].pcm().duration_secs() - 0.6).abs() < 0.01);
        assert_eq!(hi[0].sample_rate, 48_000);
        let ta = got.iter().find(|d| d.target_language == "ta").unwrap();
        assert_eq!(ta.voice, "kavya");
    }

    #[tokio::test]
    async fn targets_change_while_running_and_partials_are_ignored() {
        let targets = TranslationTargets::default();
        let (proc, out) = harness(targets.clone(), DubConfig::default(), None).await;
        proc.process_frame(Frame::translation(translation("nobody", "hi", "1")), FrameDirection::Downstream).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(out.lock().unwrap().is_empty());
        targets.set(["hi"]);
        let mut partial = translation("partial", "hi", "2");
        partial.finalized = false;
        proc.process_frame(Frame::translation(partial), FrameDirection::Downstream).await.unwrap();
        proc.process_frame(Frame::translation(translation("final", "hi", "3")), FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "final");
        assert_eq!(got[0].pace, 1.0, "the default hint");
    }

    /// Fails (or gives no audio, when `empty`) its first `misses` calls, then speaks.
    struct Shaky {
        misses: std::sync::atomic::AtomicUsize,
        empty: bool,
    }

    #[async_trait]
    impl TtsSynthesizer for Shaky {
        fn name(&self) -> &'static str { "shaky" }
        fn model(&self) -> String { "shaky:v1".into() }
        fn supports(&self, _lang: &str) -> bool { true }
        async fn synthesize(&self, _text: &str, _lang: &str, _voice: &str, _pace: f32, rate: u32) -> Result<Pcm> {
            if self.misses.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                self.misses.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if self.empty {
                    return Ok(Pcm { samples: Bytes::new(), rate, channels: 1 });
                }
                return Err(PipecatError::pipeline("connection reset"));
            }
            Ok(Pcm { samples: Bytes::from_static(&[1, 0, 1, 0]), rate, channels: 1 })
        }
    }

    /// Speech and events out of a service on `synth` dubbing Hindi, with a ledger; `translations`
    /// opened in it first.
    async fn speak_all(
        synth: Arc<dyn TtsSynthesizer>,
        translations: &[TranslationData],
    ) -> (Vec<DubbedSpeechData>, Vec<DubEventData>, Arc<SentenceLedger>) {
        let targets = TranslationTargets::new(["hi"]);
        let ledger = Arc::new(SentenceLedger::new(targets.clone()));
        let service = DubbingService::new(synth, targets, DubConfig::default())
            .with_retry_wait(Duration::from_millis(5))
            .with_ledger(Arc::clone(&ledger));
        let proc = FrameProcessor::new("dub-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new((Vec::new(), Vec::new())));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| match &frame.inner {
            FrameInner::Data(DataFrame::DubbedSpeech(d)) => seen.lock().unwrap().0.push(d.clone()),
            FrameInner::Data(DataFrame::DubEvent(e)) => seen.lock().unwrap().1.push(e.clone()),
            _ => {}
        });
        for t in translations {
            let source = TranscriptionData::new(t.source_text.clone(), "u", t.timestamp.clone()).finalized();
            ledger.open(&source, &[t.target_language.clone()]);
            proc.process_frame(Frame::translation(t.clone()), FrameDirection::Downstream).await.unwrap();
        }
        tokio::time::sleep(Duration::from_millis(80)).await;
        let out = out.lock().unwrap();
        (out.0.clone(), out.1.clone(), ledger)
    }

    #[tokio::test]
    async fn a_failed_voice_is_tried_once_more() {
        let synth = Arc::new(Shaky { misses: 1.into(), empty: false });
        let (speech, events, ledger) = speak_all(synth, &[translation("नमस्ते", "hi", "1")]).await;
        assert_eq!(speech.len(), 1, "spoken on the second try");
        assert!(!events.iter().any(|e| e.event == DubEventKind::Failed), "{events:?}");
        let key = SentenceKey::of_speech(&speech[0]);
        assert_eq!(ledger.stage(&key), Some(Stage::Playing), "handed to whatever plays it");
    }

    #[tokio::test]
    async fn no_audio_twice_is_reported_and_closed_as_lost() {
        let synth = Arc::new(Shaky { misses: 2.into(), empty: true });
        let (speech, events, ledger) = speak_all(synth, &[translation("नमस्ते", "hi", "1")]).await;
        assert!(speech.is_empty());
        let failed: Vec<&DubEventData> = events.iter().filter(|e| e.event == DubEventKind::Failed).collect();
        assert_eq!(failed.len(), 1, "no longer silent: {events:?}");
        assert_eq!(failed[0].error.as_deref(), Some("no audio"));
        assert_eq!(ledger.counts()["hi"].failed, 1);
        assert!(ledger.open_counts().is_empty());
    }

    #[tokio::test]
    async fn a_language_no_longer_dubbed_is_closed_as_abandoned() {
        let targets = TranslationTargets::new(["hi"]);
        let ledger = Arc::new(SentenceLedger::new(targets.clone()));
        let service = DubbingService::new(Arc::new(Tone), targets.clone(), DubConfig::default()).with_ledger(Arc::clone(&ledger));
        let proc = FrameProcessor::new("dub-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let t = translation("नमस्ते", "hi", "1");
        ledger.open(&TranscriptionData::new(t.source_text.clone(), "u", "1").finalized(), &["hi".into()]);
        targets.set(Vec::<String>::new());
        proc.process_frame(Frame::translation(t), FrameDirection::Downstream).await.unwrap();
        assert_eq!(ledger.counts()["hi"].abandoned, 1);
    }

    #[tokio::test]
    async fn usage_is_billed_per_utterance_with_the_language() {
        struct Record(Mutex<Vec<BillingEvent>>);
        impl BillingCollector for Record {
            fn record(&self, event: BillingEvent) { self.0.lock().unwrap().push(event); }
            fn session_id(&self) -> uuid::Uuid { uuid::Uuid::nil() }
        }
        let billing = Arc::new(Record(Mutex::new(Vec::new())));
        let service = DubbingService::new(Arc::new(Tone), TranslationTargets::new(["hi"]), DubConfig::default())
            .with_billing(billing.clone());
        let proc = FrameProcessor::new("dub-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        proc.process_frame(Frame::translation(translation("  नमस्ते  ", "hi", "1")), FrameDirection::Downstream).await.unwrap();
        for _ in 0..100 {
            if !billing.0.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let events = billing.0.lock().unwrap();
        match &events[0] {
            BillingEvent::TtsUsage { provider, language, char_count, .. } => {
                assert_eq!(provider, "tone");
                assert_eq!(language.as_deref(), Some("hi"));
                assert_eq!(*char_count, "नमस्ते".chars().count());
            }
            other => panic!("{other:?}"),
        }
    }
}
