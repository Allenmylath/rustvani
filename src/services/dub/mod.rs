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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::mpsc;

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, DubbedSpeechData, Frame, FrameDirection, FrameHandler, FrameInner,
    FrameProcessor, SystemFrame, TranslationData,
};
use crate::services::translate::TranslationTargets;
use crate::services::tts::synth::TtsSynthesizer;

/// Decides the pace of an utterance (1.0 is natural); the service clamps it to the provider's range.
pub type PaceHint = Arc<dyn Fn(&TranslationData) -> f32 + Send + Sync>;

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
        if translation.text.trim().is_empty() || !self.targets.get().iter().any(|t| *t == translation.target_language) {
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
        let synth = Arc::clone(&self.synth);
        let voice = self.config.voice_for(&lang).to_string();
        let sample_rate = self.config.sample_rate;
        let pace_hint = Arc::clone(&self.pace);
        let billing = self.billing.clone();
        let (pace_min, pace_max) = synth.pace_range();
        tokio::spawn(async move {
            while let Some(translation) = translations.recv().await {
                let pace = pace_hint(&translation).clamp(pace_min, pace_max);
                let text = translation.text.trim();
                match synth.synthesize(text, &lang, &voice, pace, sample_rate).await {
                    Ok(pcm) if pcm.samples.is_empty() => {}
                    Ok(pcm) => {
                        if let Some(bc) = &billing {
                            bc.record(BillingEvent::TtsUsage {
                                session_id: bc.session_id(),
                                provider: synth.name().to_string(),
                                voice: voice.clone(),
                                language: Some(lang.clone()),
                                char_count: text.chars().count(),
                                occurred_at: Utc::now(),
                            });
                        }
                        let data = DubbedSpeechData::new(&translation, &voice, pace, pcm);
                        if let Err(e) = processor.push_frame(Frame::dubbed_speech(data), FrameDirection::Downstream).await {
                            log::warn!("Dubbing: could not push dubbed speech ({lang}): {e}");
                        }
                    }
                    Err(e) => log::warn!("Dubbing ({lang}): {e}"),
                }
            }
        });
        queue
    }

    fn stop_workers(&self) {
        self.workers.lock().unwrap().clear();
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
        let mut service = DubbingService::new(Arc::new(Tone), targets, config);
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
