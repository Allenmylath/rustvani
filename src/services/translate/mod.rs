//! Translation: final transcripts into other languages, as a pipeline stage.
//!
//! ```text
//!   STT ─▶ TranslationService ─▶ …
//!          TranscriptionFrame (final) ──▶ provider ──▶ TranslationFrame per target language
//! ```
//!
//! [`TranslationService`] sits after the STT. Every frame passes through it unchanged; each
//! **final** transcript (a `TranscriptionFrame` with `finalized`, or the turn-closing transcript
//! the STT's turn gate bundles onto `VADUserStoppedSpeaking`) is also handed to a
//! [`TranslationProvider`] once per target language, and the result goes downstream as a
//! `TranslationFrame` ([`TranslationData`]). Interim transcripts are never translated: they change
//! too often to be worth a call.
//!
//! Translation calls run off the pipeline's tasks, one ordered queue per target language, so a
//! slow provider never holds up the transcript or another language, and a language's translations
//! come out in the order their transcripts did.
//!
//! The target languages are a [`TranslationTargets`] handle the host may change at any time (a
//! captioning product translates only the languages someone is watching). A transcript whose
//! language is already the target is passed through as its own translation, so a target is never
//! silent.
//!
//! Downstream, [`RaviObserver`](crate::ravi::RaviObserver) sends each `TranslationFrame` to the
//! client as a `user-translation` message, and the service reports
//! [`BillingEvent::TranslationUsage`] per call when given a billing collector.
//!
//! Providers: [`sarvam`] (feature `translate-sarvam`). Any text-to-text backend fits behind
//! [`TranslationProvider`].

#[cfg(feature = "translate-sarvam")]
pub mod sarvam;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::mpsc;

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData, TranslationData,
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// What a provider returns for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translated {
    pub text: String,
    /// The source language the provider used or detected, when it says.
    pub source_language: Option<String>,
}

/// A text-to-text translation backend.
///
/// Language tags are BCP-47 as the caller names them (`hi`, `en-IN`, `pt-BR`); a provider maps
/// them to its own codes and says which it can serve through [`supports`](Self::supports).
#[async_trait]
pub trait TranslationProvider: Send + Sync + 'static {
    /// Provider name for billing and logs (`"sarvam"`).
    fn name(&self) -> &'static str;

    /// Model name for billing.
    fn model(&self) -> String;

    /// Whether `lang` can be a target (and a known source) of this provider.
    fn supports(&self, lang: &str) -> bool;

    /// The longest input, in characters, one call accepts. The service splits longer transcripts at
    /// sentence ends and joins the pieces.
    fn max_chars(&self) -> usize {
        usize::MAX
    }

    /// Translates `text` from `from` (unknown when `None`: detect it) into `to`.
    async fn translate(&self, text: &str, from: Option<&str>, to: &str) -> Result<Translated>;
}

// ---------------------------------------------------------------------------
// Targets
// ---------------------------------------------------------------------------

/// The languages a [`TranslationService`] translates into, changeable while the pipeline runs.
/// Clones share the same set.
#[derive(Clone, Debug, Default)]
pub struct TranslationTargets(Arc<RwLock<Vec<String>>>);

impl TranslationTargets {
    pub fn new<L: Into<String>>(langs: impl IntoIterator<Item = L>) -> Self {
        let targets = Self::default();
        targets.set(langs);
        targets
    }

    /// Replaces the set. Duplicates are dropped; order is kept.
    pub fn set<L: Into<String>>(&self, langs: impl IntoIterator<Item = L>) {
        let mut seen = Vec::new();
        for lang in langs {
            let lang = lang.into();
            if !lang.is_empty() && !seen.contains(&lang) {
                seen.push(lang);
            }
        }
        *self.0.write().unwrap() = seen;
    }

    pub fn get(&self) -> Vec<String> {
        self.0.read().unwrap().clone()
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().unwrap().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// The frame handler: forwards everything, translates final transcripts. See the module docs.
pub struct TranslationService {
    provider: Arc<dyn TranslationProvider>,
    targets: TranslationTargets,
    billing: Option<Arc<dyn BillingCollector>>,
    /// The source language when a transcript names none.
    default_source: Option<String>,
    /// One queue per target language, made on first use.
    workers: Mutex<HashMap<String, mpsc::UnboundedSender<TranscriptionData>>>,
    /// The last transcript submitted: a transcript seen twice is translated once.
    last: Mutex<Option<(String, String)>>,
}

impl TranslationService {
    /// Translate into `targets` with `provider`.
    pub fn new(provider: impl TranslationProvider, targets: TranslationTargets) -> Self {
        Self::with_provider(Arc::new(provider), targets)
    }

    pub fn with_provider(provider: Arc<dyn TranslationProvider>, targets: TranslationTargets) -> Self {
        Self {
            provider,
            targets,
            billing: None,
            default_source: None,
            workers: Mutex::new(HashMap::new()),
            last: Mutex::new(None),
        }
    }

    /// Report a [`BillingEvent::TranslationUsage`] per call.
    pub fn with_billing(mut self, billing: Arc<dyn BillingCollector>) -> Self {
        self.billing = Some(billing);
        self
    }

    /// The language to translate from when a transcript carries none (an STT configured for one
    /// language reports none per phrase). Without it the provider detects the language.
    pub fn with_default_source_language(mut self, lang: impl Into<String>) -> Self {
        self.default_source = Some(lang.into()).filter(|l| !l.is_empty());
        self
    }

    /// The handle the targets can be changed through.
    pub fn targets(&self) -> TranslationTargets {
        self.targets.clone()
    }

    pub fn provider(&self) -> &Arc<dyn TranslationProvider> {
        &self.provider
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("Translation", Box::new(self), false)
    }

    /// Queues a final transcript for every current target.
    fn submit(&self, processor: &FrameProcessor, transcript: &TranscriptionData) {
        let text = transcript.text.trim();
        if text.is_empty() {
            return;
        }
        {
            let key = (transcript.timestamp.clone(), transcript.text.clone());
            let mut last = self.last.lock().unwrap();
            if last.as_ref() == Some(&key) {
                return;
            }
            *last = Some(key);
        }
        let targets = self.targets.get();
        if targets.is_empty() {
            return;
        }
        let mut workers = self.workers.lock().unwrap();
        for target in targets {
            let queue = workers
                .entry(target.clone())
                .or_insert_with(|| self.worker(processor.clone(), target));
            let _ = queue.send(transcript.clone());
        }
    }

    /// Translates one language's transcripts in order, until the queue is dropped.
    fn worker(&self, processor: FrameProcessor, target: String) -> mpsc::UnboundedSender<TranscriptionData> {
        let (queue, mut transcripts) = mpsc::unbounded_channel::<TranscriptionData>();
        let provider = Arc::clone(&self.provider);
        let billing = self.billing.clone();
        let default_source = self.default_source.clone();
        tokio::spawn(async move {
            while let Some(mut transcript) = transcripts.recv().await {
                if transcript.language.is_none() {
                    transcript.language = default_source.clone();
                }
                let translation = translate_one(provider.as_ref(), &transcript, &target).await;
                let Some(translation) = translation else { continue };
                if let Some(bc) = &billing {
                    bc.record(BillingEvent::TranslationUsage {
                        session_id: bc.session_id(),
                        provider: provider.name().to_string(),
                        model: provider.model(),
                        source_language: translation.source_language.clone(),
                        target_language: target.clone(),
                        char_count: transcript.text.trim().chars().count(),
                        occurred_at: Utc::now(),
                    });
                }
                if let Err(e) = processor
                    .push_frame(Frame::translation(translation), FrameDirection::Downstream)
                    .await
                {
                    log::warn!("Translation: could not push a translation ({target}): {e}");
                }
            }
        });
        queue
    }

    fn stop_workers(&self) {
        self.workers.lock().unwrap().clear();
    }
}

/// One transcript into one language: the provider's answer as a `TranslationData`, or `None` when
/// there is nothing to say (the call failed, or it answered nothing).
async fn translate_one(
    provider: &dyn TranslationProvider,
    transcript: &TranscriptionData,
    target: &str,
) -> Option<TranslationData> {
    let text = transcript.text.trim();
    let from = transcript.language.as_deref();
    if from.is_some_and(|from| same_language(from, target)) {
        // Already in the target language: the transcript is its own translation.
        return Some(TranslationData::new(transcript, target, text));
    }

    let mut pieces = Vec::new();
    let mut detected = None;
    for chunk in split_at_sentences(text, provider.max_chars()) {
        match provider.translate(chunk, from, target).await {
            Ok(Translated { text, source_language }) => {
                if !text.trim().is_empty() {
                    pieces.push(text.trim().to_string());
                }
                if detected.is_none() {
                    detected = source_language;
                }
            }
            Err(e) => {
                log::warn!("Translation ({target}): {e}");
                return None;
            }
        }
    }
    if pieces.is_empty() {
        return None;
    }
    let mut translation = TranslationData::new(transcript, target, pieces.join(" "));
    if translation.source_language.is_none() {
        translation.source_language = detected;
    }
    Some(translation)
}

/// Whether two BCP-47 tags name the same language, by primary subtag (`ml-IN` and `ml`).
pub fn same_language(a: &str, b: &str) -> bool {
    primary_subtag(a).eq_ignore_ascii_case(primary_subtag(b))
}

/// The primary language subtag of a BCP-47 tag (`ml` of `ml-IN`).
pub fn primary_subtag(tag: &str) -> &str {
    tag.split(['-', '_']).next().unwrap_or(tag)
}

/// `text` in pieces of at most `max_chars` characters, cut at sentence ends when there is one,
/// else at a space, else mid-word. One piece when it fits.
pub fn split_at_sentences(text: &str, max_chars: usize) -> Vec<&str> {
    let max = max_chars.max(1);
    let mut pieces = Vec::new();
    let mut rest = text.trim();
    while rest.chars().count() > max {
        // Byte offset of the character after the `max`th one.
        let limit = rest.char_indices().nth(max).map_or(rest.len(), |(i, _)| i);
        let head = &rest[..limit];
        let cut = head
            .rmatch_indices(['.', '?', '!', '।', '॥', '\n'])
            .next()
            .map(|(i, m)| i + m.len())
            .filter(|&i| i > 0)
            // A word ends exactly at the limit: keep it whole.
            .or_else(|| rest[limit..].starts_with(char::is_whitespace).then_some(limit))
            .or_else(|| head.rmatch_indices(' ').next().map(|(i, _)| i))
            .filter(|&i| i > 0)
            .unwrap_or(limit);
        pieces.push(rest[..cut].trim());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() || pieces.is_empty() {
        pieces.push(rest);
    }
    pieces
}

#[async_trait]
impl FrameHandler for TranslationService {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::Transcription(t)) if t.finalized => {
                self.submit(processor, t);
            }
            // The turn-closing transcript rides the stop frame (STT TurnGate).
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript: Some(t), .. })
                if t.finalized =>
            {
                self.submit(processor, t);
            }
            FrameInner::Control(ControlFrame::End { .. })
            | FrameInner::System(SystemFrame::Cancel { .. })
            | FrameInner::System(SystemFrame::Stop { .. }) => {
                self.stop_workers();
            }
            _ => {}
        }
        processor.push_frame(frame, direction).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::NoopBillingCollector;
    use crate::error::PipecatError;
    use crate::frames::StartFrameData;
    use std::time::Duration;

    /// `[from>to] text`; "slow" takes a while; "fail" errors.
    struct Tagger;

    #[async_trait]
    impl TranslationProvider for Tagger {
        fn name(&self) -> &'static str { "tagger" }
        fn model(&self) -> String { "tag:v1".into() }
        fn supports(&self, lang: &str) -> bool { lang != "xx" }
        fn max_chars(&self) -> usize { 40 }
        async fn translate(&self, text: &str, from: Option<&str>, to: &str) -> Result<Translated> {
            if text.contains("slow") {
                tokio::time::sleep(Duration::from_millis(80)).await;
            }
            if text.contains("fail") {
                return Err(PipecatError::pipeline("no"));
            }
            Ok(Translated {
                text: format!("[{}>{to}] {text}", from.unwrap_or("?")),
                source_language: from.map(str::to_string).or(Some("detected".into())),
            })
        }
    }

    /// A processor in direct mode that records what the service pushes.
    async fn harness(
        targets: TranslationTargets,
        billing: Option<Arc<dyn BillingCollector>>,
    ) -> (FrameProcessor, Arc<Mutex<Vec<TranslationData>>>) {
        let mut service = TranslationService::new(Tagger, targets);
        if let Some(b) = billing {
            service = service.with_billing(b);
        }
        let proc = FrameProcessor::new("translate-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream)
            .await
            .unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::Translation(t)) = &frame.inner {
                seen.lock().unwrap().push(t.clone());
            }
        });
        (proc, out)
    }

    async fn wait_for(out: &Arc<Mutex<Vec<TranslationData>>>, n: usize) -> Vec<TranslationData> {
        for _ in 0..100 {
            if out.lock().unwrap().len() >= n {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        out.lock().unwrap().clone()
    }

    fn final_transcript(text: &str, ts: &str) -> TranscriptionData {
        TranscriptionData::new(text, "u", ts).with_language("ml-IN").finalized()
    }

    #[tokio::test]
    async fn finals_are_translated_per_target_in_order_and_partials_are_not() {
        let (proc, out) = harness(TranslationTargets::new(["en", "hi"]), None).await;
        let frames = [
            Frame::transcription(TranscriptionData::new("partial", "u", "1")),
            Frame::transcription(final_transcript("slow first", "2")),
            Frame::transcription(final_transcript("second", "3")),
        ];
        for f in frames {
            proc.process_frame(f, FrameDirection::Downstream).await.unwrap();
        }
        let got = wait_for(&out, 4).await;
        assert_eq!(got.len(), 4, "{got:?}");
        let en: Vec<&str> = got.iter().filter(|t| t.target_language == "en").map(|t| t.text.as_str()).collect();
        let hi: Vec<&str> = got.iter().filter(|t| t.target_language == "hi").map(|t| t.text.as_str()).collect();
        assert_eq!(en, ["[ml-IN>en] slow first", "[ml-IN>en] second"], "a slow first still comes first");
        assert_eq!(hi, ["[ml-IN>hi] slow first", "[ml-IN>hi] second"]);
        let first = got.iter().find(|t| t.source_text == "slow first").unwrap();
        assert_eq!((first.timestamp.as_str(), first.source_language.as_deref()), ("2", Some("ml-IN")));
        assert!(first.finalized);
    }

    #[tokio::test]
    async fn the_bundled_turn_closing_transcript_is_translated_once() {
        let (proc, out) = harness(TranslationTargets::new(["en"]), None).await;
        let stop = Frame::vad_user_stopped_speaking(0.8, 1.0)
            .with_vad_stop_transcript(final_transcript("closing words", "9"));
        proc.process_frame(stop, FrameDirection::Downstream).await.unwrap();
        // The same transcript again (a re-push): not translated twice.
        proc.process_frame(Frame::transcription(final_transcript("closing words", "9")), FrameDirection::Downstream)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(60)).await;
        let got = wait_for(&out, 1).await;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "[ml-IN>en] closing words");
    }

    #[tokio::test]
    async fn targets_change_while_running_and_nothing_happens_without_any() {
        let targets = TranslationTargets::default();
        let (proc, out) = harness(targets.clone(), None).await;
        proc.process_frame(Frame::transcription(final_transcript("unwatched", "1")), FrameDirection::Downstream)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(out.lock().unwrap().is_empty());

        targets.set(["ta", "ta", ""]);
        assert_eq!(targets.get(), ["ta"]);
        proc.process_frame(Frame::transcription(final_transcript("watched", "2")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got[0].target_language, "ta");
        assert_eq!(got[0].text, "[ml-IN>ta] watched");
    }

    #[tokio::test]
    async fn a_transcript_without_a_language_takes_the_default_source() {
        let service = TranslationService::new(Tagger, TranslationTargets::new(["en"]))
            .with_default_source_language("ml-IN");
        let proc = FrameProcessor::new("translate-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream)
            .await
            .unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::Translation(t)) = &frame.inner {
                seen.lock().unwrap().push(t.clone());
            }
        });
        let unknown = TranscriptionData::new("നാളെ", "u", "1").finalized();
        proc.process_frame(Frame::transcription(unknown), FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got[0].text, "[ml-IN>en] നാളെ");
        assert_eq!(got[0].source_language.as_deref(), Some("ml-IN"));
    }

    #[tokio::test]
    async fn a_transcript_already_in_the_target_language_passes_through() {
        let (proc, out) = harness(TranslationTargets::new(["ml"]), None).await;
        proc.process_frame(Frame::transcription(final_transcript("നാളെ", "1")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got[0].text, "നാളെ");
        assert_eq!(got[0].source_language.as_deref(), Some("ml-IN"));
    }

    #[tokio::test]
    async fn long_text_is_split_and_joined_and_failures_are_skipped() {
        let (proc, out) = harness(TranslationTargets::new(["en"]), None).await;
        let long = "One sentence here. Another sentence follows it. And a third one ends.";
        assert!(long.chars().count() > 40);
        proc.process_frame(Frame::transcription(final_transcript("fail now", "1")), FrameDirection::Downstream)
            .await
            .unwrap();
        proc.process_frame(Frame::transcription(final_transcript(long, "2")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got.len(), 1, "the failed one is skipped: {got:?}");
        assert_eq!(
            got[0].text,
            "[ml-IN>en] One sentence here. [ml-IN>en] Another sentence follows it. [ml-IN>en] And a third one ends."
        );
    }

    #[tokio::test]
    async fn usage_is_billed_per_call_with_the_source_characters() {
        struct Record(Mutex<Vec<BillingEvent>>);
        impl BillingCollector for Record {
            fn record(&self, event: BillingEvent) { self.0.lock().unwrap().push(event); }
            fn session_id(&self) -> uuid::Uuid { uuid::Uuid::nil() }
        }
        let billing = Arc::new(Record(Mutex::new(Vec::new())));
        let (proc, out) = harness(TranslationTargets::new(["en"]), Some(billing.clone())).await;
        let unknown = TranscriptionData::new("  hello  ", "u", "1").finalized();
        proc.process_frame(Frame::transcription(unknown), FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got[0].source_language.as_deref(), Some("detected"), "the provider's detection is kept");
        let events = billing.0.lock().unwrap();
        match &events[0] {
            BillingEvent::TranslationUsage { provider, model, source_language, target_language, char_count, .. } => {
                assert_eq!((provider.as_str(), model.as_str()), ("tagger", "tag:v1"));
                assert_eq!(source_language.as_deref(), Some("detected"));
                assert_eq!(target_language, "en");
                assert_eq!(*char_count, 5);
            }
            other => panic!("{other:?}"),
        }
        let _ = NoopBillingCollector;
    }

    #[test]
    fn sentences_split_at_ends_then_spaces() {
        assert_eq!(split_at_sentences("short", 10), ["short"]);
        assert_eq!(split_at_sentences("", 10), [""]);
        assert_eq!(split_at_sentences("Hello there. How are you? Fine.", 14), ["Hello there.", "How are you?", "Fine."]);
        assert_eq!(split_at_sentences("no sentence ends in this text at all", 12), ["no sentence", "ends in this", "text at all"]);
        assert_eq!(split_at_sentences("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        // Limits count code points (combining marks included), not bytes or glyphs.
        assert_eq!(split_at_sentences("നാളെ മീറ്റിംഗ് ഉണ്ടോ? ഉണ്ട്.", 15), ["നാളെ മീറ്റിംഗ്", "ഉണ്ടോ? ഉണ്ട്."]);
    }

    #[test]
    fn languages_compare_by_primary_subtag() {
        assert!(same_language("ml-IN", "ml"));
        assert!(same_language("EN", "en-IN"));
        assert!(!same_language("hi-IN", "mr-IN"));
        assert_eq!(primary_subtag("pt_BR"), "pt");
    }
}
