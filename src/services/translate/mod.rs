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
//! `TranslationFrame` ([`TranslationData`], with the provider and how long it took). A call that fails
//! goes downstream as a `TranslationErrorFrame` ([`TranslationErrorData`], with what the error was
//! about: out of credits, rate limited, rejected, …), so captions, dubs and monitors know the
//! line will not come.
//!
//! A final transcript is not given up on after one call: a call that was rate limited, timed out
//! or failed otherwise is tried again ([`RETRY_WAITS`]), and then, with
//! [`with_fallback_provider`](TranslationService::with_fallback_provider), the other provider
//! gets it (at once when the first is out of credits, refused the text, or answered nothing).
//! With a fallback, the provider has [`PROVIDER_BUDGET`] for a line, retries included, so a
//! provider that hangs costs seconds, not its HTTP timeout. Once a line has gone to the fallback,
//! the next ones go to it first for [`FALLBACK_HOLD`] (the provider is asked only if the fallback
//! fails); then the provider is tried again, once per line, until it answers: a provider that is
//! down costs its retries once, not on every line, which would hold the queue back.
//! The error frame goes out only when all of them failed. With a
//! [`SentenceLedger`](crate::services::ledger::SentenceLedger) the sentence is opened here in
//! every dubbed language, and closed as `Failed` when it could not be translated.
//!
//! Interim transcripts are translated only with [`TranslationService::with_interim`], into its own
//! set of languages (those someone reads live, not those only dubbed): the newest interim
//! transcript per language, at most one call per interval, as a `TranslationFrame` with
//! `finalized: false`; one overtaken by a final transcript before its answer came is dropped.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, watch};

use crate::billing::{BillingCollector, BillingEvent};
use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, ErrorKind, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame, TranscriptionData, TranslationData, TranslationErrorData,
};
use crate::services::ledger::{Closed, SentenceKey, SentenceLedger, Stage};

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
    /// Who translates a final transcript the provider could not.
    fallback: Option<Arc<dyn TranslationProvider>>,
    /// The waits before each retry of a failed call (the fallback gets the first only).
    retry_waits: Vec<Duration>,
    /// With a fallback: how long the provider has for a line, retries included.
    provider_budget: Duration,
    ledger: Option<Arc<SentenceLedger>>,
    targets: TranslationTargets,
    billing: Option<Arc<dyn BillingCollector>>,
    /// The source language when a transcript names none.
    default_source: Option<String>,
    /// One queue per target language, made on first use.
    workers: Mutex<HashMap<String, mpsc::UnboundedSender<TranscriptionData>>>,
    /// The last transcript submitted: a transcript seen twice is translated once.
    last: Mutex<Option<(String, String)>>,
    /// Interim translation, when on ([`with_interim`](Self::with_interim)).
    interim: Option<Interim>,
    /// Counts final transcripts: an interim translation asked for before the latest final is
    /// stale, its utterance over.
    finals: Arc<AtomicU64>,
    /// Final translations being made: interim ones wait while any is, so the finals (what dubs
    /// are made of) never queue behind them at a provider that limits concurrent calls.
    finals_in_flight: Arc<AtomicU64>,
    /// Until when the fallback is asked first: the provider failed a line it then translated.
    provider_down_until: Arc<Mutex<Option<Instant>>>,
}

/// Interim translation: its languages, how often, its provider when not the service's, and the
/// newest interim transcript per language.
struct Interim {
    every: Duration,
    targets: TranslationTargets,
    provider: Option<Arc<dyn TranslationProvider>>,
    workers: Mutex<HashMap<String, watch::Sender<Option<(u64, TranscriptionData)>>>>,
}

/// Interim transcripts shorter than this are not worth a call.
const MIN_INTERIM_CHARS: usize = 8;

/// The waits before the retries of a failed final translation: rate limited, timed out, or failed
/// otherwise (out of credits or a refused text are not retried).
pub const RETRY_WAITS: [Duration; 2] = [Duration::from_millis(500), Duration::from_millis(1500)];

/// After the fallback translated a line the provider could not, how long the fallback is asked
/// first.
pub const FALLBACK_HOLD: Duration = Duration::from_secs(60);

/// With a fallback, how long the provider has for a line, retries included (Sarvam Translate
/// answers in 0.6-0.8 s on a normal day).
pub const PROVIDER_BUDGET: Duration = Duration::from_secs(3);

/// What became of one transcript in one language.
enum Outcome {
    Translated(TranslationData),
    Failed(String),
    /// The provider answered nothing.
    Nothing,
}

impl TranslationService {
    /// Translate into `targets` with `provider`.
    pub fn new(provider: impl TranslationProvider, targets: TranslationTargets) -> Self {
        Self::with_provider(Arc::new(provider), targets)
    }

    pub fn with_provider(provider: Arc<dyn TranslationProvider>, targets: TranslationTargets) -> Self {
        Self {
            provider,
            fallback: None,
            retry_waits: RETRY_WAITS.to_vec(),
            provider_budget: PROVIDER_BUDGET,
            ledger: None,
            targets,
            billing: None,
            default_source: None,
            workers: Mutex::new(HashMap::new()),
            last: Mutex::new(None),
            interim: None,
            finals: Arc::new(AtomicU64::new(0)),
            finals_in_flight: Arc::new(AtomicU64::new(0)),
            provider_down_until: Arc::new(Mutex::new(None)),
        }
    }

    /// Also translate interim transcripts into `targets` (a set apart from the final targets: the
    /// languages read live), the newest one per language and at most one call per `every`.
    pub fn with_interim(mut self, every: Duration, targets: TranslationTargets) -> Self {
        self.interim = Some(Interim { every, targets, provider: None, workers: Mutex::new(HashMap::new()) });
        self
    }

    /// Make interim translations with `provider` instead of the service's: a fast one with room for
    /// many calls, so the final translations' provider is left to the finals. (Sarvam Translate
    /// slows every call of a key that calls it this often: finals then queued for a minute.)
    /// After [`with_interim`](Self::with_interim).
    pub fn with_interim_provider(mut self, provider: Arc<dyn TranslationProvider>) -> Self {
        if let Some(interim) = &mut self.interim {
            interim.provider = Some(provider);
        }
        self
    }

    /// The handle the interim targets can be changed through, when interim translation is on.
    pub fn interim_targets(&self) -> Option<TranslationTargets> {
        self.interim.as_ref().map(|i| i.targets.clone())
    }

    /// Translate the final transcripts `provider` (the service's) could not, after its retries:
    /// out of credits, rate limited, down, or answering nothing.
    pub fn with_fallback_provider(mut self, provider: Arc<dyn TranslationProvider>) -> Self {
        self.fallback = Some(provider);
        self
    }

    /// The waits before each retry of a failed final translation ([`RETRY_WAITS`] by default).
    pub fn with_retry_waits(mut self, waits: Vec<Duration>) -> Self {
        self.retry_waits = waits;
        self
    }

    /// With a fallback, how long the provider has for a line, retries included
    /// ([`PROVIDER_BUDGET`] by default).
    pub fn with_provider_budget(mut self, budget: Duration) -> Self {
        self.provider_budget = budget;
        self
    }

    /// Open every final transcript in `ledger` in the languages it dubs, and close the ones that
    /// could not be translated.
    pub fn with_ledger(mut self, ledger: Arc<SentenceLedger>) -> Self {
        self.ledger = Some(ledger);
        self
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
        // Whatever interim translation is in flight is for an utterance now over.
        self.finals.fetch_add(1, Ordering::SeqCst);
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
        if let Some(ledger) = &self.ledger {
            ledger.open(transcript, &targets);
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
        let mut providers = vec![Arc::clone(&self.provider)];
        providers.extend(self.fallback.clone());
        let retry_waits = self.retry_waits.clone();
        let down_until = Arc::clone(&self.provider_down_until);
        let budget = self.provider_budget;
        let ledger = self.ledger.clone();
        let billing = self.billing.clone();
        let default_source = self.default_source.clone();
        let in_flight = Arc::clone(&self.finals_in_flight);
        tokio::spawn(async move {
            while let Some(mut transcript) = transcripts.recv().await {
                let key = SentenceKey::of_transcript(&transcript, &target);
                if transcript.language.is_none() {
                    transcript.language = default_source.clone();
                }
                // Already in the target language: passed through, no call made, nothing to bill.
                let passed_through = transcript.language.as_deref().is_some_and(|from| same_language(from, &target));
                in_flight.fetch_add(1, Ordering::SeqCst);
                let (outcome, provider) = translate_final(&providers, &retry_waits, budget, &down_until, &transcript, &target).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let translation = match outcome {
                    Outcome::Translated(t) => t,
                    Outcome::Nothing | Outcome::Failed(_) => {
                        let error = match outcome {
                            Outcome::Failed(e) => e,
                            _ => format!("{} answered nothing", provider.name()),
                        };
                        if let Some(ledger) = &ledger {
                            ledger.close(&key, Closed::Failed(format!("translation: {error}")));
                        }
                        let failed = TranslationErrorData::new(&transcript, &target, provider.name(), error);
                        if let Err(e) = processor.push_frame(Frame::translation_error(failed), FrameDirection::Downstream).await {
                            log::warn!("Translation: could not push a translation error ({target}): {e}");
                        }
                        continue;
                    }
                };
                if let Some(ledger) = &ledger {
                    // Before it is pushed: the dubbing stage may take it at once.
                    ledger.advance(&key, Stage::Speaking);
                }
                if let (Some(bc), false) = (&billing, passed_through) {
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

    /// Hands an interim transcript to every interim target's worker, replacing the one waiting.
    fn submit_interim(&self, processor: &FrameProcessor, transcript: &TranscriptionData) {
        let Some(interim) = &self.interim else { return };
        if transcript.text.trim().chars().count() < MIN_INTERIM_CHARS {
            return;
        }
        let generation = self.finals.load(Ordering::SeqCst);
        let mut workers = interim.workers.lock().unwrap();
        for target in interim.targets.get() {
            let from = transcript.language.as_deref().or(self.default_source.as_deref());
            if from.is_some_and(|from| same_language(from, &target)) {
                continue; // its own transcript says it
            }
            let slot = workers
                .entry(target.clone())
                .or_insert_with(|| self.interim_worker(processor.clone(), target, interim.every));
            slot.send_replace(Some((generation, transcript.clone())));
        }
    }

    /// Translates one language's newest interim transcript, at most once per `every`, until its
    /// sender is dropped.
    fn interim_worker(
        &self,
        processor: FrameProcessor,
        target: String,
        every: Duration,
    ) -> watch::Sender<Option<(u64, TranscriptionData)>> {
        let (slot, mut newest) = watch::channel::<Option<(u64, TranscriptionData)>>(None);
        let provider = self
            .interim
            .as_ref()
            .and_then(|i| i.provider.clone())
            .unwrap_or_else(|| Arc::clone(&self.provider));
        let billing = self.billing.clone();
        let default_source = self.default_source.clone();
        let finals = Arc::clone(&self.finals);
        let finals_in_flight = Arc::clone(&self.finals_in_flight);
        tokio::spawn(async move {
            let mut last = String::new();
            while newest.changed().await.is_ok() {
                let Some((generation, mut transcript)) = newest.borrow_and_update().clone() else { continue };
                let text: String = transcript.text.trim().chars().take(provider.max_chars()).collect();
                if generation != finals.load(Ordering::SeqCst) || text == last {
                    continue;
                }
                if transcript.language.is_none() {
                    transcript.language = default_source.clone();
                }
                // A final being translated goes first; this one waits (a newer may replace it).
                while finals_in_flight.load(Ordering::SeqCst) > 0 {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                if generation != finals.load(Ordering::SeqCst) {
                    continue;
                }
                let started = Instant::now();
                let answer = provider.translate(&text, transcript.language.as_deref(), &target).await;
                let stale = generation != finals.load(Ordering::SeqCst);
                match answer {
                    Ok(Translated { text: translated, source_language }) if !translated.trim().is_empty() => {
                        last = text.clone();
                        if let Some(bc) = &billing {
                            bc.record(BillingEvent::TranslationUsage {
                                session_id: bc.session_id(),
                                provider: provider.name().to_string(),
                                model: provider.model(),
                                source_language: transcript.language.clone().or(source_language.clone()),
                                target_language: target.clone(),
                                char_count: text.chars().count(),
                                occurred_at: Utc::now(),
                            });
                        }
                        if !stale {
                            let mut translation = TranslationData::new(&transcript, &target, translated.trim());
                            translation.finalized = false;
                            translation.provider = Some(provider.name().to_string());
                            translation.latency_ms = Some(started.elapsed().as_millis() as u64);
                            if translation.source_language.is_none() {
                                translation.source_language = source_language;
                            }
                            let _ = processor.push_frame(Frame::translation(translation), FrameDirection::Downstream).await;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        log::debug!("Translation ({target}, interim): {e}");
                        let failed = TranslationErrorData::new(&transcript, &target, provider.name(), e.to_string());
                        let _ = processor.push_frame(Frame::translation_error(failed), FrameDirection::Downstream).await;
                    }
                }
                tokio::time::sleep(every).await;
            }
        });
        slot
    }

    fn stop_workers(&self) {
        self.workers.lock().unwrap().clear();
        if let Some(interim) = &self.interim {
            interim.workers.lock().unwrap().clear();
        }
    }
}

/// One final transcript into one language, not given up on lightly: the service's provider, its
/// failed call tried again after each of `retry_waits` when it was rate limited, timed out or
/// failed otherwise; then the fallback (`providers[1]`, when there is one), retried after the
/// first wait only. With a fallback the provider has `budget` for the line, retries included.
/// While `down_until` is ahead (the provider failed a line the fallback then translated), the
/// fallback goes first and the provider is asked once, after it; once it has passed, the provider
/// is asked first again but only once per line, until it answers. The outcome, and the provider it
/// came from (the last one asked).
async fn translate_final<'a>(
    providers: &'a [Arc<dyn TranslationProvider>],
    retry_waits: &[Duration],
    budget: Duration,
    down_until: &Mutex<Option<Instant>>,
    transcript: &TranscriptionData,
    target: &str,
) -> (Outcome, &'a dyn TranslationProvider) {
    let first_wait = &retry_waits[..retry_waits.len().min(1)];
    let fallback = providers.get(1).filter(|f| f.supports(target));
    let (provider_down, recovering) = match *down_until.lock().unwrap() {
        Some(until) if fallback.is_some() => (Instant::now() < until, Instant::now() >= until),
        _ => (false, false),
    };
    let mut order: Vec<(&'a dyn TranslationProvider, &[Duration], bool)> = Vec::new(); // (who, waits, is the fallback)
    match fallback {
        Some(f) if provider_down => {
            order.push((f.as_ref(), first_wait, true));
            order.push((providers[0].as_ref(), &[], false));
        }
        Some(f) => {
            order.push((providers[0].as_ref(), if recovering { &[] } else { retry_waits }, false));
            order.push((f.as_ref(), first_wait, true));
        }
        None => order.push((providers[0].as_ref(), retry_waits, false)),
    }
    // Only with somewhere else to go is the provider hurried.
    let deadline = fallback.map(|_| Instant::now() + budget);
    let mut last = (Outcome::Nothing, providers[0].as_ref());
    let mut provider_failed = false;
    for (i, &(provider, waits, is_fallback)) in order.iter().enumerate() {
        let mut tries = 0;
        loop {
            let outcome = match deadline.filter(|_| !is_fallback) {
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    match tokio::time::timeout(left, translate_one(provider, transcript, target)).await {
                        Ok(outcome) => outcome,
                        Err(_) => Outcome::Failed(format!("no answer within {} ms", budget.as_millis())),
                    }
                }
                None => translate_one(provider, transcript, target).await,
            };
            let retry = match &outcome {
                Outcome::Translated(t) => {
                    // A pass-through (no call made) says nothing about the providers.
                    if t.provider.is_some() {
                        let mut down = down_until.lock().unwrap();
                        if !is_fallback {
                            *down = None;
                        } else if provider_failed && !provider_down {
                            log::warn!(
                                "Translation ({target}): {} first for {} s",
                                provider.name(),
                                FALLBACK_HOLD.as_secs()
                            );
                            *down = Some(Instant::now() + FALLBACK_HOLD);
                        }
                    }
                    return (outcome, provider);
                }
                Outcome::Nothing => false,
                Outcome::Failed(e) => matches!(ErrorKind::of(e), ErrorKind::RateLimited | ErrorKind::Timeout | ErrorKind::Other),
            };
            last = (outcome, provider);
            // No retry the budget leaves no time for.
            let in_budget = |wait: &Duration| is_fallback || deadline.is_none_or(|d| Instant::now() + *wait < d);
            match waits.get(tries) {
                Some(wait) if retry && in_budget(wait) => {
                    log::info!("Translation ({target}): {} failed; again in {} ms", provider.name(), wait.as_millis());
                    tokio::time::sleep(*wait).await;
                    tries += 1;
                }
                _ => break,
            }
        }
        if !is_fallback {
            provider_failed = true;
        }
        if let Some((next, _, _)) = order.get(i + 1) {
            log::warn!("Translation ({target}): {} could not; trying {}", provider.name(), next.name());
        }
    }
    last
}

/// One transcript into one language: the provider's answer as a `TranslationData` (with the
/// provider and how long it took), the error when the call failed, or nothing when it answered
/// nothing.
async fn translate_one(
    provider: &dyn TranslationProvider,
    transcript: &TranscriptionData,
    target: &str,
) -> Outcome {
    let text = transcript.text.trim();
    let from = transcript.language.as_deref();
    if from.is_some_and(|from| same_language(from, target)) {
        // Already in the target language: the transcript is its own translation.
        return Outcome::Translated(TranslationData::new(transcript, target, text));
    }

    let started = Instant::now();
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
                return Outcome::Failed(e.to_string());
            }
        }
    }
    if pieces.is_empty() {
        return Outcome::Nothing;
    }
    let mut translation = TranslationData::new(transcript, target, pieces.join(" "));
    if translation.source_language.is_none() {
        translation.source_language = detected;
    }
    translation.provider = Some(provider.name().to_string());
    translation.latency_ms = Some(started.elapsed().as_millis() as u64);
    Outcome::Translated(translation)
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
            FrameInner::Data(DataFrame::Transcription(t)) => {
                self.submit_interim(processor, t);
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

    const QUICK_RETRIES: [Duration; 2] = [Duration::from_millis(5), Duration::from_millis(5)];

    /// A processor in direct mode that records what the service pushes.
    async fn harness(
        targets: TranslationTargets,
        billing: Option<Arc<dyn BillingCollector>>,
    ) -> (FrameProcessor, Arc<Mutex<Vec<TranslationData>>>) {
        let mut service = TranslationService::new(Tagger, targets).with_retry_waits(QUICK_RETRIES.to_vec());
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

    #[tokio::test]
    async fn a_transcript_already_in_the_target_language_is_passed_through_unbilled() {
        struct Record(Mutex<Vec<BillingEvent>>);
        impl BillingCollector for Record {
            fn record(&self, event: BillingEvent) { self.0.lock().unwrap().push(event); }
            fn session_id(&self) -> uuid::Uuid { uuid::Uuid::nil() }
        }
        let billing = Arc::new(Record(Mutex::new(Vec::new())));
        let (proc, out) = harness(TranslationTargets::new(["ml", "en"]), Some(billing.clone())).await;
        proc.process_frame(Frame::transcription(final_transcript("നാളെ", "1")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 2).await;
        let ml = got.iter().find(|t| t.target_language == "ml").unwrap();
        assert_eq!(ml.text, "നാളെ", "its own translation");
        let billed: Vec<String> = billing
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                BillingEvent::TranslationUsage { target_language, .. } => Some(target_language.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(billed, ["en"], "only the real translation is billed");
    }

    #[tokio::test]
    async fn a_failed_call_goes_downstream_as_an_error() {
        let service = TranslationService::new(Tagger, TranslationTargets::new(["en"])).with_retry_waits(QUICK_RETRIES.to_vec());
        let proc = FrameProcessor::new("translate-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let errors = Arc::new(Mutex::new(Vec::new()));
        let seen = errors.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::TranslationError(e)) = &frame.inner {
                seen.lock().unwrap().push(e.clone());
            }
        });
        proc.process_frame(Frame::transcription(final_transcript("fail now", "4")), FrameDirection::Downstream)
            .await
            .unwrap();
        for _ in 0..100 {
            if !errors.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let errors = errors.lock().unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!((errors[0].timestamp.as_str(), errors[0].target_language.as_str()), ("4", "en"));
        assert_eq!((errors[0].provider.as_str(), errors[0].finalized), ("tagger", true));
        assert_eq!(errors[0].kind, crate::frames::ErrorKind::Other);
        assert_eq!(crate::frames::ErrorKind::of("HTTP 402 Payment Required"), crate::frames::ErrorKind::Credits);
    }

    #[tokio::test]
    async fn translations_say_who_made_them_and_how_long_it_took() {
        let (proc, out) = harness(TranslationTargets::new(["en"]), None).await;
        proc.process_frame(Frame::transcription(final_transcript("slow one", "1")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got[0].provider.as_deref(), Some("tagger"));
        assert!(got[0].latency_ms.unwrap() >= 70, "{:?}", got[0].latency_ms);
    }

    #[tokio::test]
    async fn interim_transcripts_are_translated_into_the_interim_languages_only() {
        let service = TranslationService::new(Tagger, TranslationTargets::new(["en", "hi"]))
            .with_interim(Duration::from_millis(20), TranslationTargets::new(["en"]));
        let proc = FrameProcessor::new("translate-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::Translation(t)) = &frame.inner {
                seen.lock().unwrap().push(t.clone());
            }
        });
        let partial = |text: &str| Frame::transcription(TranscriptionData::new(text, "u", "1").with_language("ml-IN"));
        proc.process_frame(partial("short"), FrameDirection::Downstream).await.unwrap();
        proc.process_frame(partial("words being spoken"), FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!(got.len(), 1, "too short, and not into hi: {got:?}");
        assert_eq!((got[0].text.as_str(), got[0].finalized), ("[ml-IN>en] words being spoken", false));
        assert_eq!(got[0].provider.as_deref(), Some("tagger"));

        // One asked for before a final, answered after it: its utterance is over, so it is dropped.
        proc.process_frame(partial("slow words still coming"), FrameDirection::Downstream).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        proc.process_frame(Frame::transcription(final_transcript("the final words", "2")), FrameDirection::Downstream)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let got = out.lock().unwrap().clone();
        assert!(!got.iter().any(|t| t.source_text.contains("slow words")), "{got:?}");
        assert!(got.iter().any(|t| t.finalized && t.target_language == "hi"), "finals still go to every target");
    }

    #[tokio::test]
    async fn interim_translations_can_have_a_provider_of_their_own() {
        struct Quick;
        #[async_trait]
        impl TranslationProvider for Quick {
            fn name(&self) -> &'static str { "quick" }
            fn model(&self) -> String { "quick:v1".into() }
            fn supports(&self, _lang: &str) -> bool { true }
            async fn translate(&self, text: &str, _from: Option<&str>, to: &str) -> Result<Translated> {
                Ok(Translated { text: format!("quick {to}: {text}"), source_language: None })
            }
        }
        let service = TranslationService::new(Tagger, TranslationTargets::new(["en"]))
            .with_interim(Duration::from_millis(10), TranslationTargets::new(["en"]))
            .with_interim_provider(Arc::new(Quick));
        let proc = FrameProcessor::new("translate-test", Box::new(service), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::Data(DataFrame::Translation(t)) = &frame.inner {
                seen.lock().unwrap().push(t.clone());
            }
        });
        let partial = Frame::transcription(TranscriptionData::new("words being spoken", "u", "1").with_language("ml-IN"));
        proc.process_frame(partial, FrameDirection::Downstream).await.unwrap();
        let got = wait_for(&out, 1).await;
        assert_eq!((got[0].text.as_str(), got[0].provider.as_deref()), ("quick en: words being spoken", Some("quick")));
        proc.process_frame(Frame::transcription(final_transcript("the final words", "2")), FrameDirection::Downstream)
            .await
            .unwrap();
        let got = wait_for(&out, 2).await;
        assert_eq!(got[1].provider.as_deref(), Some("tagger"), "finals keep the service's provider");
    }

    /// Fails its first `fails` calls with `error`, then translates; answers nothing when `empty`.
    struct Flaky {
        name: &'static str,
        fails: std::sync::atomic::AtomicUsize,
        error: &'static str,
        empty: bool,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Flaky {
        fn new(name: &'static str, fails: usize, error: &'static str) -> Arc<Self> {
            Arc::new(Self { name, fails: fails.into(), error, empty: false, calls: 0.into() })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl TranslationProvider for Flaky {
        fn name(&self) -> &'static str { self.name }
        fn model(&self) -> String { format!("{}:v1", self.name) }
        fn supports(&self, _lang: &str) -> bool { true }
        async fn translate(&self, text: &str, _from: Option<&str>, to: &str) -> Result<Translated> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fails.load(Ordering::SeqCst) > 0 {
                self.fails.fetch_sub(1, Ordering::SeqCst);
                return Err(PipecatError::pipeline(self.error));
            }
            let text = if self.empty { String::new() } else { format!("{} {to}: {text}", self.name) };
            Ok(Translated { text, source_language: None })
        }
    }

    /// Translations and errors out of a service in direct mode.
    async fn run(
        service: TranslationService,
        transcripts: &[TranscriptionData],
        wait: Duration,
    ) -> (Vec<TranslationData>, Vec<TranslationErrorData>) {
        let proc = FrameProcessor::new("translate-test", Box::new(service.with_retry_waits(QUICK_RETRIES.to_vec())), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new((Vec::new(), Vec::new())));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| match &frame.inner {
            FrameInner::Data(DataFrame::Translation(t)) => seen.lock().unwrap().0.push(t.clone()),
            FrameInner::Data(DataFrame::TranslationError(e)) => seen.lock().unwrap().1.push(e.clone()),
            _ => {}
        });
        for t in transcripts {
            proc.process_frame(Frame::transcription(t.clone()), FrameDirection::Downstream).await.unwrap();
        }
        tokio::time::sleep(wait).await;
        let out = out.lock().unwrap();
        (out.0.clone(), out.1.clone())
    }

    #[tokio::test]
    async fn a_rate_limited_call_is_tried_again() {
        let primary = Flaky::new("primary", 2, "HTTP 429 Too Many Requests");
        let service = TranslationService::with_provider(primary.clone(), TranslationTargets::new(["en"]));
        let (got, errors) = run(service, &[final_transcript("one", "1")], Duration::from_millis(100)).await;
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(primary.calls(), 3);
    }

    #[tokio::test]
    async fn what_the_provider_cannot_translate_goes_to_the_fallback_in_order() {
        // Out of credits: no retry, straight to the fallback, which takes the next line too, in order.
        let primary = Flaky::new("primary", 1, "HTTP 402 Payment Required");
        let fallback = Flaky::new("fallback", 0, "");
        let service = TranslationService::with_provider(primary.clone(), TranslationTargets::new(["en"]))
            .with_fallback_provider(fallback.clone());
        let (got, errors) =
            run(service, &[final_transcript("one", "1"), final_transcript("two", "2")], Duration::from_millis(100)).await;
        assert!(errors.is_empty(), "{errors:?}");
        let got: Vec<(&str, Option<&str>)> = got.iter().map(|t| (t.text.as_str(), t.provider.as_deref())).collect();
        assert_eq!(got, [("fallback en: one", Some("fallback")), ("fallback en: two", Some("fallback"))], "the fallback first, for a while");
        assert_eq!((primary.calls(), fallback.calls()), (1, 2));
    }

    /// A provider that hangs costs the budget, not its HTTP timeout.
    #[tokio::test]
    async fn a_provider_that_hangs_has_a_budget_when_there_is_a_fallback() {
        struct Hangs;
        #[async_trait]
        impl TranslationProvider for Hangs {
            fn name(&self) -> &'static str { "hangs" }
            fn model(&self) -> String { "hangs:v1".into() }
            fn supports(&self, _lang: &str) -> bool { true }
            async fn translate(&self, _text: &str, _from: Option<&str>, _to: &str) -> Result<Translated> {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Err(PipecatError::pipeline("too late"))
            }
        }
        let service = TranslationService::with_provider(Arc::new(Hangs), TranslationTargets::new(["en"]))
            .with_fallback_provider(Flaky::new("fallback", 0, ""))
            .with_provider_budget(Duration::from_millis(50));
        let (got, _) = run(service, &[final_transcript("one", "1")], Duration::from_millis(200)).await;
        assert_eq!(got.len(), 1, "the fallback answered within the budget and a bit");
        assert_eq!(got[0].provider.as_deref(), Some("fallback"));
    }

    /// After the hold, the provider is asked once per line until it answers.
    #[tokio::test]
    async fn after_the_hold_the_provider_gets_one_try_per_line() {
        let primary = Flaky::new("primary", 99, "connection refused");
        let fallback = Flaky::new("fallback", 0, "");
        let service = TranslationService::with_provider(primary.clone(), TranslationTargets::new(["en"]))
            .with_fallback_provider(fallback.clone());
        // The hold already over: the provider was down a while ago.
        *service.provider_down_until.lock().unwrap() = Some(Instant::now());
        let (got, _) = run(service, &[final_transcript("one", "1")], Duration::from_millis(100)).await;
        assert_eq!(got[0].provider.as_deref(), Some("fallback"));
        assert_eq!(primary.calls(), 1, "one try, no retries");
    }

    /// A provider that is down costs its retries once, not on every line.
    #[tokio::test]
    async fn after_one_line_to_the_fallback_the_next_ones_go_to_it_first() {
        let primary = Flaky::new("primary", 99, "connection refused");
        let fallback = Flaky::new("fallback", 0, "");
        let service = TranslationService::with_provider(primary.clone(), TranslationTargets::new(["en"]))
            .with_fallback_provider(fallback.clone());
        let lines: Vec<TranscriptionData> = (1..=4).map(|i| final_transcript(&format!("line {i}"), &i.to_string())).collect();
        let (got, errors) = run(service, &lines, Duration::from_millis(150)).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|t| t.provider.as_deref() == Some("fallback")));
        assert_eq!(primary.calls(), 3, "three tries on the first line only");
        assert_eq!(fallback.calls(), 4);
    }

    #[tokio::test]
    async fn a_provider_answering_nothing_is_covered_by_the_fallback() {
        let primary = Arc::new(Flaky { empty: true, ..Arc::into_inner(Flaky::new("primary", 0, "")).unwrap() });
        let fallback = Flaky::new("fallback", 0, "");
        let service = TranslationService::with_provider(primary, TranslationTargets::new(["en"])).with_fallback_provider(fallback);
        let (got, errors) = run(service, &[final_transcript("one", "1")], Duration::from_millis(100)).await;
        assert!(errors.is_empty());
        assert_eq!(got[0].text, "fallback en: one");
    }

    #[tokio::test]
    async fn when_every_provider_fails_the_sentence_is_one_error_and_closed_as_lost() {
        let ledger = Arc::new(SentenceLedger::new(TranslationTargets::new(["en"])));
        let primary = Flaky::new("primary", 99, "connection reset");
        let fallback = Flaky::new("fallback", 99, "connection reset");
        let service = TranslationService::with_provider(primary.clone(), TranslationTargets::new(["en"]))
            .with_fallback_provider(fallback.clone())
            .with_ledger(Arc::clone(&ledger));
        let (got, errors) = run(service, &[final_transcript("one", "1")], Duration::from_millis(150)).await;
        assert!(got.is_empty());
        assert_eq!(errors.len(), 1, "one error, after every attempt: {errors:?}");
        assert_eq!(errors[0].provider, "fallback");
        assert_eq!((primary.calls(), fallback.calls()), (3, 2), "the primary twice more, the fallback once more");
        assert_eq!(ledger.counts()["en"].failed, 1);
        assert!(ledger.open_counts().is_empty());
    }

    #[tokio::test]
    async fn a_dubbed_sentence_is_opened_and_moves_on_once_translated() {
        let ledger = Arc::new(SentenceLedger::new(TranslationTargets::new(["hi"])));
        let service = TranslationService::new(Tagger, TranslationTargets::new(["en", "hi"])).with_ledger(Arc::clone(&ledger));
        let (got, _) = run(service, &[final_transcript("one", "1"), final_transcript("", "2")], Duration::from_millis(50)).await;
        assert_eq!(got.len(), 2);
        let hi = SentenceKey::of_transcript(&final_transcript("one", "1"), "hi");
        assert_eq!(ledger.stage(&hi), Some(Stage::Speaking), "now the dubbing's to close");
        assert_eq!(ledger.open_counts().get("hi"), Some(&1), "en is not dubbed; the empty one opens nothing");
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
