//! The sentence ledger: what became of every sentence, in every language it is dubbed into.
//!
//! The STT's turn gate makes a turn's boundary exactly-once; this does the same for a sentence
//! after it. Each final transcript with words in it is **opened** once per dubbed language when
//! [`TranslationService`](crate::services::translate::TranslationService) takes it, moves through
//! [`Stage`]s as it is translated and spoken, and is **closed** exactly once:
//!
//! ```text
//!   open ─▶ Translating ─▶ Speaking ─▶ Playing ─▶ close: Played | Failed(why) | Abandoned(why)
//! ```
//!
//! - `Played`: it was heard.
//! - `Failed`: it was lost, after every retry and every provider: the only outcome that costs a
//!   sentence, always with the reason.
//! - `Abandoned`: nobody was there for it (its listeners left, or its language stopped being
//!   dubbed before it was spoken). Not a loss.
//!
//! Closing is a claim: the entry is removed from the open set under its lock, so of two parties
//! racing to close a sentence exactly one wins (the second close is a no-op), like the turn
//! gate's `take()` of its pending stop. An entry still open long after it was opened
//! ([`stuck`](SentenceLedger::stuck)) is a path that forgot to close: a bug, reported with the
//! stage it is stuck in.
//!
//! A turn with no words (music, noise, or a pause the VAD took for speech, that the STT answers
//! with an empty final or not at all) opens nothing, so it can neither be lost nor stuck.
//!
//! The services take the ledger through `with_ledger`; whatever plays a `DubbedSpeechFrame` (it
//! is `Playing` by then) closes it. Hosts without one are unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::frames::{DubbedSpeechData, TranscriptionData, TranslationData};
use crate::services::translate::TranslationTargets;

/// An entry open this long is reported as [stuck](SentenceLedger::stuck).
pub const STUCK_AFTER: Duration = Duration::from_secs(60);

/// A sentence in one language: the transcript's `timestamp` and text, and the language.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SentenceKey {
    pub timestamp: String,
    pub source_text: String,
    pub target_language: String,
}

impl SentenceKey {
    pub fn of_transcript(transcript: &TranscriptionData, target_language: &str) -> Self {
        Self {
            timestamp: transcript.timestamp.clone(),
            source_text: transcript.text.clone(),
            target_language: target_language.to_string(),
        }
    }

    pub fn of_translation(translation: &TranslationData) -> Self {
        Self {
            timestamp: translation.timestamp.clone(),
            source_text: translation.source_text.clone(),
            target_language: translation.target_language.clone(),
        }
    }

    pub fn of_speech(speech: &DubbedSpeechData) -> Self {
        Self {
            timestamp: speech.timestamp.clone(),
            source_text: speech.source_text.clone(),
            target_language: speech.target_language.clone(),
        }
    }
}

/// Where an open sentence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Translating,
    /// Translated; its speech is being made.
    Speaking,
    /// Its speech was handed to whatever plays it.
    Playing,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Translating => "translating",
            Stage::Speaking => "speaking",
            Stage::Playing => "playing",
        }
    }
}

/// How a sentence ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Closed {
    Played,
    Failed(String),
    Abandoned(String),
}

/// Sentences closed in one language.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub played: u64,
    pub failed: u64,
    pub abandoned: u64,
}

/// An entry open past [`STUCK_AFTER`].
#[derive(Debug, Clone)]
pub struct Stuck {
    pub key: SentenceKey,
    pub stage: Stage,
    pub open_for: Duration,
}

struct Entry {
    stage: Stage,
    opened: Instant,
}

/// Called with every sentence closed (a host counting outcomes across ledgers).
pub type OnClose = Arc<dyn Fn(&SentenceKey, &Closed) + Send + Sync>;

/// See the module docs.
pub struct SentenceLedger {
    /// The languages dubbed: a sentence is opened in those only.
    dubbed: TranslationTargets,
    open: Mutex<HashMap<SentenceKey, Entry>>,
    counts: Mutex<HashMap<String, Counts>>,
    stuck_after: Duration,
    on_close: Option<OnClose>,
}

impl SentenceLedger {
    /// A ledger of the sentences dubbed into `dubbed` (the `DubbingService`'s targets).
    pub fn new(dubbed: TranslationTargets) -> Self {
        Self {
            dubbed,
            open: Mutex::new(HashMap::new()),
            counts: Mutex::new(HashMap::new()),
            stuck_after: STUCK_AFTER,
            on_close: None,
        }
    }

    pub fn with_stuck_after(mut self, after: Duration) -> Self {
        self.stuck_after = after;
        self
    }

    pub fn with_on_close(mut self, on_close: OnClose) -> Self {
        self.on_close = Some(on_close);
        self
    }

    /// Opens `transcript` in each of `targets` that is dubbed. Nothing for a transcript without
    /// words; an entry already open stays as it is.
    pub fn open(&self, transcript: &TranscriptionData, targets: &[String]) {
        if transcript.text.trim().is_empty() {
            return;
        }
        let dubbed = self.dubbed.get();
        let mut open = self.open.lock().unwrap();
        for target in targets.iter().filter(|t| dubbed.contains(t)) {
            open.entry(SentenceKey::of_transcript(transcript, target))
                .or_insert_with(|| Entry { stage: Stage::Translating, opened: Instant::now() });
        }
    }

    /// Moves an open sentence on to `stage`; nothing when it is not open.
    pub fn advance(&self, key: &SentenceKey, stage: Stage) {
        if let Some(entry) = self.open.lock().unwrap().get_mut(key) {
            entry.stage = stage;
        }
    }

    /// Closes `key`: true for the one call that closed it, false when it was not open (closed
    /// already, or never opened).
    pub fn close(&self, key: &SentenceKey, closed: Closed) -> bool {
        if self.open.lock().unwrap().remove(key).is_none() {
            log::debug!("SentenceLedger: {:?} not open; {:?} ignored", key, closed);
            return false;
        }
        {
            let mut counts = self.counts.lock().unwrap();
            let counts = counts.entry(key.target_language.clone()).or_default();
            match &closed {
                Closed::Played => counts.played += 1,
                Closed::Failed(_) => counts.failed += 1,
                Closed::Abandoned(_) => counts.abandoned += 1,
            }
        }
        if let Closed::Failed(why) = &closed {
            log::warn!("SentenceLedger: lost ({}) {:?}: {why}", key.target_language, key.source_text);
        }
        if let Some(on_close) = &self.on_close {
            on_close(key, &closed);
        }
        true
    }

    pub fn is_open(&self, key: &SentenceKey) -> bool {
        self.open.lock().unwrap().contains_key(key)
    }

    /// The stage of an open sentence.
    pub fn stage(&self, key: &SentenceKey) -> Option<Stage> {
        self.open.lock().unwrap().get(key).map(|e| e.stage)
    }

    /// Open sentences per language.
    pub fn open_counts(&self) -> HashMap<String, u64> {
        let mut out = HashMap::new();
        for key in self.open.lock().unwrap().keys() {
            *out.entry(key.target_language.clone()).or_default() += 1;
        }
        out
    }

    /// Closed sentences per language.
    pub fn counts(&self) -> HashMap<String, Counts> {
        self.counts.lock().unwrap().clone()
    }

    /// Sentences open longer than they should ever be.
    pub fn stuck(&self) -> Vec<Stuck> {
        let now = Instant::now();
        self.open
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, e)| now.duration_since(e.opened) >= self.stuck_after)
            .map(|(key, e)| Stuck { key: key.clone(), stage: e.stage, open_for: now.duration_since(e.opened) })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(text: &str, ts: &str) -> TranscriptionData {
        TranscriptionData::new(text, "u", ts).finalized()
    }

    #[test]
    fn a_sentence_opens_in_the_dubbed_languages_and_closes_once() {
        let ledger = SentenceLedger::new(TranslationTargets::new(["hi", "ta"]));
        ledger.open(&transcript("നാളെ", "1"), &["hi".into(), "en".into()]);
        let hi = SentenceKey::of_transcript(&transcript("നാളെ", "1"), "hi");
        let en = SentenceKey::of_transcript(&transcript("നാളെ", "1"), "en");
        assert!(ledger.is_open(&hi));
        assert!(!ledger.is_open(&en), "en is translated, not dubbed");
        ledger.advance(&hi, Stage::Playing);
        assert_eq!(ledger.stage(&hi), Some(Stage::Playing));
        assert!(ledger.close(&hi, Closed::Played));
        assert!(!ledger.close(&hi, Closed::Failed("late".into())), "the second close is a no-op");
        assert_eq!(ledger.counts()["hi"], Counts { played: 1, failed: 0, abandoned: 0 });
    }

    /// Music, noise, a pause: the STT answers with no words, and nothing is opened.
    #[test]
    fn a_turn_without_words_opens_nothing() {
        let ledger = SentenceLedger::new(TranslationTargets::new(["hi"])).with_stuck_after(Duration::ZERO);
        ledger.open(&transcript("", "1"), &["hi".into()]);
        ledger.open(&transcript("   ", "2"), &["hi".into()]);
        assert!(ledger.open_counts().is_empty());
        assert!(ledger.stuck().is_empty());
    }

    #[test]
    fn an_entry_left_open_is_stuck_with_its_stage() {
        let ledger = SentenceLedger::new(TranslationTargets::new(["hi"])).with_stuck_after(Duration::ZERO);
        ledger.open(&transcript("one", "1"), &["hi".into()]);
        let key = SentenceKey::of_transcript(&transcript("one", "1"), "hi");
        ledger.advance(&key, Stage::Speaking);
        let stuck = ledger.stuck();
        assert_eq!(stuck.len(), 1);
        assert_eq!((stuck[0].key.source_text.as_str(), stuck[0].stage), ("one", Stage::Speaking));
    }

    #[test]
    fn every_close_is_reported() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let ledger = SentenceLedger::new(TranslationTargets::new(["hi"]))
            .with_on_close(Arc::new(move |key, closed| sink.lock().unwrap().push((key.source_text.clone(), closed.clone()))));
        ledger.open(&transcript("one", "1"), &["hi".into()]);
        ledger.close(&SentenceKey::of_transcript(&transcript("one", "1"), "hi"), Closed::Abandoned("no listener".into()));
        assert_eq!(*seen.lock().unwrap(), [("one".to_string(), Closed::Abandoned("no listener".into()))]);
    }
}
