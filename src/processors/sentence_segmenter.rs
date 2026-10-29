//! Sentences out of partial transcripts, before the turn ends.
//!
//! A realtime STT reports the current segment's hypothesis as it grows (`TranscriptionFrame`s
//! with `finalized: false`) and the segment's final text when the turn ends or is cut. Anything
//! that waits for finals (translation, dubbing, an LLM) therefore waits for the pause and the
//! STT's finalisation, about two seconds after the last word. The segmenter finalises earlier:
//! once a partial shows a sentence end with a few characters of the next sentence after it (so
//! the boundary is unlikely to be revised), the sentence goes downstream as a final
//! `TranscriptionFrame` of its own, the partial is forwarded with only the remainder, and when the
//! segment's real final arrives the part already finalised is stripped from it. Interims from
//! the STT must be on ([`InterimPolicy::Emit`](crate::services::InterimPolicy)).
//!
//! ```text
//!   partial "Good morning everyone. This"        →  final "Good morning everyone."  +  partial "This"
//!   partial "Good morning everyone. This is a test."   →  partial "This is a test."
//!   final   "Good morning everyone. This is a test."   →  final "This is a test."
//! ```
//!
//! A hypothesis the STT revises in front of the committed text (rare: the sentence end was
//! already followed by more words) is taken up from the longest run of words it still shares.

use async_trait::async_trait;
use std::sync::Mutex;

use crate::error::Result;
use crate::frames::{
    DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor, SystemFrame,
    TranscriptionData,
};
use crate::utils::sentence_splitter::find_sentence_end;

#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// Characters of the next sentence that must follow a sentence end before it counts.
    pub min_tail_chars: usize,
    /// Shortest sentence finalised on its own; a shorter one waits for the next end.
    pub min_sentence_chars: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self { min_tail_chars: 3, min_sentence_chars: 6 }
    }
}

/// What of the current segment has already gone downstream as finals (as the partial spelt it).
#[derive(Default)]
struct State {
    committed: String,
}

pub struct SentenceSegmenter {
    config: SegmenterConfig,
    state: Mutex<State>,
}

impl SentenceSegmenter {
    pub fn new(config: SegmenterConfig) -> Self {
        Self { config, state: Mutex::new(State::default()) }
    }

    pub fn into_processor(self) -> FrameProcessor {
        FrameProcessor::new("SentenceSegmenter", Box::new(self), false)
    }

    /// `text` with what was committed stripped: the exact prefix when the hypothesis kept it,
    /// else what follows the longest run of words both share. Resets nothing.
    fn remainder<'a>(committed: &str, text: &'a str) -> &'a str {
        if committed.is_empty() {
            return text;
        }
        if let Some(rest) = text.strip_prefix(committed) {
            return rest;
        }
        // Words compared without their punctuation and case: a revision often only changes a
        // comma to a full stop, and what was finalised must not go out twice.
        let core = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let mut shared = 0;
        for (a, b) in committed.split_whitespace().zip(text.split_whitespace()) {
            if core(a) != core(b) {
                break;
            }
            shared += 1;
        }
        let mut words = text.split_whitespace();
        let mut end = 0;
        for _ in 0..shared {
            if let Some(w) = words.next() {
                end = w.as_ptr() as usize - text.as_ptr() as usize + w.len();
            }
        }
        &text[end..]
    }

    /// The byte end (past the punctuation), within `rest`, of the last sentence followed by at
    /// least `min_tail_chars` of the next.
    fn stable_boundary(&self, rest: &str) -> Option<usize> {
        let mut from = 0;
        let mut last = None;
        while from < rest.len() {
            let Some(end) = find_sentence_end(&rest[from..]) else { break };
            // `end` is the last byte of the closing character (all three of a danda's), not its start.
            let after = from + end + 1;
            let sentence = rest[..after].trim();
            let tail = rest[after..].trim();
            if tail.chars().count() >= self.config.min_tail_chars
                && sentence.chars().count() >= self.config.min_sentence_chars
            {
                last = Some(after);
            }
            from = after;
        }
        last
    }
}

#[async_trait]
impl FrameHandler for SentenceSegmenter {
    async fn on_process_frame(&self, processor: &FrameProcessor, frame: Frame, direction: FrameDirection) -> Result<()> {
        match &frame.inner {
            FrameInner::Data(DataFrame::Transcription(t)) if !t.finalized => {
                let (finals, partial) = {
                    let mut state = self.state.lock().unwrap();
                    let rest = Self::remainder(&state.committed, &t.text);
                    let rest_start = t.text.len() - rest.len();
                    let mut finals = Vec::new();
                    let partial_text = match self.stable_boundary(rest) {
                        Some(at) => {
                            finals.push(rest[..at].trim().to_string());
                            state.committed = t.text[..rest_start + at].to_string();
                            rest[at..].trim().to_string()
                        }
                        None => rest.trim().to_string(),
                    };
                    (finals, partial_text)
                };
                for text in finals {
                    let mut data = TranscriptionData::new(text, t.user_id.clone(), t.timestamp.clone());
                    data.language = t.language.clone();
                    data.finalized = true;
                    processor.push_frame(Frame::transcription(data), direction).await?;
                }
                if partial != t.text {
                    let mut data = t.clone();
                    data.text = partial;
                    processor.push_frame(Frame::transcription(data), direction).await?;
                } else {
                    processor.push_frame(frame, direction).await?;
                }
            }
            FrameInner::Data(DataFrame::Transcription(t)) => {
                let rest = {
                    let mut state = self.state.lock().unwrap();
                    let rest = Self::remainder(&state.committed, &t.text).trim().to_string();
                    state.committed.clear();
                    rest
                };
                if rest == t.text {
                    processor.push_frame(frame, direction).await?;
                } else if !rest.is_empty() {
                    let mut data = t.clone();
                    data.text = rest;
                    processor.push_frame(Frame::transcription(data), direction).await?;
                }
                // Nothing new in it: already finalised sentence by sentence.
            }
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { stop_secs, timestamp, transcript }) => {
                let committed = std::mem::take(&mut self.state.lock().unwrap().committed);
                match transcript {
                    Some(t) if !committed.is_empty() => {
                        let rest = Self::remainder(&committed, &t.text).trim().to_string();
                        let transcript = (!rest.is_empty()).then(|| {
                            let mut data = t.clone();
                            data.text = rest;
                            data
                        });
                        let stop = Frame {
                            id: frame.id,
                            sibling_id: frame.sibling_id,
                            inner: FrameInner::System(SystemFrame::VADUserStoppedSpeaking {
                                stop_secs: *stop_secs,
                                timestamp: *timestamp,
                                transcript,
                            }),
                        };
                        processor.push_frame(stop, direction).await?;
                    }
                    _ => processor.push_frame(frame, direction).await?,
                }
            }
            FrameInner::System(SystemFrame::Interruption) => {
                self.state.lock().unwrap().committed.clear();
                processor.push_frame(frame, direction).await?;
            }
            _ => processor.push_frame(frame, direction).await?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{PassthroughHandler, StartFrameData};
    use std::sync::Arc;

    async fn harness() -> (FrameProcessor, Arc<Mutex<Vec<(String, bool, bool)>>>) {
        let proc = FrameProcessor::new("seg-test", Box::new(SentenceSegmenter::new(SegmenterConfig::default())), true);
        proc.process_frame(Frame::start(StartFrameData::default()), FrameDirection::Downstream).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let seen = out.clone();
        proc.on_before_push_frame(move |frame| match &frame.inner {
            FrameInner::Data(DataFrame::Transcription(t)) => seen.lock().unwrap().push((t.text.clone(), t.finalized, false)),
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript, .. }) => {
                seen.lock().unwrap().push((transcript.as_ref().map(|t| t.text.clone()).unwrap_or_default(), true, true))
            }
            _ => {}
        });
        let _ = PassthroughHandler;
        (proc, out)
    }

    fn partial(text: &str) -> Frame {
        Frame::transcription(TranscriptionData::new(text, "u", "1").with_language("en-IN"))
    }
    fn final_(text: &str) -> Frame {
        Frame::transcription(TranscriptionData::new(text, "u", "2").with_language("en-IN").finalized())
    }

    #[tokio::test]
    async fn sentences_are_finalised_from_partials_and_stripped_from_the_final() {
        let (proc, out) = harness().await;
        for p in ["Good", "Good morning everyone.", "Good morning everyone. Th", "Good morning everyone. This is", "Good morning everyone. This is a test. We f"] {
            proc.process_frame(partial(p), FrameDirection::Downstream).await.unwrap();
        }
        proc.process_frame(final_("Good morning everyone. This is a test. We fly."), FrameDirection::Downstream).await.unwrap();
        let got = out.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                ("Good".to_string(), false, false),
                ("Good morning everyone.".to_string(), false, false),    // no tail yet: not stable
                ("Good morning everyone. Th".to_string(), false, false), // two characters of tail: still not
                ("Good morning everyone.".to_string(), true, false),     // "This is" followed it
                ("This is".to_string(), false, false),
                ("This is a test.".to_string(), true, false),
                ("We f".to_string(), false, false),
                ("We fly.".to_string(), true, false),
            ]
        );
    }

    #[tokio::test]
    async fn a_final_with_nothing_new_is_dropped_and_the_bundled_stop_is_stripped() {
        let (proc, out) = harness().await;
        proc.process_frame(partial("One sentence here. And"), FrameDirection::Downstream).await.unwrap();
        proc.process_frame(final_("One sentence here."), FrameDirection::Downstream).await.unwrap();
        // A new segment: nothing committed, a final passes through whole.
        proc.process_frame(final_("Second segment."), FrameDirection::Downstream).await.unwrap();
        // A turn whose closing transcript rides the stop frame.
        proc.process_frame(partial("Third one ends here. More"), FrameDirection::Downstream).await.unwrap();
        let stop = Frame::vad_user_stopped_speaking(0.8, 1.0).with_vad_stop_transcript(
            TranscriptionData::new("Third one ends here. More words.", "u", "3").finalized(),
        );
        proc.process_frame(stop, FrameDirection::Downstream).await.unwrap();
        let got = out.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                ("One sentence here.".to_string(), true, false),
                ("And".to_string(), false, false),
                ("Second segment.".to_string(), true, false),
                ("Third one ends here.".to_string(), true, false),
                ("More".to_string(), false, false),
                ("More words.".to_string(), true, true),
            ]
        );
    }

    #[tokio::test]
    async fn short_sentences_wait_abbreviations_do_not_count_and_revisions_are_taken_up() {
        let (proc, out) = harness().await;
        proc.process_frame(partial("Hi. Dr. Rao said the"), FrameDirection::Downstream).await.unwrap();
        // "Hi." is too short to stand alone; "Dr." is an abbreviation: nothing finalised.
        assert_eq!(out.lock().unwrap().clone(), vec![("Hi. Dr. Rao said the".to_string(), false, false)]);
        proc.process_frame(partial("Hi. Dr. Rao said the tide is in. Boats"), FrameDirection::Downstream).await.unwrap();
        // The STT revises the start of the next sentence: the committed words are still skipped.
        proc.process_frame(partial("Hi. Dr. Rao said the tide is in. Boats return now"), FrameDirection::Downstream).await.unwrap();
        proc.process_frame(partial("Hi, Dr. Rao said the tide is in. Both return now."), FrameDirection::Downstream).await.unwrap();
        let got = out.lock().unwrap().clone();
        assert_eq!(got[1], ("Hi. Dr. Rao said the tide is in.".to_string(), true, false));
        assert_eq!(got[2], ("Boats".to_string(), false, false));
        assert_eq!(got[3], ("Boats return now".to_string(), false, false));
        assert_eq!(got[4], ("Both return now.".to_string(), false, false), "revised to 'Hi,': the finalised words are still skipped");
    }

    /// A danda (`।`, three bytes) ends a Hindi sentence: the cut goes after the whole of it
    /// (0.4.0-dev.19 cut two bytes into it and panicked, poisoning the segmenter's lock).
    #[tokio::test]
    async fn a_danda_ends_a_sentence_without_cutting_into_it() {
        let (proc, out) = harness().await;
        for p in ["।", "बहुत अच्छा है।", "बहुत अच्छा है। अगला", "बहुत अच्छा है। अगला वाक्य"] {
            proc.process_frame(partial(p), FrameDirection::Downstream).await.unwrap();
        }
        let got = out.lock().unwrap().clone();
        assert!(got.iter().any(|(t, fin, _)| *fin && t == "बहुत अच्छा है।"), "{got:?}");
    }

    #[test]
    fn remainder_strips_the_exact_prefix_or_the_shared_words() {
        assert_eq!(SentenceSegmenter::remainder("", "abc"), "abc");
        assert_eq!(SentenceSegmenter::remainder("Good morning.", "Good morning. Hello"), " Hello");
        assert_eq!(SentenceSegmenter::remainder("Good morning.", "Good day. Hello"), " day. Hello");
        assert_eq!(SentenceSegmenter::remainder("Good morning.", "good morning, hello"), " hello");
        assert_eq!(SentenceSegmenter::remainder("Good morning.", "Bad day"), "Bad day");
    }
}
