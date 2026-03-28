use crate::frames::Frame;

// ---------------------------------------------------------------------------
// Token usage (for LLM processors)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct LLMTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// FrameProcessorMetrics
//
// Full interface stub — timing hooks are no-ops here.
// Production use: replace the no-op bodies with real instrumentation.
// ---------------------------------------------------------------------------

pub struct FrameProcessorMetrics {
    processor_name: std::sync::Mutex<String>,
}

impl Default for FrameProcessorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameProcessorMetrics {
    pub fn new() -> Self {
        Self {
            processor_name: std::sync::Mutex::new(String::new()),
        }
    }

    pub fn set_processor_name(&self, name: &str) {
        *self.processor_name.lock().unwrap() = name.to_owned();
    }

    // ---- TTFB (Time To First Byte) ----

    pub async fn start_ttfb_metrics(
        &self,
        _start_time: Option<f64>,
        _report_only_initial: bool,
    ) {
    }

    /// Returns `Some(MetricsFrame)` if ready to push, else `None`.
    pub async fn stop_ttfb_metrics(&self, _end_time: Option<f64>) -> Option<Frame> {
        None
    }

    // ---- Processing time ----

    pub async fn start_processing_metrics(&self, _start_time: Option<f64>) {}

    pub async fn stop_processing_metrics(&self, _end_time: Option<f64>) -> Option<Frame> {
        None
    }

    // ---- LLM token usage ----

    pub async fn start_llm_usage_metrics(&self, _tokens: &LLMTokenUsage) -> Option<Frame> {
        None
    }

    // ---- TTS usage ----

    pub async fn start_tts_usage_metrics(&self, _text: &str) -> Option<Frame> {
        None
    }

    // ---- Text aggregation timing ----

    pub async fn start_text_aggregation_metrics(&self) {}

    pub async fn stop_text_aggregation_metrics(&self) -> Option<Frame> {
        None
    }
}
