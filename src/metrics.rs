use crate::frames::Frame;

#[derive(Debug, Clone)]
pub struct MetricsData {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LLMTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct FrameProcessorMetrics;

impl FrameProcessorMetrics {
    pub fn new() -> Self { Self }
    pub fn set_processor_name(&self, _name: &str) {}

    pub async fn start_ttfb_metrics(&self, _start: Option<f64>, _initial_only: bool) {}
    pub async fn stop_ttfb_metrics(&self, _end: Option<f64>) -> Option<Frame> { None }
    pub async fn start_processing_metrics(&self, _start: Option<f64>) {}
    pub async fn stop_processing_metrics(&self, _end: Option<f64>) -> Option<Frame> { None }
    pub async fn start_llm_usage_metrics(&self, _tokens: &LLMTokenUsage) -> Option<Frame> { None }
    pub async fn start_tts_usage_metrics(&self, _text: &str) -> Option<Frame> { None }
    pub async fn start_text_aggregation_metrics(&self) {}
    pub async fn stop_text_aggregation_metrics(&self) -> Option<Frame> { None }
}
