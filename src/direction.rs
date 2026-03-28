/// Direction of frame flow in the processing pipeline.
///
/// - `Downstream`: Frames flowing from input toward output (forward through the pipeline).
/// - `Upstream`:   Frames flowing back from output toward input (backward through the pipeline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDirection {
    Downstream,
    Upstream,
}
