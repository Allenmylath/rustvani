/// RAVI — **R**eal-time **A**udio **Video **I**nterface
///
/// Rust-native protocol layer for rustvani, spiritually equivalent to
/// Daily's RTVI but named for the sun (रवि) to pair with वाणी (voice).
///
/// # Module layout
///
/// | Module       | Contents                                                  |
/// |--------------|-----------------------------------------------------------|
/// | `models`     | Protocol constants, inbound deserialisers, outbound builders |
/// | `observer`   | `RaviObserver` — watches frames, emits RAVI messages      |
/// | `processor`  | `RaviProcessor` / `RaviHandler` — handles inbound client msgs |
///
/// # Quick start
///
/// ```rust
/// use rustvani::ravi::{
///     processor::{RaviProcessor, RaviParams},
///     observer::RaviObserverParams,
/// };
///
/// // 1. Create the processor (goes in your pipeline).
/// let ravi = RaviProcessor::new(RaviParams::default());
///
/// // 2. Create the observer (attach to your PipelineTask).
/// let observer = RaviProcessor::create_observer(&ravi, RaviObserverParams::default());
///
/// // 3. Wire into pipeline: [Input] → [ravi] → [...] → [Output]
/// ```
pub mod models;
pub mod observer;
pub mod processor;

pub use observer::{RaviObserver, RaviObserverParams};
pub use processor::{RaviParams, RaviProcessor};
