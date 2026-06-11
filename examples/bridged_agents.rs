//! Two bridged agents on a LocalAgentBus: an echo "brain" and a driver.
//!
//! Demonstrates:
//!   - `BaseAgent::bridged_pipeline` — pipelines bridged via `BusOutputEdge`
//!   - task dispatch with a named handler (`on_task`)
//!   - a streaming reply (`stream_start` / `stream_data` / `stream_end`)
//!   - task cancellation (`cancel_task` → terminal `Cancelled` response)
//!   - frames crossing the bridge (driver pipeline → brain pipeline)
//!
//! Run:
//!   cargo run --example bridged_agents

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use rustvani::agents::{
    AgentRunner, BaseAgent, BusOutputEdge, LocalAgentBus, TaskHandler, TaskRequestCtx, TaskStatus,
    TaskUpdate,
};
use rustvani::{system_clock, Frame, FrameDirection, FrameKind, PipelineParams, PipelineTask};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // ---- Brain: replies to "echo" with a streamed answer, and has a
    //      deliberately slow "think" handler we will cancel. ----
    let echo: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            ctx.stream_start(Some(json!("thinking..."))).await;
            for i in 1..=3 {
                ctx.stream_data(Some(json!(format!("chunk {i}")))).await;
            }
            ctx.stream_end(None).await;
            let payload = ctx.payload.clone();
            ctx.complete(TaskStatus::Completed, payload).await;
        })
    });
    let think: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(60)).await; // never finishes
            ctx.complete(TaskStatus::Completed, None).await;
        })
    });

    // The brain's edge excludes LLMText: the driver *sends* LLMText over
    // the bridge, so the brain must not re-publish it — on a two-way
    // bridge, each side excludes the frame kinds the peer produces or
    // re-injected frames ping-pong forever.
    let brain_edge = BusOutputEdge::with_exclude(
        "brain",
        vec!["driver".to_string()],
        HashSet::from([FrameKind::LLMText]),
    );
    let brain_task = PipelineTask::new(vec![brain_edge.to_processor()], PipelineParams::default());
    let brain = BaseAgent::new(
        "brain",
        brain_task,
        Some(vec!["driver".to_string()]), // accept bridged frames from driver
        true,
    )
    .with_output_edge(brain_edge)
    .on_task("echo", echo)
    .on_task("think", think);

    // Print every text frame that crosses the bridge into the brain.
    let mut filter = HashSet::new();
    filter.insert(FrameKind::LLMText);
    brain.pipeline().set_downstream_filter(filter);
    brain.pipeline().add_on_frame_reached_downstream(|frame| {
        Box::pin(async move {
            println!("[brain]  bridged frame arrived: {}", frame.name());
        })
    });
    let brain = Arc::new(brain);

    // ---- Driver: dispatches tasks to the brain. ----
    let driver = Arc::new(BaseAgent::bridged_pipeline(
        "driver",
        vec![],
        PipelineParams::default(),
        vec!["brain".to_string()],
        true,
    ));

    // ---- Runner ----
    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("demo", bus, system_clock()));
    runner.add_agent(brain).await?;
    runner.add_agent(driver.clone()).await?;

    let r = runner.clone();
    let run = tokio::spawn(async move { r.run().await });

    // Wait for both agents to be ready.
    while runner.registry().get("brain").await.is_none()
        || runner.registry().get("driver").await.is_none()
    {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let ctx = driver.task_ctx().expect("driver is set up");

    // 1. Dispatch + streaming reply.
    println!("[driver] dispatching 'echo'...");
    let handle = ctx
        .dispatch(
            "driver",
            "brain",
            Some("echo".into()),
            Some(json!("hello brain")),
        )
        .await?;
    let (updates, result) = handle.stream_updates(Some(Duration::from_secs(5))).await?;
    for u in &updates {
        match u {
            TaskUpdate::StreamStart { data } => println!("[driver] stream start: {data:?}"),
            TaskUpdate::StreamData { data } => println!("[driver] stream data:  {data:?}"),
            TaskUpdate::StreamEnd { .. } => println!("[driver] stream end"),
            _ => {}
        }
    }
    println!("[driver] result: {:?} {:?}", result.status, result.response);

    // 2. Dispatch + cancel.
    println!("[driver] dispatching 'think' and cancelling it...");
    let handle = ctx
        .dispatch("driver", "brain", Some("think".into()), None)
        .await?;
    let task_id = handle.task_id.clone();
    tokio::time::sleep(Duration::from_millis(100)).await;
    ctx.cancel_task("driver", "brain", task_id, Some("too slow".into()))
        .await;
    let result = handle
        .await_completion(Some(Duration::from_secs(5)))
        .await?;
    println!("[driver] cancelled task resolved as: {:?}", result.status);

    // 3. A frame pushed into the driver's pipeline crosses the bridge.
    driver
        .pipeline()
        .push_frame(
            Frame::llm_text("hello over the bridge".into()),
            FrameDirection::Downstream,
        )
        .await?;
    tokio::time::sleep(Duration::from_millis(300)).await;

    // ---- Shutdown ----
    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(10), run).await;
    println!("done.");
    Ok(())
}
