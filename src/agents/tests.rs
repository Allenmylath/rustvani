//! Cross-module integration tests for the agents subsystem:
//! task routing (Phase 2), ready-gating (Phase 3), bus edges (Phase 4),
//! and runner cascade (Phase 6).

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::clock::system_clock;
use crate::frames::{Frame, FrameKind};
use crate::pipeline::{PipelineParams, PipelineTask};

use super::base::{Agent, BaseAgent, TaskHandler, TaskRequestCtx};
use super::bus::{AgentBus, BusMessage, BusPayload, LocalAgentBus, TaskStatus};
use super::coordinator::{AgenticCoordinator, FenceOutcome};
use super::edges::BusOutputEdge;
use super::runner::AgentRunner;
use crate::context::{LLMContext, Message};

/// An empty pipeline (TaskSource → TaskSink) that runs until EndFrame.
fn idle_task() -> PipelineTask {
    PipelineTask::new(vec![], PipelineParams::default())
}

/// Spawn `runner.run()` and wait until all `names` are registered.
async fn start_and_wait(runner: &Arc<AgentRunner>, names: &[&str]) -> tokio::task::JoinHandle<()> {
    let r = runner.clone();
    let handle = tokio::spawn(async move {
        let _ = r.run().await;
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    'outer: loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "agents never became ready"
        );
        for name in names {
            if runner.registry().get(name).await.is_none() {
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue 'outer;
            }
        }
        break;
    }
    handle
}

fn echo_handler() -> TaskHandler {
    Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            let payload = ctx.payload.clone();
            ctx.complete(TaskStatus::Completed, payload).await;
        })
    })
}

/// Phase 2 test 1: dispatch → named handler → complete → await_completion.
#[tokio::test]
async fn task_round_trip() {
    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));
    let b = Arc::new(BaseAgent::new("b", idle_task(), None, true).on_task("echo", echo_handler()));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    let ctx = a.task_ctx().unwrap();
    let handle = ctx
        .dispatch("a", "b", Some("echo".into()), Some(json!({"x": 42})))
        .await
        .unwrap();
    let result = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();

    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.response, Some(json!({"x": 42})));

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test 2: StreamStart + 3×StreamData + StreamEnd then Response —
/// stream_updates yields 5 updates and the result.
#[tokio::test]
async fn task_streaming() {
    let streamer: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            ctx.stream_start(Some(json!("start"))).await;
            for i in 0..3 {
                ctx.stream_data(Some(json!(i))).await;
            }
            ctx.stream_end(Some(json!("end"))).await;
            ctx.complete(TaskStatus::Completed, Some(json!("done")))
                .await;
        })
    });

    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));
    let b = Arc::new(BaseAgent::new("b", idle_task(), None, true).on_task("stream", streamer));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    let ctx = a.task_ctx().unwrap();
    let handle = ctx
        .dispatch("a", "b", Some("stream".into()), None)
        .await
        .unwrap();
    let (updates, result) = handle
        .stream_updates(Some(Duration::from_secs(2)))
        .await
        .unwrap();

    assert_eq!(updates.len(), 5, "expected 5 stream updates: {updates:?}");
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.response, Some(json!("done")));

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test 3: TaskCancel aborts the handler and the requester
/// receives a terminal Cancelled response.
#[tokio::test]
async fn task_cancel() {
    let sleepy: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            ctx.complete(TaskStatus::Completed, None).await;
        })
    });

    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));
    let b = Arc::new(BaseAgent::new("b", idle_task(), None, true).on_task("sleep", sleepy));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    let ctx = a.task_ctx().unwrap();
    let handle = ctx
        .dispatch("a", "b", Some("sleep".into()), None)
        .await
        .unwrap();
    let task_id = handle.task_id.clone();

    // Give B a moment to start the handler, then cancel.
    tokio::time::sleep(Duration::from_millis(100)).await;
    ctx.cancel_task("a", "b", task_id, Some("changed my mind".into()))
        .await;

    let result = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();
    assert_eq!(result.status, TaskStatus::Cancelled);

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test 4: dispatch with an unknown handler name resolves Failed
/// instead of hanging.
#[tokio::test]
async fn task_no_handler_fails_fast() {
    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));
    let b = Arc::new(BaseAgent::new("b", idle_task(), None, true));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    let ctx = a.task_ctx().unwrap();
    let handle = ctx
        .dispatch("a", "b", Some("nonexistent".into()), None)
        .await
        .unwrap();
    let result = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();
    assert_eq!(result.status, TaskStatus::Failed);

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test 5: B ends while a job is active — A's pending handle
/// resolves (Cancelled or error) within 1s instead of hanging.
#[tokio::test]
async fn task_resolves_when_executor_dies() {
    let sleepy: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            ctx.complete(TaskStatus::Completed, None).await;
        })
    });

    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));
    let b = Arc::new(BaseAgent::new("b", idle_task(), None, true).on_task("sleep", sleepy));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    let ctx = a.task_ctx().unwrap();
    let handle = ctx
        .dispatch("a", "b", Some("sleep".into()), None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // End B directly while the job is in flight.
    b.end(Some("shutting down".into())).await.unwrap();

    let result = tokio::time::timeout(
        Duration::from_secs(1),
        handle.await_completion(Some(Duration::from_secs(1))),
    )
    .await
    .expect("handle did not resolve within 1s");
    // A closed-channel error is also acceptable.
    if let Ok(r) = result {
        assert_eq!(r.status, TaskStatus::Cancelled);
    }

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 3: dispatch to a target that is not ready yet waits (no polling)
/// and proceeds once the target announces readiness.
#[tokio::test]
async fn dispatch_waits_for_target_ready() {
    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus.clone(), system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a"]).await;

    // Dispatch to "late" before it exists. Spawn the dispatch, then bring
    // the agent up by registering it manually after a delay.
    let ctx = a.task_ctx().unwrap();
    let registry = runner.registry().clone();
    let register = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        registry
            .register(super::registry::AgentInfo {
                name: "late".into(),
                runner: "r".into(),
                parent: None,
                active: true,
                bridged: false,
                started_at: None,
            })
            .await;
    });

    let started = tokio::time::Instant::now();
    let handle = ctx
        .dispatch_with("a", "late", None, None, Some(Duration::from_secs(5)))
        .await
        .expect("dispatch should succeed once the target registers");
    assert!(
        started.elapsed() >= Duration::from_millis(150),
        "dispatch returned before the target was ready"
    );
    drop(handle);
    register.await.unwrap();

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 3: dispatch to a target that never becomes ready errors after the
/// ready timeout instead of silently dropping the request.
#[tokio::test]
async fn dispatch_times_out_when_target_never_ready() {
    let a = Arc::new(BaseAgent::new("a", idle_task(), None, true));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a"]).await;

    let ctx = a.task_ctx().unwrap();
    let result = ctx
        .dispatch_with("a", "ghost", None, None, Some(Duration::from_millis(200)))
        .await;
    assert!(result.is_err(), "dispatch to a missing agent should error");

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

// ---------------------------------------------------------------------------
// Phase 6 — runner parent/child cascade
// ---------------------------------------------------------------------------

/// Ending the runner stops the child before the parent's pipeline
/// finishes, and the runner exits within the timeout.
#[tokio::test]
async fn end_cascades_child_before_parent() {
    let finish_order: Arc<std::sync::Mutex<Vec<&'static str>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    let parent_task = idle_task();
    let order = finish_order.clone();
    parent_task.add_on_pipeline_finished(move |_f, _r| {
        let order = order.clone();
        Box::pin(async move {
            order.lock().unwrap().push("parent");
        })
    });

    let child_task = idle_task();
    let order = finish_order.clone();
    child_task.add_on_pipeline_finished(move |_f, _r| {
        let order = order.clone();
        Box::pin(async move {
            order.lock().unwrap().push("child");
        })
    });

    let parent = Arc::new(BaseAgent::new("parent", parent_task, None, true));
    let child = Arc::new(BaseAgent::new("child", child_task, None, true).with_parent("parent"));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(parent).await.unwrap();
    runner.add_agent(child).await.unwrap();
    let run = start_and_wait(&runner, &["parent", "child"]).await;

    runner.end(None).await;
    tokio::time::timeout(Duration::from_secs(8), run)
        .await
        .expect("runner did not exit within timeout")
        .unwrap();

    let order = finish_order.lock().unwrap().clone();
    assert_eq!(
        order,
        vec!["child", "parent"],
        "child must finish before the parent's pipeline"
    );
}

// ---------------------------------------------------------------------------
// Phase 4 — bus edge processors
// ---------------------------------------------------------------------------

/// Build a bridged agent whose pipeline is `[BusOutputEdge]`, counting
/// frames of `count_kind` that reach the pipeline's downstream boundary.
fn bridged_agent(
    name: &str,
    peers: Vec<String>,
    exclude: HashSet<FrameKind>,
    count_kind: FrameKind,
) -> (Arc<BaseAgent>, Arc<AtomicUsize>) {
    let edge = BusOutputEdge::with_exclude(name, peers.clone(), exclude);
    let task = PipelineTask::new(vec![edge.to_processor()], PipelineParams::default());

    let mut filter = HashSet::new();
    filter.insert(count_kind);
    task.set_downstream_filter(filter);

    let count = Arc::new(AtomicUsize::new(0));
    let count_cb = count.clone();
    task.add_on_frame_reached_downstream(move |_frame| {
        let count = count_cb.clone();
        Box::pin(async move {
            count.fetch_add(1, Ordering::SeqCst);
        })
    });

    let agent = Arc::new(BaseAgent::new(name, task, Some(peers), true).with_output_edge(edge));
    (agent, count)
}

/// Phase 4 test 1: two pipelines bridged via output edges — a frame pushed
/// into A arrives in B's pipeline, and a frame published by B targeted at
/// A arrives at A's head. Mutual exclusions break the re-publish loop.
#[tokio::test]
async fn bridged_pipelines_exchange_frames() {
    // A publishes to b, never re-publishes Transcription (B's output kind).
    let (a, a_transcripts) = bridged_agent(
        "a",
        vec!["b".into()],
        HashSet::from([FrameKind::Transcription]),
        FrameKind::Transcription,
    );
    // B publishes to a, never re-publishes LLMText (A's output kind).
    let (b, b_texts) = bridged_agent(
        "b",
        vec!["a".into()],
        HashSet::from([FrameKind::LLMText]),
        FrameKind::LLMText,
    );

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    // A → B: text frame pushed into A's pipeline reaches B's processor.
    a.pipeline()
        .push_frame(
            Frame::llm_text("hello b".into()),
            crate::frames::FrameDirection::Downstream,
        )
        .await
        .unwrap();

    // B → A: transcription pushed into B's pipeline reaches A's head.
    b.pipeline()
        .push_frame(
            Frame::transcription(crate::frames::TranscriptionData::new(
                "hello a", "user", "now",
            )),
            crate::frames::FrameDirection::Downstream,
        )
        .await
        .unwrap();

    let ok = async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            if b_texts.load(Ordering::SeqCst) >= 1 && a_transcripts.load(Ordering::SeqCst) >= 1 {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }
    .await;
    assert!(
        ok,
        "bridge failed: b_texts={}, a_transcripts={}",
        b_texts.load(Ordering::SeqCst),
        a_transcripts.load(Ordering::SeqCst)
    );

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 4 test 2: deactivating A keeps frames flowing inside A's pipeline
/// but publishes nothing; reactivating resumes the flow.
#[tokio::test]
async fn edge_activation_gating() {
    // A counts its own LLMText locally to prove local flow continues.
    let (a, a_local) = bridged_agent("a", vec!["b".into()], HashSet::new(), FrameKind::LLMText);
    let (b, b_texts) = bridged_agent(
        "b",
        vec![],
        HashSet::from([FrameKind::LLMText]),
        FrameKind::LLMText,
    );

    let bus: Arc<dyn AgentBus> = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus.clone(), system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    // Deactivate A, push a frame: local flow continues, nothing reaches B.
    bus.send(BusMessage::new(
        "r",
        Some("a".into()),
        BusPayload::Deactivate,
    ))
    .await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while a.active() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(!a.active(), "Deactivate never took effect");

    a.pipeline()
        .push_frame(
            Frame::llm_text("while inactive".into()),
            crate::frames::FrameDirection::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(a_local.load(Ordering::SeqCst), 1, "local flow stalled");
    assert_eq!(
        b_texts.load(Ordering::SeqCst),
        0,
        "published while inactive"
    );

    // Reactivate and push again: flow to B resumes.
    bus.send(BusMessage::new(
        "r",
        Some("a".into()),
        BusPayload::Activate { args: None },
    ))
    .await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while !a.active() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    a.pipeline()
        .push_frame(
            Frame::llm_text("while active".into()),
            crate::frames::FrameDirection::Downstream,
        )
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while b_texts.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(b_texts.load(Ordering::SeqCst), 1, "flow did not resume");
    assert_eq!(a_local.load(Ordering::SeqCst), 2);

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 4 test 3: excluded frame kinds are forwarded locally but never
/// published.
#[tokio::test]
async fn edge_exclusion() {
    let (a, a_local) = bridged_agent(
        "a",
        vec!["b".into()],
        HashSet::from([FrameKind::LLMText]),
        FrameKind::LLMText,
    );
    let (b, b_texts) = bridged_agent("b", vec![], HashSet::new(), FrameKind::LLMText);

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(a.clone()).await.unwrap();
    runner.add_agent(b.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["a", "b"]).await;

    a.pipeline()
        .push_frame(
            Frame::llm_text("excluded".into()),
            crate::frames::FrameDirection::Downstream,
        )
        .await
        .unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while a_local.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(a_local.load(Ordering::SeqCst), 1, "local forward failed");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        b_texts.load(Ordering::SeqCst),
        0,
        "excluded frame published"
    );

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test 6: bridged peer filter — frames from a listed source are
/// injected, frames from others are ignored.
#[tokio::test]
async fn bridged_peer_filter() {
    let task = idle_task();
    let mut filter = HashSet::new();
    filter.insert(FrameKind::LLMText);
    task.set_downstream_filter(filter);

    let received = Arc::new(AtomicUsize::new(0));
    let received_cb = received.clone();
    task.add_on_frame_reached_downstream(move |_frame| {
        let received = received_cb.clone();
        Box::pin(async move {
            received.fetch_add(1, Ordering::SeqCst);
        })
    });

    let agent = Arc::new(BaseAgent::new(
        "listener",
        task,
        Some(vec!["voice".to_string()]),
        true,
    ));

    let bus: Arc<dyn AgentBus> = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus.clone(), system_clock()));
    runner.add_agent(agent.clone()).await.unwrap();
    let run = start_and_wait(&runner, &["listener"]).await;

    // Frame from "voice" — accepted.
    bus.send(BusMessage::new(
        "voice",
        Some("listener".into()),
        BusPayload::Frame {
            frame: Frame::llm_text("hello".into()),
            direction: crate::frames::FrameDirection::Downstream,
        },
    ))
    .await;
    // Frame from "other" — ignored.
    bus.send(BusMessage::new(
        "other",
        Some("listener".into()),
        BusPayload::Frame {
            frame: Frame::llm_text("spam".into()),
            direction: crate::frames::FrameDirection::Downstream,
        },
    ))
    .await;

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        received.load(Ordering::SeqCst),
        1,
        "only the frame from 'voice' should be injected"
    );

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

// ===========================================================================
// Phase 2: turn-level ACID across the bus — epoch fencing + coordinator.
// See doc/turn-acid-phase2.md.
// ===========================================================================

/// A worker that echoes the turn epoch it observed back to the requester,
/// so a test can prove the fence threaded `turn_epoch` end-to-end.
fn epoch_reporting_handler() -> TaskHandler {
    Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            let saw = ctx.turn_epoch;
            ctx.complete(TaskStatus::Completed, Some(json!({ "saw_epoch": saw })))
                .await;
        })
    })
}

fn shared_ctx() -> Arc<std::sync::Mutex<LLMContext>> {
    Arc::new(std::sync::Mutex::new(LLMContext::new(None)))
}

/// Phase 2 unit: the bus message builder stamps and clears the turn epoch.
#[test]
fn bus_message_carries_turn_epoch() {
    let plain = BusMessage::new("a", Some("b".into()), BusPayload::Deactivate);
    assert_eq!(plain.turn_epoch, None);
    let fenced = plain.with_turn_epoch(7);
    assert_eq!(fenced.turn_epoch, Some(7));
}

/// Phase 2 test: `dispatch_fenced` stamps the epoch on the request and the
/// executor sees it on `TaskRequestCtx` and echoes it back.
#[tokio::test]
async fn fenced_dispatch_threads_epoch_to_executor() {
    let coord = Arc::new(BaseAgent::new("coord", idle_task(), None, true));
    let worker = Arc::new(
        BaseAgent::new("worker", idle_task(), None, true)
            .on_task("report", epoch_reporting_handler()),
    );

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(coord.clone()).await.unwrap();
    runner.add_agent(worker).await.unwrap();
    let run = start_and_wait(&runner, &["coord", "worker"]).await;

    let context = shared_ctx();
    // Simulate the user aggregator opening a turn (epoch 0 -> 1).
    let epoch = context.lock().unwrap().begin_turn();
    let coordinator = AgenticCoordinator::new("coord", context.clone(), coord.task_ctx().unwrap());
    assert_eq!(coordinator.open_turn(), epoch);

    let handle = coordinator
        .dispatch("worker", Some("report".into()), None)
        .await
        .unwrap();
    let result = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();

    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.response, Some(json!({ "saw_epoch": epoch })));

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test: a result for the current epoch is staged and commits into
/// the shared context.
#[tokio::test]
async fn current_result_is_staged_and_committed() {
    let coord = Arc::new(BaseAgent::new("coord", idle_task(), None, true));
    let worker = Arc::new(
        BaseAgent::new("worker", idle_task(), None, true)
            .on_task("report", epoch_reporting_handler()),
    );

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(coord.clone()).await.unwrap();
    runner.add_agent(worker).await.unwrap();
    let run = start_and_wait(&runner, &["coord", "worker"]).await;

    let context = shared_ctx();
    context.lock().unwrap().begin_turn();
    let coordinator = AgenticCoordinator::new("coord", context.clone(), coord.task_ctx().unwrap());
    coordinator.open_turn();

    let handle = coordinator
        .dispatch("worker", Some("report".into()), None)
        .await
        .unwrap();
    let tid = handle.task_id.clone();
    let _ = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();

    // The coordinator integrates the worker's answer as an assistant turn.
    let outcome = coordinator.stage_result(
        &tid,
        Message::Assistant {
            content: Some("the answer is 42".into()),
            tool_calls: None,
        },
    );
    assert_eq!(outcome, FenceOutcome::Staged);
    assert_eq!(context.lock().unwrap().staged_len(), 1);

    let committed = coordinator.commit();
    assert_eq!(committed, 1);
    assert_eq!(context.lock().unwrap().messages.len(), 1);
    assert_eq!(context.lock().unwrap().staged_len(), 0);

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test (the load-bearing one): a result whose turn was superseded by
/// a barge-in is quarantined and never reaches the context.
#[tokio::test]
async fn stale_result_is_quarantined() {
    let coord = Arc::new(BaseAgent::new("coord", idle_task(), None, true));
    let worker = Arc::new(
        BaseAgent::new("worker", idle_task(), None, true)
            .on_task("report", epoch_reporting_handler()),
    );

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(coord.clone()).await.unwrap();
    runner.add_agent(worker).await.unwrap();
    let run = start_and_wait(&runner, &["coord", "worker"]).await;

    let context = shared_ctx();
    context.lock().unwrap().begin_turn(); // turn N
    let coordinator = AgenticCoordinator::new("coord", context.clone(), coord.task_ctx().unwrap());
    coordinator.open_turn();

    // Dispatch turn N's work, then await its (slow) answer.
    let handle = coordinator
        .dispatch("worker", Some("report".into()), None)
        .await
        .unwrap();
    let tid = handle.task_id.clone();
    let _ = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();

    // Barge-in: a new turn opens (epoch N -> N+1) before we stage turn N's answer.
    context.lock().unwrap().begin_turn();
    coordinator.open_turn();

    let outcome = coordinator.stage_result(
        &tid,
        Message::Assistant {
            content: Some("stale answer from turn N".into()),
            tool_calls: None,
        },
    );
    assert_eq!(outcome, FenceOutcome::Quarantined);
    assert_eq!(context.lock().unwrap().staged_len(), 0);
    assert_eq!(coordinator.commit(), 0);
    assert!(
        context.lock().unwrap().messages.is_empty(),
        "stale turn-N answer must not contaminate turn N+1"
    );

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}

/// Phase 2 test: on barge-in the coordinator cancels in-flight workers and
/// rolls back staged context; the worker resolves Cancelled.
#[tokio::test]
async fn interruption_cancels_and_rolls_back() {
    let sleepy: TaskHandler = Arc::new(|ctx: TaskRequestCtx| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            ctx.complete(TaskStatus::Completed, None).await;
        })
    });

    let coord = Arc::new(BaseAgent::new("coord", idle_task(), None, true));
    let worker =
        Arc::new(BaseAgent::new("worker", idle_task(), None, true).on_task("sleep", sleepy));

    let bus = Arc::new(LocalAgentBus::new());
    let runner = Arc::new(AgentRunner::new("r", bus, system_clock()));
    runner.add_agent(coord.clone()).await.unwrap();
    runner.add_agent(worker).await.unwrap();
    let run = start_and_wait(&runner, &["coord", "worker"]).await;

    let context = shared_ctx();
    context.lock().unwrap().begin_turn();
    let coordinator = AgenticCoordinator::new("coord", context.clone(), coord.task_ctx().unwrap());
    coordinator.open_turn();

    // Pre-stage a half-written round, then dispatch slow work.
    context.lock().unwrap().stage_message(Message::Assistant {
        content: Some("partial".into()),
        tool_calls: None,
    });
    let handle = coordinator
        .dispatch("worker", Some("sleep".into()), None)
        .await
        .unwrap();
    assert_eq!(coordinator.in_flight(), 1);

    // Give the worker a moment to start, then barge in.
    tokio::time::sleep(Duration::from_millis(100)).await;
    coordinator
        .on_interruption(Some("user barged in".into()))
        .await;

    // Staging rolled back, nothing in flight.
    assert_eq!(context.lock().unwrap().staged_len(), 0);
    assert_eq!(coordinator.in_flight(), 0);

    // The worker received TaskCancel and resolved Cancelled.
    let result = handle
        .await_completion(Some(Duration::from_secs(2)))
        .await
        .unwrap();
    assert_eq!(result.status, TaskStatus::Cancelled);

    runner.end(None).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
}
