use std::sync::Arc;
use std::time::Instant;

use async_stream::stream;
use futures::StreamExt;
use ufox_llm::{ContentPart, FinishReason, Message, Role, ToolCall};

use crate::agent::Agent;
use crate::agent::execution::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};
use crate::agent::runner::{
    LoopCtx, LoopResult, MemoryCtx, append_tool_result_messages, build_request, execute_tools,
    run_loop,
};
use crate::error::ArcError;
use crate::memory::strategy;
use crate::thread::{Thread, ThreadId, ThreadShared, UserId};

use super::model::{RunEvent, RunEventStream, RunId, RunRequest, RunResult, RunTrace};

const MEMORY_CONTEXT_MESSAGE_NAME: &str = "memory_context";

pub async fn run_once(
    agent: &Agent,
    thread: &Thread,
    request: RunRequest,
) -> Result<RunResult, ArcError> {
    let wall_start = Instant::now();
    let run_id = RunId::new();
    thread.try_start_run().await?;
    let effective_config = build_effective_config(agent, &request);
    let user_message = request.input.into_message();

    let loop_result = async {
        let mut messages = thread.shared.messages.lock().await;
        messages.push(user_message);
        run_loop(
            LoopCtx {
                llm: &agent.llm,
                instructions: agent.instructions.as_deref(),
                config: &effective_config,
                tools: agent.tools.as_deref(),
                memory: agent.memory.as_deref().map(|store| MemoryCtx {
                    store,
                    user_id: &thread.user_id,
                    thread_id: &thread.thread_id,
                }),
                interrupt: agent
                    .interrupt_handler
                    .as_deref()
                    .map(|h| (h, &thread.user_id, &thread.thread_id)),
            },
            &mut messages,
        )
        .await
    }
    .await;

    thread.finish_run().await;

    let LoopResult {
        steps,
        final_response,
        state,
        total_usage,
    } = loop_result?;

    let trace = RunTrace {
        run_id: run_id.clone(),
        user_id: thread.user_id.clone(),
        thread_id: thread.thread_id.clone(),
        steps,
        state,
        total_duration: wall_start.elapsed(),
        total_usage,
    };

    Ok(RunResult {
        run_id,
        user_id: thread.user_id.clone(),
        thread_id: thread.thread_id.clone(),
        response: final_response,
        trace,
    })
}

pub async fn run_stream(
    agent: &Agent,
    thread: &Thread,
    request: RunRequest,
) -> Result<RunEventStream, ArcError> {
    let run_id = RunId::new();
    thread.try_start_run().await?;
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<RunEvent, ArcError>>(32);
    let shared = Arc::clone(&thread.shared);
    let agent = agent.clone();
    let user_id = thread.user_id.clone();
    let thread_id = thread.thread_id.clone();
    let effective_config = build_effective_config(&agent, &request);
    let user_message = request.input.into_message();
    thread.shared.messages.lock().await.push(user_message);
    let run_id_for_task = run_id.clone();

    tokio::spawn(async move {
        if let Err(err) = run_streaming_loop(
            agent,
            shared.clone(),
            user_id.clone(),
            thread_id.clone(),
            run_id_for_task,
            effective_config,
            &tx,
        )
        .await
        {
            let _ = tx.send(Err(err)).await;
        }
        *shared.state.lock().await = crate::thread::ThreadState::Idle;
    });

    let event_stream = stream! {
        let mut rx = rx;
        while let Some(item) = rx.recv().await {
            yield item;
        }
    };

    Ok(RunEventStream {
        inner: Box::pin(event_stream),
    })
}

fn build_effective_config(agent: &Agent, request: &RunRequest) -> crate::agent::AgentConfig {
    let mut config = agent.config.clone();
    if let Some(value) = request.temperature {
        config.temperature = Some(value);
    }
    if let Some(value) = request.max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = request.timeout {
        config.timeout = value;
    }
    config
}

async fn run_streaming_loop(
    agent: Agent,
    shared: Arc<ThreadShared>,
    user_id: UserId,
    thread_id: ThreadId,
    run_id: RunId,
    config: crate::agent::AgentConfig,
    tx: &tokio::sync::mpsc::Sender<Result<RunEvent, ArcError>>,
) -> Result<(), ArcError> {
    let deadline = Instant::now() + config.timeout;

    if config.enable_perceive
        && let Some(store) = agent.memory.as_deref()
    {
        let retrieved = strategy::retrieve_context(store, &thread_id, &user_id, 10).await;
        let context_text = strategy::format_context(&retrieved);
        if !context_text.is_empty() {
            shared.messages.lock().await.insert(0, build_memory_context_message(context_text));
        }
    }

    let mut iteration = 0usize;
    let mut step_idx = 0usize;

    loop {
        if iteration >= config.max_iterations {
            return Err(ArcError::MaxIterations(config.max_iterations));
        }
        iteration += 1;

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ArcError::Timeout(config.timeout));
        }

        let req = {
            let messages = shared.messages.lock().await;
            build_request(
                agent.instructions.as_deref(),
                &messages,
                &config,
                agent.tools.as_deref(),
            )
        };

        let mut accumulated_text = String::new();
        let mut accumulated_calls: Vec<ToolCall> = Vec::new();
        let mut last_finish_reason: Option<FinishReason> = None;
        let mut step_usage = None;

        let mut llm_stream = tokio::time::timeout(remaining, agent.llm.chat_stream(req))
            .await
            .map_err(|_| ArcError::Timeout(config.timeout))?
            .map_err(ArcError::Llm)?;

        while let Some(chunk_result) = llm_stream.next().await {
            let chunk = match chunk_result {
                Err(e) => return Err(ArcError::Llm(e)),
                Ok(c) => c,
            };
            if let Some(t) = &chunk.text_delta {
                accumulated_text.push_str(t);
            }
            accumulated_calls.extend(chunk.tool_calls.clone());
            if chunk.finish_reason.is_some() {
                last_finish_reason = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                step_usage = chunk.usage.clone();
            }

            if !send_event(
                tx,
                RunEvent {
                    run_id: run_id.clone(),
                    user_id: user_id.clone(),
                    thread_id: thread_id.clone(),
                    chunk: Some(chunk),
                    step: None,
                    state_change: None,
                },
            )
            .await
            {
                return Ok(());
            }
        }

        let finish_reason = last_finish_reason.unwrap_or(FinishReason::Completed);
        let has_tool_calls = !accumulated_calls.is_empty() || finish_reason == FinishReason::ToolCalls;

        append_assistant_message(&shared, accumulated_text, &accumulated_calls).await;

        if !has_tool_calls {
            let _ = send_event(
                tx,
                RunEvent {
                    run_id,
                    user_id,
                    thread_id,
                    chunk: None,
                    step: None,
                    state_change: Some(ExecutionState::Completed),
                },
            )
            .await;
            return Ok(());
        }

        let interrupt = agent
            .interrupt_handler
            .as_deref()
            .map(|handler| (handler, &user_id, &thread_id));
        let act_start = Instant::now();
        let tool_results =
            execute_tools(&accumulated_calls, agent.tools.as_deref(), interrupt).await?;

        {
            let mut messages = shared.messages.lock().await;
            append_tool_result_messages(&mut messages, &tool_results);
        }

        let act_step = ExecutionStep {
            index: step_idx,
            kind: StepKind::Act,
            input: StepInput::ToolCalls(accumulated_calls),
            output: StepOutput::ToolResults(tool_results),
            duration: act_start.elapsed(),
            usage: step_usage,
        };
        step_idx += 1;

        if !send_event(
            tx,
            RunEvent {
                run_id: run_id.clone(),
                user_id: user_id.clone(),
                thread_id: thread_id.clone(),
                chunk: None,
                step: Some(act_step),
                state_change: None,
            },
        )
        .await
        {
            return Ok(());
        }
    }
}

fn build_memory_context_message(context_text: String) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::text(context_text)],
        name: Some(MEMORY_CONTEXT_MESSAGE_NAME.to_string()),
    }
}

async fn append_assistant_message(
    shared: &ThreadShared,
    accumulated_text: String,
    accumulated_calls: &[ToolCall],
) {
    let mut content: Vec<ContentPart> = Vec::new();
    if !accumulated_text.is_empty() {
        content.push(ContentPart::text(accumulated_text));
    }
    content.extend(
        accumulated_calls
            .iter()
            .cloned()
            .map(ContentPart::ToolCall),
    );

    if content.is_empty() {
        return;
    }

    shared.messages.lock().await.push(Message {
        role: Role::Assistant,
        content,
        name: None,
    });
}

async fn send_event(
    tx: &tokio::sync::mpsc::Sender<Result<RunEvent, ArcError>>,
    event: RunEvent,
) -> bool {
    tx.send(Ok(event)).await.is_ok()
}
