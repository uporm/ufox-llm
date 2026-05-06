use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::stream;
use futures::StreamExt;
use ufox_llm::{FinishReason, ToolCall};

use crate::agent::{Agent, AgentConfig};
use crate::error::ArcError;
use crate::memory::strategy;
use crate::thread::{Thread, ThreadId, ThreadShared, UserId};

use super::helpers::{
    build_request, execute_tools, memory_context_message, observation_summary,
    push_tool_results, reflect_reason, reflect_request, reflection_message, step,
    verdict_is_retry,
};
use super::super::session::{RunEvent, RunEventStream, RunId, RunInput};
use super::super::trace::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};

/// 执行一次流式运行，并持续产出增量事件。
pub async fn run_stream(
    agent: &Agent,
    thread: &Thread,
    input: RunInput,
) -> Result<RunEventStream, ArcError> {
    let run_id = RunId::new();
    thread.try_start_run().await?;
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<RunEvent, ArcError>>(32);
    let shared = Arc::clone(&thread.shared);
    let agent = agent.clone();
    let user_id = thread.user_id.clone();
    let thread_id = thread.thread_id.clone();
    let config = agent.config.clone();
    thread
        .shared
        .messages
        .lock()
        .await
        .push(input.into_message());

    tokio::spawn(async move {
        if let Err(err) =
            stream_loop(&agent, &shared, &user_id, &thread_id, &run_id, &config, &tx).await
        {
            let _ = tx.send(Err(err)).await;
        }
        *shared.state.lock().await = crate::thread::ThreadState::Idle;
    });

    Ok(RunEventStream {
        inner: Box::pin(stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        }),
    })
}

/// 将稳定不变的运行标识集中在一起，减少每次发事件时的重复克隆。
struct Emitter<'a> {
    tx: &'a tokio::sync::mpsc::Sender<Result<RunEvent, ArcError>>,
    run_id: &'a RunId,
    user_id: &'a UserId,
    thread_id: &'a ThreadId,
}

impl<'a> Emitter<'a> {
    fn new(
        tx: &'a tokio::sync::mpsc::Sender<Result<RunEvent, ArcError>>,
        run_id: &'a RunId,
        user_id: &'a UserId,
        thread_id: &'a ThreadId,
    ) -> Self {
        Self {
            tx,
            run_id,
            user_id,
            thread_id,
        }
    }

    async fn chunk(&self, chunk: ufox_llm::ChatChunk) -> bool {
        self.send(RunEvent {
            chunk: Some(chunk),
            step: None,
            state_change: None,
            ..self.base()
        })
        .await
    }

    async fn step(&self, step: ExecutionStep) -> bool {
        self.send(RunEvent {
            chunk: None,
            step: Some(step),
            state_change: None,
            ..self.base()
        })
        .await
    }

    async fn complete(self) -> bool {
        self.send(RunEvent {
            chunk: None,
            step: None,
            state_change: Some(ExecutionState::Completed),
            ..self.base()
        })
        .await
    }

    fn base(&self) -> RunEvent {
        RunEvent {
            run_id: self.run_id.clone(),
            user_id: self.user_id.clone(),
            thread_id: self.thread_id.clone(),
            chunk: None,
            step: None,
            state_change: None,
        }
    }

    async fn send(&self, event: RunEvent) -> bool {
        self.tx.send(Ok(event)).await.is_ok()
    }
}

async fn stream_loop(
    agent: &Agent,
    shared: &Arc<ThreadShared>,
    user_id: &UserId,
    thread_id: &ThreadId,
    run_id: &RunId,
    config: &AgentConfig,
    tx: &tokio::sync::mpsc::Sender<Result<RunEvent, ArcError>>,
) -> Result<(), ArcError> {
    let deadline = Instant::now() + config.timeout;
    let emit = Emitter::new(tx, run_id, user_id, thread_id);

    if let Some(provider) = agent.memory.as_deref() {
        let retrieved = strategy::retrieve_context(provider, thread_id, user_id, 10).await;
        let context_text = strategy::format_context(&retrieved);
        if !context_text.is_empty() {
            shared
                .messages
                .lock()
                .await
                .insert(0, memory_context_message(context_text));
        }
    }

    let mut iteration = 0usize;
    let mut step_idx = 0usize;
    let mut reflect_count = 0usize;

    loop {
        if iteration >= config.max_iterations {
            return Err(ArcError::MaxIterations(config.max_iterations));
        }
        iteration += 1;

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ArcError::Timeout(config.timeout));
        }

        let req = build_request(
            agent.instructions.as_deref(),
            &shared.messages.lock().await,
            config,
            agent.tools.as_deref(),
        );
        let mut llm_stream = tokio::time::timeout(remaining, agent.llm.chat_stream(req))
            .await
            .map_err(|_| ArcError::Timeout(config.timeout))?
            .map_err(ArcError::Llm)?;

        let mut accumulated_text = String::new();
        let mut accumulated_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason = FinishReason::Completed;
        let mut step_usage = None;

        while let Some(chunk) = llm_stream.next().await {
            let chunk = chunk.map_err(ArcError::Llm)?;
            if let Some(t) = &chunk.text_delta {
                accumulated_text.push_str(t);
            }
            accumulated_calls.extend(chunk.tool_calls.clone());
            if let Some(r) = chunk.finish_reason {
                finish_reason = r;
            }
            if chunk.usage.is_some() {
                step_usage = chunk.usage.clone();
            }
            if !emit.chunk(chunk).await {
                return Ok(());
            }
        }

        let has_tool_calls =
            !accumulated_calls.is_empty() || finish_reason == FinishReason::ToolCalls;
        push_assistant_message(shared, accumulated_text, &accumulated_calls).await;

        if !has_tool_calls {
            let _ = emit.complete().await;
            return Ok(());
        }

        let t = Instant::now();
        let tool_results = execute_tools(
            &accumulated_calls,
            agent.tools.as_deref(),
            agent.interrupt_handler.as_deref(),
            user_id,
            thread_id,
        )
        .await?;
        push_tool_results(&mut *shared.messages.lock().await, &tool_results);

        if !emit
            .step(step(
                step_idx,
                StepKind::Act,
                StepInput::ToolCalls(accumulated_calls),
                StepOutput::ToolResults(tool_results.clone()),
                t.elapsed(),
                step_usage,
            ))
            .await
        {
            return Ok(());
        }
        step_idx += 1;

        if !emit
            .step(step(
                step_idx,
                StepKind::Observe,
                StepInput::ToolResults(tool_results.clone()),
                StepOutput::FormattedObservation(observation_summary(&tool_results)),
                Duration::ZERO,
                None,
            ))
            .await
        {
            return Ok(());
        }
        step_idx += 1;

        if let Some(ref reflect_cfg) = config.reflect
            && reflect_count < reflect_cfg.max_retries
        {
            let t = Instant::now();
            let reflect_req = reflect_request(
                agent.instructions.as_deref(),
                &shared.messages.lock().await,
                reflect_cfg,
            );
            let remaining = deadline.saturating_duration_since(Instant::now());
            let reflect_response = tokio::time::timeout(remaining, agent.llm.chat(reflect_req))
                .await
                .map_err(|_| ArcError::Timeout(config.timeout))?
                .map_err(ArcError::Llm)?;

            let should_retry = verdict_is_retry(&reflect_response.text);
            let reason = reflect_reason(&reflect_response.text);

            if !emit
                .step(step(
                    step_idx,
                    StepKind::Reflect,
                    StepInput::Messages(shared.messages.lock().await.clone()),
                    StepOutput::ReflectionDecision {
                        should_retry,
                        reason: reason.clone(),
                    },
                    t.elapsed(),
                    reflect_response.usage,
                ))
                .await
            {
                return Ok(());
            }
            step_idx += 1;

            if should_retry {
                reflect_count += 1;
                shared.messages.lock().await.push(reflection_message(reason));
            }
        }
    }
}

async fn push_assistant_message(shared: &ThreadShared, text: String, calls: &[ToolCall]) {
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ufox_llm::ContentPart::text(text));
    }
    content.extend(calls.iter().cloned().map(ufox_llm::ContentPart::ToolCall));
    if content.is_empty() {
        return;
    }
    shared.messages.lock().await.push(ufox_llm::Message {
        role: ufox_llm::Role::Assistant,
        content,
        name: None,
    });
}
