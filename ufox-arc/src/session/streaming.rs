use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use ufox_llm::{ContentPart, FinishReason, Message, Role, ToolCall};

use super::{
    ExecutionEvent, ExecutionState, ExecutionStep, SessionId, SessionShared, StepInput, StepKind,
    StepOutput, UserId,
};
use crate::agent::runner::{
    append_tool_result_messages, build_request, execute_tools,
};
use crate::error::ArcError;
use crate::memory::strategy;

const MEMORY_CONTEXT_MESSAGE_NAME: &str = "memory_context";

/// 完整推理循环的流式版本。
///
/// 每个 Think 步骤以 chunk 事件推送，Act 步骤以 step 事件推送，
/// 结束时发送 `state_change: Completed`。
pub(super) async fn run_streaming_loop(
    shared: Arc<SessionShared>,
    user_id: UserId,
    session_id: SessionId,
    tx: &tokio::sync::mpsc::Sender<Result<ExecutionEvent, ArcError>>,
) -> Result<(), ArcError> {
    let agent = &shared.agent;
    let config = &agent.config;
    let deadline = Instant::now() + config.timeout;

    // Perceive（可选）：检索记忆并注入上下文。
    if config.enable_perceive
        && let Some(store) = agent.memory.as_deref()
    {
        let retrieved = strategy::retrieve_context(store, &session_id, &user_id, 10).await;
        let context_text = strategy::format_context(&retrieved);
        if !context_text.is_empty() {
            shared
                .messages
                .lock()
                .await
                .insert(0, build_memory_context_message(context_text));
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
                agent.system.as_deref(),
                &messages,
                config,
                agent.tools.as_deref(),
            )
        };

        // Think（流式）。
        let mut accumulated_text = String::new();
        let mut accumulated_calls: Vec<ToolCall> = Vec::new();
        let mut last_finish_reason: Option<FinishReason> = None;
        let mut step_usage = None;

        let stream_result = tokio::time::timeout(remaining, agent.llm.chat_stream(req)).await;

        let mut llm_stream = match stream_result {
            Err(_) => return Err(ArcError::Timeout(config.timeout)),
            Ok(Err(e)) => return Err(ArcError::Llm(e)),
            Ok(Ok(s)) => s,
        };

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
                event_chunk(user_id.clone(), session_id.clone(), chunk),
            )
            .await
            {
                // 接收端已丢弃，静默退出。
                return Ok(());
            }
        }

        let finish_reason = last_finish_reason.unwrap_or(FinishReason::Completed);
        let has_tool_calls =
            !accumulated_calls.is_empty() || finish_reason == FinishReason::ToolCalls;

        append_assistant_message(&shared, accumulated_text, &accumulated_calls).await;
        step_idx += 1;

        if !has_tool_calls {
            let _ = send_event(
                tx,
                event_state_change(
                    user_id.clone(),
                    session_id.clone(),
                    ExecutionState::Completed,
                ),
            )
            .await;
            return Ok(());
        }

        // Act：执行工具调用。
        let interrupt = agent
            .interrupt_handler
            .as_deref()
            .map(|h| (h, &user_id, &session_id));
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

        if !send_event(tx, event_step(user_id.clone(), session_id.clone(), act_step)).await {
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
    shared: &SessionShared,
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

fn event_chunk(user_id: UserId, session_id: SessionId, chunk: ufox_llm::ChatChunk) -> ExecutionEvent {
    ExecutionEvent {
        user_id,
        session_id,
        chunk: Some(chunk),
        step: None,
        state_change: None,
    }
}

fn event_step(user_id: UserId, session_id: SessionId, step: ExecutionStep) -> ExecutionEvent {
    ExecutionEvent {
        user_id,
        session_id,
        chunk: None,
        step: Some(step),
        state_change: None,
    }
}

fn event_state_change(
    user_id: UserId,
    session_id: SessionId,
    state: ExecutionState,
) -> ExecutionEvent {
    ExecutionEvent {
        user_id,
        session_id,
        chunk: None,
        step: None,
        state_change: Some(state),
    }
}

async fn send_event(
    tx: &tokio::sync::mpsc::Sender<Result<ExecutionEvent, ArcError>>,
    event: ExecutionEvent,
) -> bool {
    tx.send(Ok(event)).await.is_ok()
}
