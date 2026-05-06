use std::time::{Duration, Instant};

use ufox_llm::{ChatResponse, FinishReason, Message, Usage};

use crate::agent::Agent;
use crate::error::ArcError;
use crate::memory::strategy;
use crate::thread::{Thread, ThreadId, UserId};

use super::helpers::{
    add_usage, build_request, execute_tools, last_message_text, memory_context_message,
    observation_summary, push_tool_results, reflect_reason, reflect_request,
    reflection_message, step, verdict_is_retry,
};
use super::super::session::{RunInput, RunResult, RunTrace};
use super::super::trace::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};

struct LoopResult {
    steps: Vec<ExecutionStep>,
    final_response: ChatResponse,
    state: ExecutionState,
    total_usage: Usage,
}

/// 执行一次非流式运行，并返回最终响应与完整 trace。
pub async fn run_once(
    agent: &Agent,
    thread: &Thread,
    input: RunInput,
) -> Result<RunResult, ArcError> {
    let wall_start = Instant::now();
    let run_id = super::super::session::RunId::new();
    thread.try_start_run().await?;
    let user_message = input.into_message();

    // 这里单独收窄消息锁作用域，避免 finish_run 这类独立状态更新被消息锁串行化。
    let loop_result = {
        let mut messages = thread.shared.messages.lock().await;
        messages.push(user_message);
        run_loop(agent, &thread.user_id, &thread.thread_id, &mut messages).await
    };

    thread.finish_run().await;

    let LoopResult {
        steps,
        final_response,
        state,
        total_usage,
    } = loop_result?;
    Ok(RunResult {
        run_id: run_id.clone(),
        user_id: thread.user_id.clone(),
        thread_id: thread.thread_id.clone(),
        response: final_response,
        trace: RunTrace {
            run_id,
            user_id: thread.user_id.clone(),
            thread_id: thread.thread_id.clone(),
            steps,
            state,
            total_duration: wall_start.elapsed(),
            total_usage,
        },
    })
}

#[tracing::instrument(
    name = "agent.run_loop",
    skip_all,
    fields(iterations = tracing::field::Empty, total_tokens = tracing::field::Empty)
)]
async fn run_loop(
    agent: &Agent,
    user_id: &UserId,
    thread_id: &ThreadId,
    messages: &mut Vec<Message>,
) -> Result<LoopResult, ArcError> {
    let deadline = Instant::now() + agent.config.timeout;
    let mut steps: Vec<ExecutionStep> = Vec::new();
    let mut idx = 0usize;
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut reflect_count = 0usize;

    // 记忆上下文只注入一次，保证后续 think/reflect 基于同一份前提，避免重试时上下文漂移。
    if let Some(provider) = agent.memory.as_deref() {
        let t = Instant::now();
        let query = last_message_text(messages);
        tracing::debug!(query = %query, "perceive: retrieving memory context");
        let retrieved = strategy::retrieve_context(provider, thread_id, user_id, 10).await;
        let context_text = strategy::format_context(&retrieved);
        if !context_text.is_empty() {
            // 插到最前面是为了把它固定为补充上下文，而不是改变本轮用户消息的相对位置语义。
            tracing::debug!(hits = retrieved.len(), "perceive: injecting memory context");
            messages.insert(0, memory_context_message(context_text));
        }
        steps.push(step(
            idx,
            StepKind::Perceive,
            StepInput::Query(query),
            StepOutput::MemoryHits(retrieved),
            t.elapsed(),
            None,
        ));
        idx += 1;
    }

    let mut iteration = 0usize;
    loop {
        if iteration >= agent.config.max_iterations {
            tracing::warn!(max = agent.config.max_iterations, "agent: max iterations reached");
            return Err(ArcError::MaxIterations(agent.config.max_iterations));
        }
        iteration += 1;

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(timeout = ?agent.config.timeout, "agent: deadline exceeded");
            return Err(ArcError::Timeout(agent.config.timeout));
        }

        tracing::debug!(iteration, "think: calling LLM");
        let t = Instant::now();
        let response = tokio::time::timeout(
            remaining,
            agent.llm.chat(build_request(
                agent.instructions.as_deref(),
                messages,
                &agent.config,
                agent.tools.as_deref(),
            )),
        )
        .await
        .map_err(|_| ArcError::Timeout(agent.config.timeout))?
        .map_err(ArcError::Llm)?;
        let think_dur = t.elapsed();
        add_usage(&mut total_usage, response.usage.as_ref());

        let finish_reason = response.finish_reason.unwrap_or(FinishReason::Completed);
        let has_tool_calls =
            !response.tool_calls.is_empty() || finish_reason == FinishReason::ToolCalls;
        tracing::debug!(
            ?finish_reason,
            has_tool_calls,
            duration_ms = think_dur.as_millis(),
            "think: response received"
        );

        steps.push(step(
            idx,
            StepKind::Think,
            // 记录调用前快照，便于 trace 还原“模型在什么上下文下做出了这次响应”。
            StepInput::Messages(messages.clone()),
            StepOutput::Response {
                message: response.clone().into_message(),
                finish_reason,
                tool_calls: response.tool_calls.clone(),
            },
            think_dur,
            response.usage.clone(),
        ));
        idx += 1;
        // 无论后面是否进入工具阶段，都先把 assistant 响应写回历史，保证消息顺序与真实会话一致。
        messages.push(response.clone().into_message());

        if !has_tool_calls {
            tracing::debug!(
                iteration,
                total_tokens = total_usage.total_tokens,
                "agent: completed"
            );
            tracing::Span::current().record("iterations", iteration);
            tracing::Span::current().record("total_tokens", total_usage.total_tokens);
            steps.push(step(
                idx,
                StepKind::Completion,
                StepInput::Messages(messages.clone()),
                StepOutput::Final {
                    message: response.clone().into_message(),
                    finish_reason,
                },
                // completion 只是为 trace 标记终点，不代表额外消耗了一个执行阶段。
                Duration::ZERO,
                None,
            ));
            return Ok(LoopResult {
                steps,
                final_response: response,
                state: ExecutionState::Completed,
                total_usage,
            });
        }

        tracing::debug!(tool_calls = response.tool_calls.len(), "act: executing tools");
        let t = Instant::now();
        let tool_results = execute_tools(
            &response.tool_calls,
            agent.tools.as_deref(),
            agent.interrupt_handler.as_deref(),
            user_id,
            thread_id,
        )
        .await?;
        tracing::debug!(
            results = tool_results.len(),
            errors = tool_results.iter().filter(|r| r.is_error).count(),
            "act: tools executed"
        );
        steps.push(step(
            idx,
            StepKind::Act,
            StepInput::ToolCalls(response.tool_calls.clone()),
            StepOutput::ToolResults(tool_results.clone()),
            t.elapsed(),
            None,
        ));
        idx += 1;
        // 先把 tool 消息写回历史，再做 observe/reflect，确保后续判断基于 LLM 实际可见的上下文。
        push_tool_results(messages, &tool_results);

        let t = Instant::now();
        steps.push(step(
            idx,
            StepKind::Observe,
            StepInput::ToolResults(tool_results.clone()),
            StepOutput::FormattedObservation(observation_summary(&tool_results)),
            t.elapsed(),
            None,
        ));
        idx += 1;

        if let Some(ref reflect_cfg) = agent.config.reflect
            && reflect_count < reflect_cfg.max_retries
        {
            let t = Instant::now();
            let reflect_response = tokio::time::timeout(
                deadline.saturating_duration_since(Instant::now()),
                agent.llm.chat(reflect_request(
                    agent.instructions.as_deref(),
                    messages,
                    reflect_cfg,
                )),
            )
            .await
            .map_err(|_| ArcError::Timeout(agent.config.timeout))?
            .map_err(ArcError::Llm)?;

            add_usage(&mut total_usage, reflect_response.usage.as_ref());
            let should_retry = verdict_is_retry(&reflect_response.text);
            let reason = reflect_reason(&reflect_response.text);
            tracing::debug!(should_retry, %reason, "reflect: verdict");

            steps.push(step(
                idx,
                StepKind::Reflect,
                // reflect 也保留调用前快照，便于复盘为什么会得到当前重试判定。
                StepInput::Messages(messages.clone()),
                StepOutput::ReflectionDecision {
                    should_retry,
                    reason: reason.clone(),
                },
                t.elapsed(),
                reflect_response.usage,
            ));
            idx += 1;

            if should_retry {
                reflect_count += 1;
                // 追加一条反思提示，而不是覆写历史，让下一轮 think 显式继承这次失败原因。
                messages.push(reflection_message(reason));
            }
        }
    }
}
