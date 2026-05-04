use std::time::{Duration, Instant};

use ufox_llm::{
    ChatRequest, ChatResponse, Client, ContentPart, FinishReason, Message, Role, ToolResult,
    ToolResultPayload, Usage,
};

use super::config::AgentConfig;
use super::execution::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};
use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptHandler};
use crate::memory::{MemoryStore, strategy};
use crate::thread::{ThreadId, UserId};
use crate::tools::ToolManager;

const MEMORY_CONTEXT_MESSAGE_NAME: &str = "memory_context";

/// 记忆检索所需的上下文。
pub(crate) struct MemoryCtx<'a> {
    pub store: &'a dyn MemoryStore,
    pub user_id: &'a UserId,
    pub thread_id: &'a ThreadId,
}

/// `run_loop` 的调用上下文。
///
/// 将主要依赖聚合在一起，避免函数签名随功能扩展而失控。
pub(crate) struct LoopCtx<'a> {
    pub llm: &'a Client,
    pub instructions: Option<&'a str>,
    pub config: &'a AgentConfig,
    pub tools: Option<&'a ToolManager>,
    pub memory: Option<MemoryCtx<'a>>,
    pub interrupt: Option<(&'a dyn InterruptHandler, &'a UserId, &'a ThreadId)>,
}

/// `run_loop` 的内部返回值。
pub(crate) struct LoopResult {
    pub steps: Vec<ExecutionStep>,
    pub final_response: ChatResponse,
    pub state: ExecutionState,
    pub total_usage: Usage,
}

/// 执行 Perceive → Think → Act → (Observe) → (Reflect) → Completion 循环。
#[tracing::instrument(
    name = "agent.run_loop",
    skip_all,
    fields(
        iterations = tracing::field::Empty,
        total_tokens = tracing::field::Empty,
    )
)]
pub(crate) async fn run_loop(
    ctx: LoopCtx<'_>,
    messages: &mut Vec<Message>,
) -> Result<LoopResult, ArcError> {
    let LoopCtx {
        llm,
        instructions,
        config,
        tools,
        memory,
        interrupt,
    } = ctx;
    let deadline = Instant::now() + config.timeout;
    let mut steps: Vec<ExecutionStep> = Vec::new();
    let mut idx: usize = 0;
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    // Perceive（可选）：检索记忆并注入上下文。
    if config.enable_perceive {
        let t = Instant::now();
        let query = latest_message_text(messages);
        tracing::debug!(query = %query, "perceive: retrieving memory context");

        let hits = if let Some(ref ctx) = memory {
            let retrieved =
                strategy::retrieve_context(ctx.store, ctx.thread_id, ctx.user_id, 10).await;
            let context_text = strategy::format_context(&retrieved);
            if !context_text.is_empty() {
                tracing::debug!(hits = retrieved.len(), "perceive: injecting memory context");
                messages.insert(0, build_memory_context_message(context_text));
            }
            retrieved
        } else {
            vec![]
        };

        steps.push(ExecutionStep {
            index: idx,
            kind: StepKind::Perceive,
            input: StepInput::Query(query),
            output: StepOutput::MemoryHits(hits),
            duration: t.elapsed(),
            usage: None,
        });
        idx += 1;
    }

    // 主循环。
    let mut iteration = 0usize;
    loop {
        if iteration >= config.max_iterations {
            tracing::warn!(max = config.max_iterations, "agent: max iterations reached");
            return Err(ArcError::MaxIterations(config.max_iterations));
        }
        iteration += 1;

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(timeout = ?config.timeout, "agent: deadline exceeded");
            return Err(ArcError::Timeout(config.timeout));
        }

        // Think：调用 LLM。
        tracing::debug!(iteration, "think: calling LLM");
        let think_start = Instant::now();
        let req = build_request(instructions, messages, config, tools);

        let response = tokio::time::timeout(remaining, llm.chat(req))
            .await
            .map_err(|_| ArcError::Timeout(config.timeout))?
            .map_err(ArcError::Llm)?;

        let think_dur = think_start.elapsed();

        merge_usage(&mut total_usage, response.usage.as_ref());

        let finish_reason = response.finish_reason.unwrap_or(FinishReason::Completed);
        let has_tool_calls =
            !response.tool_calls.is_empty() || finish_reason == FinishReason::ToolCalls;
        tracing::debug!(
            ?finish_reason,
            has_tool_calls,
            duration_ms = think_dur.as_millis(),
            "think: response received"
        );

        steps.push(ExecutionStep {
            index: idx,
            kind: StepKind::Think,
            input: StepInput::Messages(messages.clone()),
            output: StepOutput::Response {
                message: response.clone().into_message(),
                finish_reason,
                tool_calls: response.tool_calls.clone(),
            },
            duration: think_dur,
            usage: response.usage.clone(),
        });
        idx += 1;

        messages.push(response.clone().into_message());

        if !has_tool_calls {
            tracing::debug!(
                iteration,
                total_tokens = total_usage.total_tokens,
                "agent: completed"
            );
            tracing::Span::current().record("iterations", iteration);
            tracing::Span::current().record("total_tokens", total_usage.total_tokens);
            steps.push(ExecutionStep {
                index: idx,
                kind: StepKind::Completion,
                input: StepInput::Messages(messages.clone()),
                output: StepOutput::Final {
                    message: response.clone().into_message(),
                    finish_reason,
                },
                duration: Duration::ZERO,
                usage: None,
            });

            return Ok(LoopResult {
                steps,
                final_response: response,
                state: ExecutionState::Completed,
                total_usage,
            });
        }

        // Act：执行工具调用；用户中止时立即终止整个循环。
        tracing::debug!(
            tool_calls = response.tool_calls.len(),
            "act: executing tools"
        );
        let act_start = Instant::now();
        let tool_results = execute_tools(&response.tool_calls, tools, interrupt).await?;
        tracing::debug!(
            results = tool_results.len(),
            errors = tool_results.iter().filter(|r| r.is_error).count(),
            "act: tools executed"
        );

        steps.push(ExecutionStep {
            index: idx,
            kind: StepKind::Act,
            input: StepInput::ToolCalls(response.tool_calls.clone()),
            output: StepOutput::ToolResults(tool_results.clone()),
            duration: act_start.elapsed(),
            usage: None,
        });
        idx += 1;

        append_tool_result_messages(messages, &tool_results);

        // Observe（可选）。
        if config.enable_observe {
            let t = Instant::now();
            let observation = format!("Received {} tool result(s).", tool_results.len());
            steps.push(ExecutionStep {
                index: idx,
                kind: StepKind::Observe,
                input: StepInput::ToolResults(tool_results.clone()),
                output: StepOutput::FormattedObservation(observation),
                duration: t.elapsed(),
                usage: None,
            });
            idx += 1;
        }

        // Reflect（可选）。
        if config.enable_reflect {
            let t = Instant::now();
            steps.push(ExecutionStep {
                index: idx,
                kind: StepKind::Reflect,
                input: StepInput::Messages(messages.clone()),
                output: StepOutput::ReflectionDecision {
                    should_retry: false,
                    reason: "continuing to next iteration".to_string(),
                },
                duration: t.elapsed(),
                usage: None,
            });
            idx += 1;
        }
    }
}

/// 构建一次 LLM `chat` 请求。
pub(crate) fn build_request(
    instructions: Option<&str>,
    messages: &[Message],
    config: &AgentConfig,
    tools: Option<&ToolManager>,
) -> ChatRequest {
    let mut builder = ChatRequest::builder().messages(messages.to_vec());
    if let Some(s) = instructions {
        builder = builder.system(s.to_string());
    }
    if let Some(t) = config.temperature {
        builder = builder.temperature(t);
    }
    if let Some(manager) = tools {
        let llm_tools = manager.to_llm_tools();
        if !llm_tools.is_empty() {
            builder = builder.tools(llm_tools);
        }
    }
    builder.build()
}

/// 执行所有工具调用。
///
/// 普通错误会转换为 `is_error=true` 的工具结果继续返回；
/// 若用户明确中止，则直接返回错误中断主循环。
pub(crate) async fn execute_tools(
    calls: &[ufox_llm::ToolCall],
    tools: Option<&ToolManager>,
    interrupt: Option<(&dyn InterruptHandler, &UserId, &ThreadId)>,
) -> Result<Vec<ToolResult>, ArcError> {
    let Some(manager) = tools else {
        return Ok(stub_execute_tools(calls));
    };

    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        let ctx = interrupt.map(|(handler, user_id, thread_id)| InterruptCtx {
            handler,
            user_id,
            thread_id,
        });
        let result = manager.execute(call, ctx).await;
        match result {
            Ok(r) => results.push(r),
            Err(ref e) if is_execution_aborted_by_user(e) => {
                return Err(result.unwrap_err());
            }
            Err(e) => results.push(tool_error_result(call, e)),
        }
    }
    Ok(results)
}

/// 将工具结果追加为 `tool` 角色消息。
pub(crate) fn append_tool_result_messages(messages: &mut Vec<Message>, results: &[ToolResult]) {
    messages.extend(results.iter().cloned().map(|result| Message {
        role: Role::Tool,
        content: vec![ContentPart::ToolResult(result)],
        name: None,
    }));
}

fn latest_message_text(messages: &[Message]) -> String {
    messages.last().map(|message| message.text()).unwrap_or_default()
}

fn build_memory_context_message(context_text: String) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::text(context_text)],
        name: Some(MEMORY_CONTEXT_MESSAGE_NAME.to_string()),
    }
}

fn merge_usage(total: &mut Usage, usage: Option<&Usage>) {
    let Some(usage) = usage else {
        return;
    };

    total.prompt_tokens += usage.prompt_tokens;
    total.completion_tokens += usage.completion_tokens;
    total.total_tokens += usage.total_tokens;
}

fn is_execution_aborted_by_user(error: &ArcError) -> bool {
    matches!(
        error,
        ArcError::Tool { message, .. } if message == "execution aborted by user"
    )
}

fn tool_error_result(call: &ufox_llm::ToolCall, error: ArcError) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        tool_name: Some(call.tool_name.clone()),
        payload: ToolResultPayload::text(error.to_string()),
        is_error: true,
    }
}

fn stub_execute_tools(calls: &[ufox_llm::ToolCall]) -> Vec<ToolResult> {
    calls
        .iter()
        .map(|call| ToolResult {
            tool_call_id: call.id.clone(),
            tool_name: Some(call.tool_name.clone()),
            payload: ToolResultPayload::text(format!(
                "Tool '{}' is not registered.",
                call.tool_name
            )),
            is_error: true,
        })
        .collect()
}
