use std::time::{Duration, Instant};

use ufox_llm::{
    ChatRequest, ChatResponse, Client, ContentPart, FinishReason, Message, Role, ToolResult,
    ToolResultPayload, Usage,
};

use crate::agent::config::AgentConfig;
use crate::agent::step::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};
use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptHandler};
use crate::memory::{MemoryStore, strategy};
use crate::session::{SessionId, UserId};
use crate::tools::ToolRegistry;

/// 记忆上下文：检索时所需的 store 引用与作用域标识。
pub(crate) struct MemoryCtx<'a> {
    pub store: &'a dyn MemoryStore,
    pub user_id: &'a UserId,
    pub session_id: &'a SessionId,
}

/// `run_loop` 的调用上下文，归组所有不可变依赖，避免过长的参数列表。
pub(crate) struct LoopCtx<'a> {
    pub llm: &'a Client,
    pub system: Option<&'a str>,
    pub config: &'a AgentConfig,
    pub tools: Option<&'a ToolRegistry>,
    pub memory: Option<MemoryCtx<'a>>,
    pub interrupt: Option<(&'a dyn InterruptHandler, &'a UserId, &'a SessionId)>,
}

/// `run_loop` 的内部返回值；不暴露给公开 API。
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
        system,
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

    // Perceive（可选）：检索记忆并注入上下文
    if config.enable_perceive {
        let t = Instant::now();
        let query = messages.last().map(|m| m.text()).unwrap_or_default();
        tracing::debug!(query = %query, "perceive: retrieving memory context");

        let hits = if let Some(ref ctx) = memory {
            let retrieved =
                strategy::retrieve_context(ctx.store, ctx.session_id, ctx.user_id, 10).await;
            let context_text = strategy::format_context(&retrieved);
            if !context_text.is_empty() {
                tracing::debug!(hits = retrieved.len(), "perceive: injecting memory context");
                messages.insert(
                    0,
                    Message {
                        role: Role::User,
                        content: vec![ContentPart::text(context_text)],
                        name: Some("memory_context".to_string()),
                    },
                );
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

    // 主循环
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

        // Think：调用 LLM
        tracing::debug!(iteration, "think: calling LLM");
        let think_start = Instant::now();
        let req = build_request(system, messages, config, tools);

        let response = tokio::time::timeout(remaining, llm.chat(req))
            .await
            .map_err(|_| ArcError::Timeout(config.timeout))?
            .map_err(ArcError::Llm)?;

        let think_dur = think_start.elapsed();

        if let Some(u) = &response.usage {
            total_usage.prompt_tokens += u.prompt_tokens;
            total_usage.completion_tokens += u.completion_tokens;
            total_usage.total_tokens += u.total_tokens;
        }

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

        // Act：执行工具调用（Abort 时提前返回错误）
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

        for result in &tool_results {
            messages.push(Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult(result.clone())],
                name: None,
            });
        }

        // Observe（可选）
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

        // Reflect（可選）
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

pub(crate) fn build_request(
    system: Option<&str>,
    messages: &[Message],
    config: &AgentConfig,
    tools: Option<&ToolRegistry>,
) -> ChatRequest {
    let mut builder = ChatRequest::builder().messages(messages.to_vec());
    if let Some(s) = system {
        builder = builder.system(s.to_string());
    }
    if let Some(t) = config.temperature {
        builder = builder.temperature(t);
    }
    if let Some(registry) = tools {
        let llm_tools = registry.to_llm_tools();
        if !llm_tools.is_empty() {
            builder = builder.tools(llm_tools);
        }
    }
    builder.build()
}

/// 执行所有工具调用。
///
/// - 普通错误：封装为 `is_error=true` 结果，循环继续。
/// - `Abort` 决策：返回 `Err`，立即中止整个循环。
pub(crate) async fn execute_tools(
    calls: &[ufox_llm::ToolCall],
    tools: Option<&ToolRegistry>,
    interrupt: Option<(&dyn InterruptHandler, &UserId, &SessionId)>,
) -> Result<Vec<ToolResult>, ArcError> {
    let Some(registry) = tools else {
        return Ok(stub_execute_tools(calls));
    };

    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        let ctx = interrupt.map(|(handler, user_id, session_id)| InterruptCtx {
            handler,
            user_id,
            session_id,
        });
        let result = registry.execute(call, ctx).await;
        match result {
            Ok(r) => results.push(r),
            // Abort はエラーとして伝播させる
            Err(ArcError::Tool { ref message, .. }) if message == "execution aborted by user" => {
                return Err(result.unwrap_err());
            }
            Err(e) => results.push(ToolResult {
                tool_call_id: call.id.clone(),
                tool_name: Some(call.tool_name.clone()),
                payload: ToolResultPayload::text(e.to_string()),
                is_error: true,
            }),
        }
    }
    Ok(results)
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
