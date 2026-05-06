use std::time::Duration;

use ufox_llm::{
    ChatRequest, ContentPart, Message, Role, ToolCall, ToolResult, ToolResultPayload, Usage,
};

use crate::agent::{AgentConfig, ReflectConfig};
use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptHandler};
use crate::thread::{ThreadId, UserId};
use crate::tools::ToolManager;

use super::super::trace::{ExecutionStep, StepInput, StepKind, StepOutput};

const MEMORY_CONTEXT_MESSAGE_NAME: &str = "memory_context";

/// 构造主对话请求，并仅在确实存在工具时暴露工具描述。
pub(super) fn build_request(
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

/// 构造反思请求，把反思提示追加到主指令之后，保持同一行为边界。
pub(super) fn reflect_request(
    instructions: Option<&str>,
    messages: &[Message],
    cfg: &ReflectConfig,
) -> ChatRequest {
    let system = match instructions {
        Some(inst) => format!("{inst}\n\n{}", cfg.prompt),
        None => cfg.prompt.clone(),
    };
    ChatRequest::builder()
        .system(system)
        .messages(messages.to_vec())
        .build()
}

/// 执行一批工具调用，并把普通失败折叠成工具错误结果返回给模型。
///
/// 用户主动中止需要立刻终止整个执行流程，因此保留原始错误向上传递。
pub(super) async fn execute_tools(
    calls: &[ToolCall],
    tools: Option<&ToolManager>,
    interrupt_handler: Option<&dyn InterruptHandler>,
    user_id: &UserId,
    thread_id: &ThreadId,
) -> Result<Vec<ToolResult>, ArcError> {
    let Some(manager) = tools else {
        return Ok(calls.iter().map(unregistered_tool_result).collect());
    };
    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        let ctx = interrupt_handler.map(|handler| InterruptCtx {
            handler,
            user_id,
            thread_id,
        });
        match manager.execute(call, ctx).await {
            Ok(r) => results.push(r),
            Err(e) if is_aborted_by_user(&e) => return Err(e),
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

fn is_aborted_by_user(e: &ArcError) -> bool {
    matches!(e, ArcError::Tool { message, .. } if message == "execution aborted by user")
}

fn unregistered_tool_result(call: &ToolCall) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        tool_name: Some(call.tool_name.clone()),
        payload: ToolResultPayload::text(format!(
            "Tool '{}' is not registered.",
            call.tool_name
        )),
        is_error: true,
    }
}

/// 构造一个 `ExecutionStep`，避免 `once` 与 `stream` 分支重复保持字段对齐。
pub(super) fn step(
    index: usize,
    kind: StepKind,
    input: StepInput,
    output: StepOutput,
    duration: Duration,
    usage: Option<Usage>,
) -> ExecutionStep {
    ExecutionStep {
        index,
        kind,
        input,
        output,
        duration,
        usage,
    }
}

/// 追加工具结果消息，保持后续模型回合能看到标准 `tool` 角色输出。
pub(super) fn push_tool_results(messages: &mut Vec<Message>, results: &[ToolResult]) {
    messages.extend(results.iter().cloned().map(|r| Message {
        role: Role::Tool,
        content: vec![ContentPart::ToolResult(r)],
        name: None,
    }));
}

/// 构造记忆上下文消息，并用稳定名称标记为系统注入内容。
pub(super) fn memory_context_message(context_text: String) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::text(context_text)],
        name: Some(MEMORY_CONTEXT_MESSAGE_NAME.to_string()),
    }
}

/// 构造反思消息，把重试原因显式写回消息序列供下一轮参考。
pub(super) fn reflection_message(reason: String) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::text(format!("[Reflection]: {reason}"))],
        name: Some("reflection".to_string()),
    }
}

/// 把工具执行状态压缩成紧凑摘要，供观察阶段 trace 使用。
pub(super) fn observation_summary(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|r| {
            format!(
                "[{}:{}]",
                r.tool_name.as_deref().unwrap_or("tool"),
                if r.is_error { "error" } else { "ok" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 判断反思输出是否要求重试，协议上仅识别 `VERDICT:` 行。
pub(super) fn verdict_is_retry(text: &str) -> bool {
    text.lines()
        .find(|l| l.trim_start().starts_with("VERDICT:"))
        .map(|l| l.contains("RETRY"))
        .unwrap_or(false)
}

/// 提取反思原因，兼容带 `REASON:` 前缀和原始自由文本两种返回。
pub(super) fn reflect_reason(text: &str) -> String {
    text.lines()
        .find(|l| l.trim_start().starts_with("REASON:"))
        .map(|l| {
            l.trim_start_matches(|c: char| c.is_alphabetic() || c == ':')
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| text.to_string())
}

/// 返回最后一条消息的纯文本内容，供记忆检索生成查询词。
pub(super) fn last_message_text(messages: &[Message]) -> String {
    messages.last().map(|m| m.text()).unwrap_or_default()
}

/// 累加 token 统计，统一处理缺失 usage 的响应分支。
pub(super) fn add_usage(total: &mut Usage, usage: Option<&Usage>) {
    let Some(u) = usage else {
        return;
    };
    total.prompt_tokens += u.prompt_tokens;
    total.completion_tokens += u.completion_tokens;
    total.total_tokens += u.total_tokens;
}
