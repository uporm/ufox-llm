use std::time::Duration;

use ufox_llm::{FinishReason, Message, ToolCall, ToolResult, Usage};

pub use crate::memory::Memory;

/// 执行步骤的种类。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// 感知：从记忆/环境检索上下文（可选）
    Perceive,
    /// 思考：LLM 推理，生成回复或工具调用
    Think,
    /// 行动：执行工具调用
    Act,
    /// 观察：格式化工具结果（可选）
    Observe,
    /// 反思：自我评估，判断是否重试（可选）
    Reflect,
    /// 完成：最终响应
    Completion,
}

/// 执行步骤的输入。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum StepInput {
    Query(String),
    Messages(Vec<Message>),
    ToolCalls(Vec<ToolCall>),
    ToolResults(Vec<ToolResult>),
}

/// 执行步骤的输出。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum StepOutput {
    MemoryHits(Vec<Memory>),
    Response {
        message: Message,
        finish_reason: FinishReason,
        tool_calls: Vec<ToolCall>,
    },
    ToolResults(Vec<ToolResult>),
    FormattedObservation(String),
    ReflectionDecision {
        should_retry: bool,
        reason: String,
    },
    Final {
        message: Message,
        finish_reason: FinishReason,
    },
}

/// 单个执行步骤的完整记录。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionStep {
    pub index: usize,
    pub kind: StepKind,
    pub input: StepInput,
    pub output: StepOutput,
    pub duration: Duration,
    pub usage: Option<Usage>,
}

/// 执行状态。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ExecutionState {
    Running,
    Completed,
    Failed { error: String },
    TimedOut,
    MaxIterationsReached,
}
