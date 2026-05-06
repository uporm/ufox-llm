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
    /// 当前步骤消费的文本查询。
    Query(String),
    /// 当前步骤读取到的完整消息历史快照。
    Messages(Vec<Message>),
    /// 当前步骤准备执行的工具调用列表。
    ToolCalls(Vec<ToolCall>),
    /// 当前步骤接收到的工具结果列表。
    ToolResults(Vec<ToolResult>),
}

/// 执行步骤的输出。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum StepOutput {
    /// Perceive 阶段检索到的记忆命中结果。
    MemoryHits(Vec<Memory>),
    /// Think 阶段产出的模型响应和工具调用。
    Response {
        /// 规范化后的 assistant 消息。
        message: Message,
        /// 模型结束当前轮生成的原因。
        finish_reason: FinishReason,
        /// 模型在本轮请求中给出的工具调用。
        tool_calls: Vec<ToolCall>,
    },
    /// Act 阶段执行后的工具结果。
    ToolResults(Vec<ToolResult>),
    /// Observe 阶段整理后的文本观察结果。
    FormattedObservation(String),
    /// Reflect 阶段给出的继续或重试判断。
    ReflectionDecision {
        /// 是否建议基于当前状态重试。
        should_retry: bool,
        /// 做出该判断的原因。
        reason: String,
    },
    /// Completion 阶段产出的最终消息。
    Final {
        /// 最终返回给上层的 assistant 消息。
        message: Message,
        /// 最终一轮生成的结束原因。
        finish_reason: FinishReason,
    },
}

/// 单个执行步骤的完整记录。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionStep {
    /// 步骤在本次执行轨迹中的顺序编号。
    pub index: usize,
    /// 步骤所属的阶段类型。
    pub kind: StepKind,
    /// 步骤执行时看到的输入。
    pub input: StepInput,
    /// 步骤执行后产出的结果。
    pub output: StepOutput,
    /// 步骤自身耗时，不含轨迹外的额外处理。
    pub duration: Duration,
    /// 与该步骤相关的 token 使用量。
    pub usage: Option<Usage>,
}

/// 执行状态。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ExecutionState {
    /// 执行仍在进行中。
    Running,
    /// 执行已正常完成。
    Completed,
    /// 执行失败，并附带错误描述。
    Failed { error: String },
    /// 执行因超时终止。
    TimedOut,
    /// 执行因达到最大轮数限制而停止。
    MaxIterationsReached,
}
