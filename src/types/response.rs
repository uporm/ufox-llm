//! 响应模型。
//!
//! 定义完整响应、流式增量、结束原因和用量统计等公共类型。

use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

/// 生成结束原因。
///
/// 该枚举统一表达不同 `Provider` 返回的停止原因。对于暂未内建支持的结束原因，
/// 使用 [`FinishReason::Other`] 保留原始字符串，避免信息丢失。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// 模型正常停止。
    Stop,
    /// 达到最大生成长度后停止。
    Length,
    /// 模型请求调用工具后停止。
    ToolCalls,
    /// 因内容过滤被中止。
    ContentFilter,
    /// 因错误被中止。
    Error,
    /// 其他未内建的结束原因。
    Other(String),
}

impl FinishReason {
    /// 从 `Provider` 返回值创建结束原因。
    pub fn from_provider_value(value: impl AsRef<str>) -> Self {
        match value.as_ref() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            "error" => Self::Error,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Error => "error",
            Self::Other(value) => value.as_str(),
        }
    }

    pub const fn is_tool_calls(&self) -> bool {
        matches!(self, Self::ToolCalls)
    }
}

impl From<&str> for FinishReason {
    fn from(value: &str) -> Self {
        Self::from_provider_value(value)
    }
}

impl From<String> for FinishReason {
    fn from(value: String) -> Self {
        Self::from_provider_value(value)
    }
}

/// 推理强度。
///
/// 该枚举统一表达 `OpenAI` `reasoning_effort` 参数的三个常见档位。对于不支持该参数的
/// Provider，设置后会被静默忽略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// 快速推理，适合简单任务。
    Low,
    /// 均衡模式。
    Medium,
    /// 深度推理，适合复杂逻辑任务。
    High,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// 流式片段的内容类型。
///
/// 某些模型会先输出思考过程，再输出正式回复。该枚举用于在流式消费时区分二者。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaType {
    /// 思考过程文本增量。
    Thinking(String),
    /// 正式回复文本增量。
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaKind {
    Thinking,
    Content,
}

/// 用量统计信息。
///
/// 该结构体统一表达一次请求的输入、输出与总 `token` 用量，便于日志记录、计费统计或速率控制。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// 输入（提示词）消耗的 token 数。
    pub prompt: u32,
    /// 输出（补全）消耗的 token 数。
    pub completion: u32,
    /// 总 token 数（`prompt + completion`）。
    pub total: u32,
}

impl Usage {
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt: prompt_tokens,
            completion: completion_tokens,
            total: prompt_tokens + completion_tokens,
        }
    }
}

/// 一次完整的聊天响应。
///
/// 该结构体用于表达非流式请求的最终返回值，也可作为流式聚合完成后的统一结果对象。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    pub thinking_content: Option<String>,
    pub thinking_tokens: Option<u32>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

impl ChatResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            thinking_content: None,
            thinking_tokens: None,
            tool_calls: None,
            finish_reason: None,
            usage: None,
        }
    }

    /// 为响应附加思考过程文本。
    pub fn with_thinking_content(mut self, thinking_content: impl Into<String>) -> Self {
        self.thinking_content = Some(thinking_content.into());
        self
    }

    /// 为响应附加思考阶段消耗的 `token` 数。
    pub const fn with_thinking_tokens(mut self, thinking_tokens: u32) -> Self {
        self.thinking_tokens = Some(thinking_tokens);
        self
    }

    /// 为响应附加工具调用列表。
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// 为响应附加结束原因。
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// 为响应附加用量信息。
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }
}

/// 单个流式响应片段。
///
/// 该结构体用于表达流式输出中的单个增量事件。大多数片段只包含 `delta`，而流尾片段
/// 可能同时携带 `finish_reason`、`usage` 或最终聚合完成的工具调用列表。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamChunk {
    pub delta: String,
    pub delta_kind: DeltaKind,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

impl StreamChunk {
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            delta_kind: DeltaKind::Content,
            tool_calls: None,
            finish_reason: None,
            usage: None,
        }
    }

    pub fn thinking(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            delta_kind: DeltaKind::Thinking,
            tool_calls: None,
            finish_reason: None,
            usage: None,
        }
    }

    /// 为流式片段附加工具调用列表。
    ///
    /// 这里保存的是当前片段可对外消费的完整工具调用集合，而不是尚未闭合的部分参数碎片。
    /// 这样可以让公开 `API` 保持稳定简单，具体的片段级聚合逻辑放在 `Provider` 层内部完成。
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// 为流式片段附加结束原因。
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// 为流式片段附加用量信息。
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn delta_type(&self) -> DeltaType {
        match self.delta_kind {
            DeltaKind::Thinking => DeltaType::Thinking(self.delta.clone()),
            DeltaKind::Content => DeltaType::Content(self.delta.clone()),
        }
    }

    pub const fn is_thinking(&self) -> bool {
        matches!(self.delta_kind, DeltaKind::Thinking)
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }

    pub fn is_terminal(&self) -> bool {
        self.finish_reason.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatResponse, DeltaType, FinishReason, ReasoningEffort, StreamChunk, Usage};
    use crate::ToolCall;

    #[test]
    fn token() {
        let usage = Usage::new(15, 27);

        assert_eq!(usage.prompt, 15);
        assert_eq!(usage.completion, 27);
        assert_eq!(usage.total, 42);
    }

    #[test]
    fn response_test() {
        let response = ChatResponse::new("")
            .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")])
            .with_finish_reason(FinishReason::ToolCalls);

        assert!(response.has_tool_calls());
        assert_eq!(
            response.tool_calls.as_ref().expect("应包含工具调用")[0].name,
            "get_weather"
        );
        assert_eq!(response.finish_reason.as_ref(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn response_test_2() {
        let chunk = StreamChunk::new("")
            .with_finish_reason(FinishReason::Stop)
            .with_usage(Usage::new(10, 8));

        assert!(chunk.is_terminal());
        assert_eq!(chunk.usage.as_ref().expect("应包含用量").total, 18);
    }

    #[test]
    fn response_test_3() {
        let reason = FinishReason::from_provider_value("sensitive");

        assert_eq!(reason.as_str(), "sensitive");
        assert_eq!(reason, FinishReason::Other("sensitive".to_string()));
    }

    #[test]
    fn response_test_4() {
        let response = ChatResponse::new("最终答案")
            .with_thinking_content("先分析条件")
            .with_thinking_tokens(42);

        assert_eq!(response.thinking_content.as_deref(), Some("先分析条件"));
        assert_eq!(response.thinking_tokens, Some(42));
        assert_eq!(response.content, "最终答案");
    }

    #[test]
    fn response_test_5() {
        let thinking = StreamChunk::thinking("分析中");
        let content = StreamChunk::new("答案");

        assert!(thinking.is_thinking());
        assert_eq!(
            thinking.delta_type(),
            DeltaType::Thinking("分析中".to_string())
        );
        assert_eq!(content.delta_type(), DeltaType::Content("答案".to_string()));
    }

    #[test]
    fn response_test_6() {
        assert_eq!(ReasoningEffort::Low.as_str(), "low");
        assert_eq!(ReasoningEffort::Medium.as_str(), "medium");
        assert_eq!(ReasoningEffort::High.as_str(), "high");
    }
}
