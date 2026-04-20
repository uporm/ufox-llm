//! 响应模型模块。
//!
//! 该模块负责定义 `LLM` 对话请求返回后的统一中间层类型，包括非流式响应、
//! 流式增量片段、用量统计以及生成结束原因。
//!
//! 设计上采用“完整响应 + 增量响应并行建模”的方式：
//! 1. [`ChatResponse`] 用于表达一次完整请求结束后的最终结果，适合非流式调用；
//! 2. [`StreamChunk`] 用于表达流式输出中的单个增量片段，并允许在结束片段中附带
//!    `finish_reason` 与 `usage`；
//! 3. 工具调用在统一响应模型中直接复用 `tool` 模块里的 [`ToolCall`]，从而避免
//!    不同模块之间重复定义相同语义的数据结构。
//!
//! 该模块依赖 `serde` 进行序列化，并依赖同级 `tool` 模块中的工具调用类型。

use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

/// 生成结束原因。
///
/// 该枚举统一表达不同 `Provider` 返回的停止原因。对于暂未内建支持的结束原因，
/// 使用 [`FinishReason::Other`] 保留原始字符串，避免信息丢失。
///
/// # 示例
/// ```rust
/// use ufox_llm::FinishReason;
///
/// let reason = FinishReason::from_provider_value("tool_calls");
/// assert!(reason.is_tool_calls());
/// ```
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
    ///
    /// # Arguments
    /// * `value` - `Provider` 返回的原始结束原因字符串
    ///
    /// # Returns
    /// 统一后的结束原因枚举。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::FinishReason;
    ///
    /// let reason = FinishReason::from_provider_value("stop");
    /// assert_eq!(reason, FinishReason::Stop);
    /// ```
    #[must_use]
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

    /// 返回结束原因的稳定字符串表示。
    ///
    /// # Returns
    /// 与主流 `Chat API` 兼容的小写结束原因字符串。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::FinishReason;
    ///
    /// assert_eq!(FinishReason::Length.as_str(), "length");
    /// ```
    #[must_use]
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

    /// 返回是否因为工具调用而停止。
    ///
    /// # Returns
    /// 如果结束原因是 [`FinishReason::ToolCalls`]，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::FinishReason;
    ///
    /// assert!(FinishReason::ToolCalls.is_tool_calls());
    /// ```
    #[must_use]
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
///
/// # 示例
/// ```rust
/// use ufox_llm::ReasoningEffort;
///
/// assert_eq!(ReasoningEffort::High.as_str(), "high");
/// ```
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
    /// 返回稳定的字符串表示。
    ///
    /// # Returns
    /// 与 `OpenAI` 请求体兼容的小写字符串。
    #[must_use]
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
///
/// # 示例
/// ```rust
/// use ufox_llm::DeltaType;
///
/// let delta = DeltaType::Thinking("分析中".to_string());
/// assert!(matches!(delta, DeltaType::Thinking(_)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaType {
    /// 思考过程文本增量。
    Thinking(String),
    /// 正式回复文本增量。
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum DeltaKind {
    Thinking,
    Content,
}

/// 用量统计信息。
///
/// 该结构体统一表达一次请求的输入、输出与总 `token` 用量，便于日志记录、
/// 计费统计或速率控制。
///
/// # 示例
/// ```rust
/// use ufox_llm::Usage;
///
/// let usage = Usage::new(12, 34);
/// assert_eq!(usage.total_tokens(), 46);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    prompt: u32,
    completion: u32,
    total: u32,
}

impl Usage {
    /// 创建用量统计信息。
    ///
    /// 总 `token` 数会由输入和输出 `token` 自动求和，避免调用方重复传入并产生不一致。
    ///
    /// # Arguments
    /// * `prompt_tokens` - 输入 `token` 数
    /// * `completion_tokens` - 输出 `token` 数
    ///
    /// # Returns
    /// 新的用量统计对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Usage;
    ///
    /// let usage = Usage::new(10, 20);
    /// assert_eq!(usage.total_tokens(), 30);
    /// ```
    #[must_use]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt: prompt_tokens,
            completion: completion_tokens,
            total: prompt_tokens + completion_tokens,
        }
    }

    /// 返回输入 `token` 数。
    ///
    /// # Returns
    /// 请求输入部分消耗的 `token` 数。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Usage;
    ///
    /// let usage = Usage::new(10, 20);
    /// assert_eq!(usage.prompt_tokens(), 10);
    /// ```
    #[must_use]
    pub const fn prompt_tokens(&self) -> u32 {
        self.prompt
    }

    /// 返回输出 `token` 数。
    ///
    /// # Returns
    /// 模型生成部分消耗的 `token` 数。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Usage;
    ///
    /// let usage = Usage::new(10, 20);
    /// assert_eq!(usage.completion_tokens(), 20);
    /// ```
    #[must_use]
    pub const fn completion_tokens(&self) -> u32 {
        self.completion
    }

    /// 返回总 `token` 数。
    ///
    /// # Returns
    /// 输入和输出 `token` 的总和。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Usage;
    ///
    /// let usage = Usage::new(10, 20);
    /// assert_eq!(usage.total_tokens(), 30);
    /// ```
    #[must_use]
    pub const fn total_tokens(&self) -> u32 {
        self.total
    }
}

/// 一次完整的聊天响应。
///
/// 该结构体用于表达非流式请求的最终返回值，也可作为流式聚合完成后的统一结果对象。
///
/// # 示例
/// ```rust
/// use ufox_llm::{ChatResponse, FinishReason, Usage};
///
/// let response = ChatResponse::new("你好")
///     .with_finish_reason(FinishReason::Stop)
///     .with_usage(Usage::new(10, 8));
///
/// assert_eq!(response.content(), "你好");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponse {
    content: String,
    thinking_content: Option<String>,
    thinking_tokens: Option<u32>,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl ChatResponse {
    /// 创建完整聊天响应。
    ///
    /// # Arguments
    /// * `content` - 模型最终输出的文本内容
    ///
    /// # Returns
    /// 不包含工具调用和用量信息的基础响应对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ChatResponse;
    ///
    /// let response = ChatResponse::new("你好");
    /// assert_eq!(response.content(), "你好");
    /// ```
    #[must_use]
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
    ///
    /// # Arguments
    /// * `thinking_content` - 模型返回的思考过程文本
    ///
    /// # Returns
    /// 附加思考内容后的响应对象。
    #[must_use]
    pub fn with_thinking_content(mut self, thinking_content: impl Into<String>) -> Self {
        self.thinking_content = Some(thinking_content.into());
        self
    }

    /// 为响应附加思考阶段消耗的 `token` 数。
    ///
    /// # Arguments
    /// * `thinking_tokens` - 思考阶段的 `token` 用量
    ///
    /// # Returns
    /// 附加思考用量后的响应对象。
    #[must_use]
    pub const fn with_thinking_tokens(mut self, thinking_tokens: u32) -> Self {
        self.thinking_tokens = Some(thinking_tokens);
        self
    }

    /// 为响应附加工具调用列表。
    ///
    /// # Arguments
    /// * `tool_calls` - 模型返回的工具调用集合
    ///
    /// # Returns
    /// 附加工具调用后的响应对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, ToolCall};
    ///
    /// let response = ChatResponse::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert_eq!(response.tool_calls().expect("应包含工具调用").len(), 1);
    /// ```
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// 为响应附加结束原因。
    ///
    /// # Arguments
    /// * `finish_reason` - 生成结束原因
    ///
    /// # Returns
    /// 附加结束原因后的响应对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, FinishReason};
    ///
    /// let response = ChatResponse::new("你好").with_finish_reason(FinishReason::Stop);
    /// assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    /// ```
    #[must_use]
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// 为响应附加用量信息。
    ///
    /// # Arguments
    /// * `usage` - 用量统计信息
    ///
    /// # Returns
    /// 附加用量信息后的响应对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, Usage};
    ///
    /// let response = ChatResponse::new("你好").with_usage(Usage::new(10, 8));
    /// assert_eq!(response.usage().expect("应包含用量").total_tokens(), 18);
    /// ```
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// 返回文本内容。
    ///
    /// # Returns
    /// 模型最终输出的文本内容。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ChatResponse;
    ///
    /// let response = ChatResponse::new("你好");
    /// assert_eq!(response.content(), "你好");
    /// ```
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// 返回模型的思考过程文本。
    ///
    /// # Returns
    /// 若模型显式暴露了思考过程，则返回该文本。
    #[must_use]
    pub fn thinking_content(&self) -> Option<&str> {
        self.thinking_content.as_deref()
    }

    /// 返回思考阶段消耗的 `token` 数。
    ///
    /// # Returns
    /// 若 Provider 返回了思考阶段用量，则返回该值。
    #[must_use]
    pub const fn thinking_tokens(&self) -> Option<u32> {
        self.thinking_tokens
    }

    /// 返回工具调用列表。
    ///
    /// # Returns
    /// 若模型请求了工具调用，则返回对应切片。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, ToolCall};
    ///
    /// let response = ChatResponse::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert_eq!(response.tool_calls().expect("应包含工具调用")[0].name(), "get_weather");
    /// ```
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }

    /// 返回结束原因。
    ///
    /// # Returns
    /// 若 `Provider` 返回了结束原因，则返回该值。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, FinishReason};
    ///
    /// let response = ChatResponse::new("你好").with_finish_reason(FinishReason::Stop);
    /// assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    /// ```
    #[must_use]
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    /// 返回用量统计。
    ///
    /// # Returns
    /// 若 `Provider` 返回了用量信息，则返回该值。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, Usage};
    ///
    /// let response = ChatResponse::new("你好").with_usage(Usage::new(10, 8));
    /// assert_eq!(response.usage().expect("应包含用量").prompt_tokens(), 10);
    /// ```
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// 返回响应是否包含工具调用。
    ///
    /// # Returns
    /// 如果响应中存在至少一个工具调用，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ChatResponse, ToolCall};
    ///
    /// let response = ChatResponse::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert!(response.has_tool_calls());
    /// ```
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|calls| !calls.is_empty())
    }
}

/// 单个流式响应片段。
///
/// 该结构体用于表达流式输出中的单个增量事件。大多数片段只包含 `delta`，而流尾片段
/// 可能同时携带 `finish_reason`、`usage` 或最终聚合完成的工具调用列表。
///
/// # 示例
/// ```rust
/// use ufox_llm::StreamChunk;
///
/// let chunk = StreamChunk::new("你好");
/// assert_eq!(chunk.delta(), "你好");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamChunk {
    delta: String,
    delta_kind: DeltaKind,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl StreamChunk {
    /// 创建流式增量片段。
    ///
    /// # Arguments
    /// * `delta` - 本次流式事件新增的文本内容
    ///
    /// # Returns
    /// 不包含终止信息的基础流式片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::StreamChunk;
    ///
    /// let chunk = StreamChunk::new("你");
    /// assert_eq!(chunk.delta(), "你");
    /// ```
    #[must_use]
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            delta_kind: DeltaKind::Content,
            tool_calls: None,
            finish_reason: None,
            usage: None,
        }
    }

    /// 创建思考过程流式片段。
    ///
    /// # Arguments
    /// * `delta` - 本次新增的思考过程文本
    ///
    /// # Returns
    /// 标记为思考片段的流式对象。
    #[must_use]
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
    ///
    /// # Arguments
    /// * `tool_calls` - 当前片段附带的完整工具调用集合
    ///
    /// # Returns
    /// 附加工具调用后的流式片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{StreamChunk, ToolCall};
    ///
    /// let chunk = StreamChunk::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert!(chunk.has_tool_calls());
    /// ```
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// 为流式片段附加结束原因。
    ///
    /// # Arguments
    /// * `finish_reason` - 当前片段携带的结束原因
    ///
    /// # Returns
    /// 附加结束原因后的流式片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{FinishReason, StreamChunk};
    ///
    /// let chunk = StreamChunk::new("").with_finish_reason(FinishReason::Stop);
    /// assert!(chunk.is_terminal());
    /// ```
    #[must_use]
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// 为流式片段附加用量信息。
    ///
    /// # Arguments
    /// * `usage` - 当前片段携带的用量统计
    ///
    /// # Returns
    /// 附加用量信息后的流式片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{StreamChunk, Usage};
    ///
    /// let chunk = StreamChunk::new("").with_usage(Usage::new(10, 8));
    /// assert_eq!(chunk.usage().expect("应包含用量").total_tokens(), 18);
    /// ```
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// 返回文本增量。
    ///
    /// # Returns
    /// 当前片段新增的文本内容。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::StreamChunk;
    ///
    /// let chunk = StreamChunk::new("好");
    /// assert_eq!(chunk.delta(), "好");
    /// ```
    #[must_use]
    pub fn delta(&self) -> &str {
        &self.delta
    }

    /// 返回当前片段的内容类型。
    ///
    /// # Returns
    /// 若该片段属于思考过程，则返回 [`DeltaType::Thinking`]；否则返回
    /// [`DeltaType::Content`]。
    #[must_use]
    pub fn delta_type(&self) -> DeltaType {
        match self.delta_kind {
            DeltaKind::Thinking => DeltaType::Thinking(self.delta.clone()),
            DeltaKind::Content => DeltaType::Content(self.delta.clone()),
        }
    }

    /// 返回当前片段是否属于思考过程。
    ///
    /// # Returns
    /// 如果当前片段是思考过程增量，则返回 `true`。
    #[must_use]
    pub const fn is_thinking(&self) -> bool {
        matches!(self.delta_kind, DeltaKind::Thinking)
    }

    /// 返回当前片段附带的工具调用。
    ///
    /// # Returns
    /// 若该片段已经包含完整工具调用，则返回对应切片。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{StreamChunk, ToolCall};
    ///
    /// let chunk = StreamChunk::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert_eq!(chunk.tool_calls().expect("应包含工具调用")[0].id(), "call_1");
    /// ```
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }

    /// 返回当前片段的结束原因。
    ///
    /// # Returns
    /// 若当前片段是终止片段，则可能包含结束原因。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{FinishReason, StreamChunk};
    ///
    /// let chunk = StreamChunk::new("").with_finish_reason(FinishReason::Stop);
    /// assert_eq!(chunk.finish_reason(), Some(&FinishReason::Stop));
    /// ```
    #[must_use]
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    /// 返回当前片段的用量统计。
    ///
    /// # Returns
    /// 若 `Provider` 在尾片段返回了用量信息，则返回该值。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{StreamChunk, Usage};
    ///
    /// let chunk = StreamChunk::new("").with_usage(Usage::new(10, 8));
    /// assert_eq!(chunk.usage().expect("应包含用量").completion_tokens(), 8);
    /// ```
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// 返回当前片段是否包含工具调用。
    ///
    /// # Returns
    /// 如果当前片段中存在至少一个完整工具调用，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{StreamChunk, ToolCall};
    ///
    /// let chunk = StreamChunk::new("")
    ///     .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")]);
    ///
    /// assert!(chunk.has_tool_calls());
    /// ```
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|calls| !calls.is_empty())
    }

    /// 返回当前片段是否为终止片段。
    ///
    /// # Returns
    /// 如果当前片段已经携带结束原因，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{FinishReason, StreamChunk};
    ///
    /// let chunk = StreamChunk::new("").with_finish_reason(FinishReason::Stop);
    /// assert!(chunk.is_terminal());
    /// ```
    #[must_use]
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

        assert_eq!(usage.prompt_tokens(), 15);
        assert_eq!(usage.completion_tokens(), 27);
        assert_eq!(usage.total_tokens(), 42);
    }

    #[test]
    fn response_test() {
        let response = ChatResponse::new("")
            .with_tool_calls(vec![ToolCall::new("call_1", "get_weather", "{}")])
            .with_finish_reason(FinishReason::ToolCalls);

        assert!(response.has_tool_calls());
        assert_eq!(
            response.tool_calls().expect("应包含工具调用")[0].name(),
            "get_weather"
        );
        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn response_test_2() {
        let chunk = StreamChunk::new("")
            .with_finish_reason(FinishReason::Stop)
            .with_usage(Usage::new(10, 8));

        assert!(chunk.is_terminal());
        assert_eq!(chunk.usage().expect("应包含用量").total_tokens(), 18);
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

        assert_eq!(response.thinking_content(), Some("先分析条件"));
        assert_eq!(response.thinking_tokens(), Some(42));
        assert_eq!(response.content(), "最终答案");
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
