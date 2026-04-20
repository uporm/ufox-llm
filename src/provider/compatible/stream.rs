//! 兼容 `OpenAI` 协议的流式响应解析模块。
//!
//! 该模块负责为 `Compatible` 供应商暴露流式解析能力。由于这类供应商在设计目标上
//! 与 `OpenAI Chat Completions` 协议保持兼容，因此这里直接复用 `OpenAI` 的流式
//! 解析实现，而不重复维护一份行为等价的解析器。
//!
//! 设计上采用“包装器 + 函数转发”的方式：
//! 1. [`CompatibleStreamParser`] 内部复用 [`OpenAiStreamParser`] 的状态机实现；
//! 2. [`is_done_event`] 直接复用 `OpenAI` 的 `[DONE]` 终止标记判断；
//! 3. 包装器额外负责把内部错误里的 `OpenAI` 字样替换为 `Compatible`，
//!    这样排障时不会误以为自己正在接官方 `OpenAI` 服务。
//!
//! 该模块依赖上级 `openai::stream` 子模块提供的流式解析能力。

use crate::{LlmError, StreamChunk};

use crate::provider::openai::stream::OpenAiStreamParser;

/// 兼容 `OpenAI` 协议的流式解析器。
///
/// 该解析器内部直接复用 [`OpenAiStreamParser`] 的实现，但会在错误消息层面改写
/// Provider 名称，避免对接第三方兼容服务时出现误导性的 `OpenAI` 提示。
///
/// # 示例
/// ```rust
/// use ufox_llm::provider::compatible::stream::CompatibleStreamParser;
///
/// let mut parser = CompatibleStreamParser::new();
/// let chunk = parser
///     .parse_event(r#"{"choices":[{"delta":{"content":"你"},"finish_reason":null}]}"#)
///     .expect("事件应解析成功")
///     .expect("应产出文本增量");
///
/// assert_eq!(chunk.delta(), "你");
/// ```
#[derive(Debug, Default)]
pub struct CompatibleStreamParser {
    inner: OpenAiStreamParser,
}

impl CompatibleStreamParser {
    /// 创建兼容协议流式解析器。
    ///
    /// # Returns
    /// 空状态的兼容协议解析器。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 重置内部累积状态。
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// 解析单条 `SSE` 事件的 `data:` 文本。
    ///
    /// # Arguments
    /// * `event_data` - 单条 `SSE` 事件的数据部分
    ///
    /// # Returns
    /// 与 [`OpenAiStreamParser::parse_event`] 相同的公共流式解析结果。若同一事件中包含多个
    /// 逻辑片段，则返回最后一个片段。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    pub fn parse_event(&mut self, event_data: &str) -> Result<Option<StreamChunk>, LlmError> {
        self.inner
            .parse_event(event_data)
            .map_err(rewrite_provider_in_stream_error)
    }

    /// 解析单条 `SSE` 事件，并返回其中包含的全部公共流式片段。
    ///
    /// 该方法直接复用内部 `OpenAI-compatible` 解析逻辑，但会保证错误消息中的 Provider
    /// 名称保持为 `Compatible`。
    ///
    /// # Arguments
    /// * `event_data` - 单条 `SSE` 事件的数据部分
    ///
    /// # Returns
    /// 当前事件解析得到的零个或多个流式片段。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::compatible::stream::CompatibleStreamParser;
    ///
    /// let mut parser = CompatibleStreamParser::new();
    /// let chunks = parser
    ///     .parse_event_chunks(
    ///         r#"{"choices":[{"delta":{"reasoning_content":"先分析","content":"再回答"},"finish_reason":null}]}"#,
    ///     )
    ///     .expect("事件应解析成功");
    ///
    /// assert_eq!(chunks.len(), 2);
    /// assert!(chunks[0].is_thinking());
    /// assert_eq!(chunks[1].delta(), "再回答");
    /// ```
    pub fn parse_event_chunks(
        &mut self,
        event_data: &str,
    ) -> Result<Vec<StreamChunk>, LlmError> {
        self.inner
            .parse_event_chunks(event_data)
            .map_err(rewrite_provider_in_stream_error)
    }
}

/// 判断事件是否为兼容 `OpenAI` 协议的 `[DONE]` 终止标记。
///
/// # Arguments
/// * `event_data` - 单条 `SSE` 事件的数据部分
///
/// # Returns
/// 若事件内容为 `[DONE]`，则返回 `true`。
///
/// # 示例
/// ```rust
/// use ufox_llm::provider::compatible::stream::is_done_event;
///
/// assert!(is_done_event("[DONE]"));
/// assert!(!is_done_event("{}"));
/// ```
#[must_use]
pub fn is_done_event(event_data: &str) -> bool {
    crate::provider::openai::stream::is_done_event(event_data)
}

fn rewrite_provider_in_stream_error(error: LlmError) -> LlmError {
    match error {
        LlmError::StreamError(message) => {
            LlmError::StreamError(message.replace("OpenAI", "Compatible"))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{CompatibleStreamParser, is_done_event};

    #[test]
    fn compatible_openai() {
        let mut parser = CompatibleStreamParser::new();
        let chunk = parser
            .parse_event(r#"{"choices":[{"delta":{"content":"你"},"finish_reason":null}]}"#)
            .expect("事件应解析成功")
            .expect("应产出文本增量");

        assert_eq!(chunk.delta(), "你");
        assert!(is_done_event("[DONE]"));
    }

    #[test]
    fn compatible() {
        let mut parser = CompatibleStreamParser::new();
        let chunks = parser
            .parse_event_chunks(
                r#"{"choices":[{"delta":{"reasoning_content":"先分析","content":"再回答"},"finish_reason":null}]}"#,
            )
            .expect("事件应解析成功");

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].is_thinking());
        assert_eq!(chunks[1].delta(), "再回答");
    }
}
