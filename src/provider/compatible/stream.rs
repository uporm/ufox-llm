//! Compatible 流式解析。
//!
//! 复用 OpenAI-compatible 解析逻辑并改写 Provider 错误名。

use crate::{LlmError, StreamChunk};

use crate::provider::openai::OpenAiStreamParser;

/// 兼容 `OpenAI` 协议的流式解析器。
#[derive(Debug, Default)]
pub struct CompatibleStreamParser {
    inner: OpenAiStreamParser,
}

impl CompatibleStreamParser {
    /// 创建兼容协议流式解析器。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 重置内部累积状态。
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// 解析单条 `SSE` 事件的 `data:` 文本。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    pub fn parse_event(&mut self, event_data: &str) -> Result<Option<StreamChunk>, LlmError> {
        self.inner
            .parse_event(event_data)
            .map_err(rewrite_provider_in_stream_error)
    }

    /// 解析单条 `SSE` 事件并返回其中全部片段。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    pub fn parse_event_chunks(&mut self, event_data: &str) -> Result<Vec<StreamChunk>, LlmError> {
        self.inner
            .parse_event_chunks(event_data)
            .map_err(rewrite_provider_in_stream_error)
    }
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
    use super::CompatibleStreamParser;

    #[test]
    fn compatible_openai() {
        let mut parser = CompatibleStreamParser::new();
        let chunk = parser
            .parse_event(r#"{"choices":[{"delta":{"content":"你"},"finish_reason":null}]}"#)
            .expect("事件应解析成功")
            .expect("应产出文本增量");

        assert_eq!(chunk.delta, "你");
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
        assert_eq!(chunks[1].delta, "再回答");
    }
}
