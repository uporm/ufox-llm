//! `OpenAI` 流式响应解析模块。
//!
//! 该模块负责解析 `OpenAI Chat Completions` 的单条 `SSE` 事件数据，并将其转换为
//! SDK 公共的 [`StreamChunk`]。
//!
//! 设计上采用“有状态增量解析器”：
//! 1. `data: [DONE]` 这类终止事件本身不产出增量内容，因此直接返回 `None`；
//! 2. 普通文本增量可以即时映射为 [`StreamChunk`]；
//! 3. 工具调用参数在 `OpenAI` 流中常常会被拆成多段，因此解析器内部维护累积状态，
//!    只有在收到 `finish_reason = "tool_calls"` 时，才输出可对外消费的完整工具调用。
//!
//! 该模块依赖 `serde_json` 解析单条事件数据，并复用 `types` 模块中的流式响应与工具调用类型。

use serde::Deserialize;

use crate::{FinishReason, LlmError, StreamChunk, ToolCall, Usage};

/// `OpenAI` 流式事件解析器。
///
/// 该解析器按事件顺序消费 `SSE` 的 `data:` 文本，并在需要时维护工具调用碎片的中间状态。
///
/// # 示例
/// ```rust
/// use ufox_llm::provider::openai::stream::OpenAiStreamParser;
///
/// let mut parser = OpenAiStreamParser::new();
/// let chunk = parser
///     .parse_event(r#"{"choices":[{"delta":{"content":"你"},"finish_reason":null}]}"#)
///     .expect("事件应解析成功")
///     .expect("应产出文本增量");
///
/// assert_eq!(chunk.delta(), "你");
/// ```
#[derive(Debug, Default)]
pub struct OpenAiStreamParser {
    pending_tool_calls: Vec<PartialToolCall>,
}

impl OpenAiStreamParser {
    /// 创建流式事件解析器。
    ///
    /// # Returns
    /// 空状态的 `OpenAI` 流式解析器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::openai::stream::OpenAiStreamParser;
    ///
    /// let parser = OpenAiStreamParser::new();
    /// assert_eq!(format!("{parser:?}"), "OpenAiStreamParser { pending_tool_calls: [] }");
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 重置内部累积状态。
    ///
    /// 当同一个解析器实例被复用于下一次独立流式请求前，应先调用该方法，以避免上一次
    /// 未消费完的工具调用碎片污染新请求。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::openai::stream::OpenAiStreamParser;
    ///
    /// let mut parser = OpenAiStreamParser::new();
    /// parser.reset();
    /// ```
    pub fn reset(&mut self) {
        self.pending_tool_calls.clear();
    }

    /// 解析单条 `SSE` 事件的 `data:` 文本。
    ///
    /// # Arguments
    /// * `event_data` - 单条 `SSE` 事件的数据部分，不包含 `data:` 前缀
    ///
    /// # Returns
    /// - `Ok(None)`：表示该事件不产生可消费的公共增量，例如 `[DONE]` 或仅包含角色信息
    /// - `Ok(Some(StreamChunk))`：表示成功产出一条公共流式增量
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::openai::stream::OpenAiStreamParser;
    ///
    /// let mut parser = OpenAiStreamParser::new();
    /// let chunk = parser
    ///     .parse_event(r#"{"choices":[{"delta":{"content":"好"},"finish_reason":null}]}"#)
    ///     .expect("事件应解析成功")
    ///     .expect("应产出文本增量");
    ///
    /// assert_eq!(chunk.delta(), "好");
    /// ```
    /// 若同一事件中包含多个逻辑片段，该方法会返回最后一个片段。通常这意味着正文增量或
    /// 携带 `finish_reason`、`usage`、`tool_calls` 的尾片段。
    pub fn parse_event(&mut self, event_data: &str) -> Result<Option<StreamChunk>, LlmError> {
        Ok(self.parse_event_chunks(event_data)?.into_iter().last())
    }

    /// 解析单条 `SSE` 事件，并返回其中包含的全部公共流式片段。
    ///
    /// 某些模型可能在同一条事件里同时返回思考文本和正式回复文本。此时
    /// [`OpenAiStreamParser::parse_event`] 只会返回最后一个片段，而该方法会完整返回
    /// 当前事件中的全部片段。
    ///
    /// # Arguments
    /// * `event_data` - 单条 `SSE` 事件的数据部分，不包含 `data:` 前缀
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
    /// use ufox_llm::provider::openai::stream::OpenAiStreamParser;
    ///
    /// let mut parser = OpenAiStreamParser::new();
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
        if is_done_event(event_data) {
            self.reset();
            return Ok(Vec::new());
        }

        let response: OpenAiStreamResponse = serde_json::from_str(event_data)?;

        if response.choices.is_empty() {
            return Ok(response
                .usage
                .map(|usage| {
                    vec![
                        StreamChunk::new("")
                            .with_usage(Usage::new(usage.prompt_tokens, usage.completion_tokens)),
                    ]
                })
                .unwrap_or_default());
        }

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::StreamError("OpenAI 流式事件缺少 choices".to_string()))?;

        let reasoning_delta = choice.delta.reasoning_content.unwrap_or_default();
        let delta_text = choice.delta.content.into_text();

        if let Some(tool_calls) = choice.delta.tool_calls {
            self.merge_tool_call_deltas(tool_calls);
        }

        let mut chunks = Vec::new();
        if !reasoning_delta.is_empty() {
            chunks.push(StreamChunk::thinking(reasoning_delta));
        }
        if !delta_text.is_empty() {
            chunks.push(StreamChunk::new(delta_text));
        }

        let mut has_effective_data = chunks.iter().any(|chunk| !chunk.delta().is_empty());
        if chunks.is_empty() {
            chunks.push(StreamChunk::new(""));
        }

        if let Some(finish_reason) = choice.finish_reason.map(FinishReason::from) {
            if finish_reason.is_tool_calls() {
                let tool_calls = self.take_completed_tool_calls()?;
                if !tool_calls.is_empty() {
                    let chunk = chunks
                        .last_mut()
                        .expect("chunks 在附加工具调用前一定至少有一个元素");
                    *chunk = chunk.clone().with_tool_calls(tool_calls);
                }
            }

            let chunk = chunks
                .last_mut()
                .expect("chunks 在附加结束原因前一定至少有一个元素");
            *chunk = chunk.clone().with_finish_reason(finish_reason);
            has_effective_data = true;
        }

        if let Some(usage) = response
            .usage
            .map(|usage| Usage::new(usage.prompt_tokens, usage.completion_tokens))
        {
            let chunk = chunks
                .last_mut()
                .expect("chunks 在附加 usage 前一定至少有一个元素");
            *chunk = chunk.clone().with_usage(usage);
            has_effective_data = true;
        }

        if has_effective_data {
            Ok(chunks)
        } else {
            Ok(Vec::new())
        }
    }

    fn merge_tool_call_deltas(&mut self, deltas: Vec<OpenAiToolCallDelta>) {
        for delta in deltas {
            let index = delta.index;
            while self.pending_tool_calls.len() <= index {
                self.pending_tool_calls.push(PartialToolCall::default());
            }

            let entry = &mut self.pending_tool_calls[index];
            if let Some(id) = delta.id {
                entry.id.push_str(&id);
            }

            if let Some(function) = delta.function {
                if let Some(name) = function.name {
                    entry.name.push_str(&name);
                }

                if let Some(arguments) = function.arguments {
                    entry.arguments.push_str(&arguments);
                }
            }
        }
    }

    fn take_completed_tool_calls(&mut self) -> Result<Vec<ToolCall>, LlmError> {
        let pending = std::mem::take(&mut self.pending_tool_calls);
        pending
            .into_iter()
            .enumerate()
            .map(|(index, partial)| {
                if partial.id.is_empty() || partial.name.is_empty() {
                    return Err(LlmError::StreamError(format!(
                        "OpenAI 工具调用流式片段不完整：索引 {index} 缺少 id 或 name"
                    )));
                }

                Ok(ToolCall::new(partial.id, partial.name, partial.arguments))
            })
            .collect()
    }
}

/// 判断事件是否为 `OpenAI` 的 `[DONE]` 终止标记。
///
/// # Arguments
/// * `event_data` - 单条 `SSE` 事件的数据部分
///
/// # Returns
/// 若事件内容为 `[DONE]`，则返回 `true`。
///
/// # 示例
/// ```rust
/// use ufox_llm::provider::openai::stream::is_done_event;
///
/// assert!(is_done_event("[DONE]"));
/// assert!(!is_done_event("{}"));
/// ```
#[must_use]
pub fn is_done_event(event_data: &str) -> bool {
    event_data.trim() == "[DONE]"
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamResponse {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: OpenAiDeltaContent,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
enum OpenAiDeltaContent {
    Text(String),
    Parts(Vec<OpenAiDeltaContentPart>),
    #[default]
    Empty,
}

impl OpenAiDeltaContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .map(OpenAiDeltaContentPart::into_text)
                .collect::<String>(),
            Self::Empty => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiDeltaContentPart {
    Text {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Unsupported,
}

impl OpenAiDeltaContentPart {
    fn into_text(self) -> String {
        match self {
            Self::Text { text } => text,
            Self::Refusal { refusal } => refusal,
            Self::Unsupported => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{OpenAiStreamParser, is_done_event};
    use crate::FinishReason;

    #[test]
    fn done_chunk() {
        let mut parser = OpenAiStreamParser::new();

        let chunk = parser.parse_event("[DONE]").expect("事件应解析成功");

        assert!(chunk.is_none());
        assert!(is_done_event("[DONE]"));
    }

    #[test]
    fn stream_test() {
        let body = json!({
            "choices": [
                {
                    "delta": {
                        "content": "你"
                    },
                    "finish_reason": null
                }
            ]
        })
        .to_string();
        let mut parser = OpenAiStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出增量");

        assert_eq!(chunk.delta(), "你");
        assert!(!chunk.is_terminal());
    }

    #[test]
    fn stream_test_2() {
        let mut parser = OpenAiStreamParser::new();
        let first = json!({
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":"
                                }
                            }
                        ]
                    },
                    "finish_reason": null
                }
            ]
        })
        .to_string();
        let second = json!({
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "\"杭州\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        })
        .to_string();

        assert!(
            parser
                .parse_event(&first)
                .expect("第一段应解析成功")
                .is_none()
        );

        let chunk = parser
            .parse_event(&second)
            .expect("第二段应解析成功")
            .expect("应输出完整工具调用");

        assert_eq!(chunk.finish_reason(), Some(&FinishReason::ToolCalls));
        assert_eq!(
            chunk.tool_calls().expect("应包含工具调用")[0].arguments(),
            "{\"city\":\"杭州\"}"
        );
    }

    #[test]
    fn usage() {
        let body = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        })
        .to_string();
        let mut parser = OpenAiStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出 usage 尾片段");

        assert_eq!(chunk.usage().expect("应包含 usage").total_tokens(), 20);
    }

    #[test]
    fn reasoning_content() {
        let body = json!({
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "先分析"
                    },
                    "finish_reason": null
                }
            ]
        })
        .to_string();
        let mut parser = OpenAiStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出思考增量");

        assert!(chunk.is_thinking());
        assert_eq!(chunk.delta(), "先分析");
    }

    #[test]
    fn stream_test_3() {
        let body = json!({
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "先分析",
                        "content": "最终答案"
                    },
                    "finish_reason": null
                }
            ]
        })
        .to_string();
        let mut parser = OpenAiStreamParser::new();

        let chunks = parser
            .parse_event_chunks(&body)
            .expect("事件应解析成功");

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].is_thinking());
        assert_eq!(chunks[0].delta(), "先分析");
        assert_eq!(chunks[1].delta(), "最终答案");
        assert!(!chunks[1].is_thinking());
    }

    #[test]
    fn parse_event_2() {
        let body = json!({
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "先分析",
                        "content": "最终答案"
                    },
                    "finish_reason": null
                }
            ]
        })
        .to_string();
        let mut parser = OpenAiStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应返回最后片段");

        assert!(!chunk.is_thinking());
        assert_eq!(chunk.delta(), "最终答案");
    }
}
