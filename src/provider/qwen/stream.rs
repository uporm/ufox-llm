//! `Qwen` 流式解析。
//!
//! 解析单条 `SSE` 事件并转换为公共 `StreamChunk`。

use serde::Deserialize;
use serde_json::Value;

use crate::{FinishReason, LlmError, StreamChunk, ToolCall, Usage};

/// `Qwen` 流式事件解析器。
#[derive(Debug, Default)]
pub struct QwenStreamParser {
    pending_tool_calls: Vec<PartialToolCall>,
}

impl QwenStreamParser {
    /// 创建流式事件解析器。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 重置内部累积状态。
    pub fn reset(&mut self) {
        self.pending_tool_calls.clear();
    }

    /// 解析单条 `SSE` 事件的 `data:` 文本。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    pub fn parse_event(&mut self, event_data: &str) -> Result<Option<StreamChunk>, LlmError> {
        Ok(self.parse_event_chunks(event_data)?.into_iter().last())
    }

    /// 解析单条 `SSE` 事件并返回其中全部片段。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件缺少必要字段或工具调用碎片不完整时触发
    pub fn parse_event_chunks(&mut self, event_data: &str) -> Result<Vec<StreamChunk>, LlmError> {
        if is_done_event(event_data) {
            self.reset();
            return Ok(Vec::new());
        }

        let response: QwenStreamResponse = serde_json::from_str(event_data)?;

        if response.output.choices.is_empty() {
            return Ok(response
                .usage
                .map(|usage| {
                    vec![
                        StreamChunk::new("")
                            .with_usage(Usage::new(usage.input_tokens, usage.output_tokens)),
                    ]
                })
                .unwrap_or_default());
        }

        let choice =
            response.output.choices.into_iter().next().ok_or_else(|| {
                LlmError::StreamError("Qwen 流式事件缺少 output.choices".to_string())
            })?;

        let reasoning_delta = choice.message.reasoning_content.unwrap_or_default();
        let delta_text = choice.message.content.into_text();
        if let Some(tool_calls) = choice.message.tool_calls {
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
            .map(|usage| Usage::new(usage.input_tokens, usage.output_tokens))
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

    fn merge_tool_call_deltas(&mut self, deltas: Vec<QwenToolCallDelta>) {
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
                        "Qwen 工具调用流式片段不完整：索引 {index} 缺少 id 或 name"
                    )));
                }

                Ok(ToolCall::new(partial.id, partial.name, partial.arguments))
            })
            .collect()
    }
}

/// 判断事件是否为 `[DONE]` 终止标记。
#[must_use]
pub fn is_done_event(event_data: &str) -> bool {
    event_data.trim() == "[DONE]"
}

#[derive(Debug, Deserialize)]
struct QwenStreamResponse {
    output: QwenOutput,
    #[serde(default)]
    usage: Option<QwenUsage>,
}

#[derive(Debug, Deserialize)]
struct QwenOutput {
    #[serde(default)]
    choices: Vec<QwenChoice>,
}

#[derive(Debug, Deserialize)]
struct QwenChoice {
    message: QwenMessageDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QwenMessageDelta {
    #[serde(default)]
    content: QwenDeltaContent,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<QwenToolCallDelta>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
enum QwenDeltaContent {
    Text(String),
    Parts(Vec<QwenDeltaContentPart>),
    #[default]
    Empty,
}

impl QwenDeltaContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .map(QwenDeltaContentPart::into_text)
                .collect::<String>(),
            Self::Empty => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum QwenDeltaContentPart {
    Text { text: String },
    Refusal { refusal: String },
    Image { image: String },
    Other(Value),
}

impl QwenDeltaContentPart {
    fn into_text(self) -> String {
        match self {
            Self::Text { text } => text,
            Self::Refusal { refusal } => refusal,
            Self::Image { image } => {
                let _ = image;
                String::new()
            }
            Self::Other(value) => {
                let _ = value;
                String::new()
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct QwenToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<QwenToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct QwenToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QwenUsage {
    input_tokens: u32,
    output_tokens: u32,
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

    use super::{QwenStreamParser, is_done_event};
    use crate::FinishReason;

    #[test]
    fn done_chunk() {
        let mut parser = QwenStreamParser::new();

        let chunk = parser.parse_event("[DONE]").expect("事件应解析成功");

        assert!(chunk.is_none());
        assert!(is_done_event("[DONE]"));
    }

    #[test]
    fn stream_test() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": "你"
                        },
                        "finish_reason": null
                    }
                ]
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出增量");

        assert_eq!(chunk.delta(), "你");
        assert!(!chunk.is_terminal());
    }

    #[test]
    fn stop() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": ""
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出尾片段");

        assert_eq!(chunk.finish_reason(), Some(&FinishReason::Stop));
        assert!(chunk.is_terminal());
    }

    #[test]
    fn stream_test_2() {
        let mut parser = QwenStreamParser::new();
        let first = json!({
            "output": {
                "choices": [
                    {
                        "message": {
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
            }
        })
        .to_string();
        let second = json!({
            "output": {
                "choices": [
                    {
                        "message": {
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
            }
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
            "output": {
                "choices": []
            },
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出 usage 尾片段");

        assert_eq!(chunk.usage().expect("应包含 usage").total_tokens(), 20);
    }

    #[test]
    fn reasoning_content_qwen() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "先分析"
                        },
                        "finish_reason": null
                    }
                ]
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应产出思考增量");

        assert!(chunk.is_thinking());
        assert_eq!(chunk.delta(), "先分析");
    }

    #[test]
    fn qwen() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "先分析",
                            "content": "最终答案"
                        },
                        "finish_reason": null
                    }
                ]
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunks = parser.parse_event_chunks(&body).expect("事件应解析成功");

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].is_thinking());
        assert_eq!(chunks[0].delta(), "先分析");
        assert_eq!(chunks[1].delta(), "最终答案");
        assert!(!chunks[1].is_thinking());
    }

    #[test]
    fn parse_event_qwen() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "先分析",
                            "content": "最终答案"
                        },
                        "finish_reason": null
                    }
                ]
            }
        })
        .to_string();
        let mut parser = QwenStreamParser::new();

        let chunk = parser
            .parse_event(&body)
            .expect("事件应解析成功")
            .expect("应返回最后片段");

        assert!(!chunk.is_thinking());
        assert_eq!(chunk.delta(), "最终答案");
    }
}
