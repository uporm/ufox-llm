//! `Qwen` 响应反序列化模块。
//!
//! 该模块负责将 `DashScope` / `Qwen` 的非流式响应体解析为 SDK 公共响应模型。
//!
//! 设计上采用“先解析私有结构，再映射公共类型”的方式：
//! 1. 先以 `Qwen` 私有结构体承接响应，避免公共类型直接耦合 `output`、`usage`
//!    等供应商特有字段层级；
//! 2. 再把首个 `choice` 转换为 [`ChatResponse`]，统一提取文本内容、工具调用、
//!    结束原因与用量信息；
//! 3. 当响应体包含业务错误对象时，优先转换为 [`LlmError::ApiError`]，让上层得到
//!    稳定的错误语义。
//!
//! 该模块依赖 `serde_json` 解析原始响应体，并复用 `types` 模块中的响应与工具调用类型。

use serde::Deserialize;
use serde_json::Value;

use crate::{ChatResponse, FinishReason, LlmError, ToolCall, Usage};

/// 将 `Qwen` 非流式响应体解析为公共聊天响应。
///
/// # Arguments
/// * `body` - `Qwen` 返回的原始响应体字节
///
/// # Returns
/// 统一后的聊天响应对象。
///
/// # Errors
/// - [`LlmError::ParseError`]：当响应体不是合法 `JSON` 时触发
/// - [`LlmError::ApiError`]：当响应体是 `Qwen` 错误对象，或缺少必要字段时触发
pub fn parse_chat_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
    if let Ok(error_response) = serde_json::from_slice::<QwenErrorEnvelope>(body) {
        return Err(LlmError::ApiError {
            status_code: 0,
            message: error_response.message,
            provider: "Qwen".to_string(),
        });
    }

    let response: QwenChatResponse = serde_json::from_slice(body)?;
    let choice = response
        .output
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::ApiError {
            status_code: 0,
            message: "Qwen 响应中缺少 output.choices".to_string(),
            provider: "Qwen".to_string(),
        })?;

    let mut chat_response = ChatResponse::new(choice.message.content.into_text());

    if let Some(reasoning_content) = choice.message.reasoning_content.filter(|text| !text.is_empty()) {
        chat_response = chat_response.with_thinking_content(reasoning_content);
    }

    if let Some(tool_calls) = choice.message.tool_calls.map(convert_tool_calls) {
        chat_response = chat_response.with_tool_calls(tool_calls);
    }

    if let Some(finish_reason) = choice.finish_reason.map(FinishReason::from) {
        chat_response = chat_response.with_finish_reason(finish_reason);
    }

    if let Some(thinking_tokens) = response.usage.as_ref().and_then(QwenUsage::thinking_tokens) {
        chat_response = chat_response.with_thinking_tokens(thinking_tokens);
    }

    if let Some(usage) = response
        .usage
        .map(|usage| Usage::new(usage.input, usage.output))
    {
        chat_response = chat_response.with_usage(usage);
    }

    Ok(chat_response)
}

fn convert_tool_calls(tool_calls: Vec<QwenToolCall>) -> Vec<ToolCall> {
    tool_calls
        .into_iter()
        .map(|tool_call| ToolCall::new(tool_call.id, tool_call.function.name, tool_call.function.arguments))
        .collect()
}

#[derive(Debug, Deserialize)]
struct QwenChatResponse {
    output: QwenOutput,
    usage: Option<QwenUsage>,
}

#[derive(Debug, Deserialize)]
struct QwenOutput {
    choices: Vec<QwenChoice>,
}

#[derive(Debug, Deserialize)]
struct QwenChoice {
    message: QwenAssistantMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QwenAssistantMessage {
    #[serde(default)]
    content: QwenAssistantContent,
    #[serde(default)]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<QwenToolCall>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
enum QwenAssistantContent {
    Text(String),
    Parts(Vec<QwenAssistantContentPart>),
    #[default]
    Empty,
}

impl QwenAssistantContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .map(QwenAssistantContentPart::into_text)
                .collect::<String>(),
            Self::Empty => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum QwenAssistantContentPart {
    Text {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    Image {
        image: String,
    },
    Other(Value),
}

impl QwenAssistantContentPart {
    fn into_text(self) -> String {
        match self {
            Self::Text { text } => text,
            Self::Refusal { refusal } => refusal,
            // 图片片段本身不应注入文本输出，否则会污染上层最终内容。
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
struct QwenToolCall {
    id: String,
    function: QwenToolFunction,
}

#[derive(Debug, Deserialize)]
struct QwenToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct QwenUsage {
    #[serde(rename = "input_tokens")]
    input: u32,
    #[serde(rename = "output_tokens")]
    output: u32,
    #[serde(default)]
    #[serde(rename = "reasoning_tokens")]
    reasoning: Option<u32>,
    #[serde(default)]
    #[serde(rename = "thinking_tokens")]
    thinking: Option<u32>,
}

impl QwenUsage {
    fn thinking_tokens(&self) -> Option<u32> {
        self.reasoning.or(self.thinking)
    }
}

#[derive(Debug, Deserialize)]
struct QwenErrorEnvelope {
    message: String,
    #[allow(dead_code)]
    #[serde(default)]
    code: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::parse_chat_response;
    use crate::FinishReason;

    #[test]
    fn chat_response() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": "你好，我可以帮你分析代码。"
                        },
                        "finish_reason": "stop"
                    }
                ]
            },
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20
            }
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.content(), "你好，我可以帮你分析代码。");
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
        assert_eq!(response.usage().expect("应包含用量").total_tokens(), 20);
    }

    #[test]
    fn response_test() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": null,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{\"city\":\"杭州\"}"
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

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert!(response.has_tool_calls());
        assert_eq!(
            response.tool_calls().expect("应包含工具调用")[0].name(),
            "get_weather"
        );
        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn response_test_2() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "text": "第一段。"
                                },
                                {
                                    "refusal": "第二段。"
                                },
                                {
                                    "image": "https://example.com/photo.jpg"
                                }
                            ]
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.content(), "第一段。第二段。");
    }

    #[test]
    fn api_error() {
        let body = json!({
            "code": "InvalidApiKey",
            "message": "API Key 无效",
            "request_id": "req-123"
        })
        .to_string();

        let error = parse_chat_response(body.as_bytes()).expect_err("应返回错误");

        match error {
            crate::LlmError::ApiError {
                status_code,
                message,
                provider,
            } => {
                assert_eq!(status_code, 0);
                assert_eq!(message, "API Key 无效");
                assert_eq!(provider, "Qwen");
            }
            other => panic!("错误类型不符合预期：{other:?}"),
        }
    }

    #[test]
    fn reasoning_content_qwen() {
        let body = json!({
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": "最终答案",
                            "reasoning_content": "先分析题意"
                        },
                        "finish_reason": "stop"
                    }
                ]
            },
            "usage": {
                "input_tokens": 12,
                "output_tokens": 20,
                "reasoning_tokens": 9
            }
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.thinking_content(), Some("先分析题意"));
        assert_eq!(response.thinking_tokens(), Some(9));
        assert_eq!(response.content(), "最终答案");
    }
}
