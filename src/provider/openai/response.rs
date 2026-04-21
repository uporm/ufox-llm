//! `OpenAI` 响应反序列化。
//!
//! 将 OpenAI 非流式响应解析为公共响应模型。

use serde::Deserialize;
use serde_json::Value;

use crate::{ChatResponse, FinishReason, LlmError, ToolCall, Usage};

/// 将 `OpenAI` 非流式响应体解析为公共聊天响应。
/// # Errors
/// - [`LlmError::ParseError`]：当响应体不是合法 `JSON` 时触发
/// - [`LlmError::ApiError`]：当响应体是 `OpenAI` 错误对象，或缺少必要字段时触发
pub fn parse_chat_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
    parse_chat_response_with_provider(body, "OpenAI")
}

pub(crate) fn parse_chat_response_with_provider(
    body: &[u8],
    provider_name: &str,
) -> Result<ChatResponse, LlmError> {
    if let Ok(error_response) = serde_json::from_slice::<OpenAiErrorEnvelope>(body) {
        return Err(LlmError::ApiError {
            status_code: 0,
            message: error_response.error.message,
            provider: provider_name.to_string(),
        });
    }

    let response: OpenAiChatResponse = serde_json::from_slice(body)?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::ApiError {
            status_code: 0,
            message: format!("{provider_name} 响应中缺少 choices"),
            provider: provider_name.to_string(),
        })?;

    let mut chat_response = ChatResponse::new(choice.message.content.into_text());

    if let Some(reasoning_content) = choice
        .message
        .reasoning_content
        .filter(|text| !text.is_empty())
    {
        chat_response = chat_response.with_thinking_content(reasoning_content);
    }

    if let Some(tool_calls) = choice.message.tool_calls.map(convert_tool_calls) {
        chat_response = chat_response.with_tool_calls(tool_calls);
    }

    if let Some(finish_reason) = choice.finish_reason.map(FinishReason::from) {
        chat_response = chat_response.with_finish_reason(finish_reason);
    }

    if let Some(reasoning_tokens) = response.usage.as_ref().and_then(|usage| {
        usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
    }) {
        chat_response = chat_response.with_thinking_tokens(reasoning_tokens);
    }

    if let Some(usage) = response
        .usage
        .map(|usage| Usage::new(usage.prompt_tokens, usage.completion_tokens))
    {
        chat_response = chat_response.with_usage(usage);
    }

    Ok(chat_response)
}

fn convert_tool_calls(tool_calls: Vec<OpenAiToolCall>) -> Vec<ToolCall> {
    tool_calls
        .into_iter()
        .map(|tool_call| {
            ToolCall::new(
                tool_call.id,
                tool_call.function.name,
                tool_call.function.arguments,
            )
        })
        .collect()
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiAssistantMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiAssistantMessage {
    #[serde(default)]
    content: OpenAiAssistantContent,
    #[serde(default)]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
enum OpenAiAssistantContent {
    Text(String),
    Parts(Vec<OpenAiAssistantContentPart>),
    #[default]
    Empty,
}

impl OpenAiAssistantContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .map(OpenAiAssistantContentPart::into_text)
                .collect::<String>(),
            Self::Empty => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiAssistantContentPart {
    Text {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Unsupported,
}

impl OpenAiAssistantContentPart {
    fn into_text(self) -> String {
        match self {
            Self::Text { text } => text,
            // 将 refusal 一并保留到文本中，可以避免调用方在解析非标准回复时丢失模型输出。
            Self::Refusal { refusal } => refusal,
            Self::Unsupported => String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiToolFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    #[serde(default)]
    completion_tokens_details: Option<OpenAiCompletionTokenDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorBody {
    message: String,
    #[allow(dead_code)]
    #[serde(default)]
    r#type: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    param: Option<Value>,
    #[allow(dead_code)]
    #[serde(default)]
    code: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::parse_chat_response;
    use crate::FinishReason;

    #[test]
    fn chat_response() {
        let body = json!({
            "choices": [
                {
                    "message": {
                        "content": "你好，我可以帮你审查代码。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.content(), "你好，我可以帮你审查代码。");
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
        assert_eq!(response.usage().expect("应包含用量").total_tokens(), 20);
    }

    #[test]
    fn response_test() {
        let body = json!({
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
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "第一段。"
                            },
                            {
                                "type": "refusal",
                                "refusal": "第二段。"
                            }
                        ]
                    },
                    "finish_reason": "stop"
                }
            ]
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.content(), "第一段。第二段。");
    }

    #[test]
    fn reasoning_content() {
        let body = json!({
            "choices": [
                {
                    "message": {
                        "content": "最终答案",
                        "reasoning_content": "先分析问题"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "completion_tokens_details": {
                    "reasoning_tokens": 12
                }
            }
        })
        .to_string();

        let response = parse_chat_response(body.as_bytes()).expect("响应应解析成功");

        assert_eq!(response.thinking_content(), Some("先分析问题"));
        assert_eq!(response.thinking_tokens(), Some(12));
        assert_eq!(response.content(), "最终答案");
    }

    #[test]
    fn api_error() {
        let body = json!({
            "error": {
                "message": "模型不可用",
                "type": "invalid_request_error"
            }
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
                assert_eq!(message, "模型不可用");
                assert_eq!(provider, "OpenAI");
            }
            other => panic!("错误类型不符合预期：{other:?}"),
        }
    }
}
