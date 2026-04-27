//! Chat Completions 协议 adapter。
//!
//! 实现 [`ProviderAdapter`] trait，使用 OpenAI Chat Completions API
//! (`POST /chat/completions`) 处理对话请求。
//!
//! 本模块职责：
//! - 将通用 [`Message`] / [`ChatRequest`] 转换为 Chat Completions 请求体
//! - 解析 Chat Completions 响应（含 tool calls、usage）
//! - 通过 `stream` 子模块支持 SSE 流式输出
//! - 将 `embed`、`audio`、`image` 等能力委托给对应工具模块
//!
//! 流式状态机实现见 [`stream`] 子模块。

mod stream;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{
    error::LlmError,
    types::{
        content::{ContentPart, Message, Role, ToolCall, ToolChoice, ToolResultPayload},
        request::ChatRequest,
        response::{
            ChatChunk, ChatResponse, EmbeddingResponse, ImageGenResponse, SpeechToTextResponse,
            TextToSpeechResponse, VideoGenResponse,
        },
    },
};

use super::{
    CHAT_COMPLETIONS_PATH, audio, embedding,
    http::{
        HttpContext, OpenAiRequestBuilder, parse_finish_reason, parse_usage, send_json_request,
    },
    image,
    media::resolve_media_source_to_image_url,
    unsupported_multimodal_error,
};
use crate::provider::ProviderAdapter;

// ── Adapter struct ────────────────────────────────────────────────────────────

/// Chat Completions 协议 adapter。
///
/// 实现全部 [`ProviderAdapter`] 方法：聊天、流式聊天、embedding、语音、图片生成。
pub(super) struct ChatCompletionsAdapter {
    pub(super) http_context: HttpContext,
}

impl ChatCompletionsAdapter {
    /// 使用共享 HTTP 上下文构造 adapter。
    pub(super) fn new(http_context: HttpContext) -> Self {
        Self { http_context }
    }
}

impl OpenAiRequestBuilder for ChatCompletionsAdapter {
    fn http_context(&self) -> &HttpContext {
        &self.http_context
    }
}

// ── 消息转换 ──────────────────────────────────────────────────────────────────

impl ChatCompletionsAdapter {
    /// 将 `tool` role 消息转换为一组 `{ role: "tool", tool_call_id, content }` 对象。
    fn to_tool_role_messages(&self, message: &Message) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out = Vec::with_capacity(message.content.len());
        for part in &message.content {
            let ContentPart::ToolResult(result) = part else {
                return Err(LlmError::ToolProtocol {
                    message: "tool role 仅允许 ToolResult".into(),
                });
            };

            let content = match &result.payload {
                ToolResultPayload::Text(text) => text.clone(),
                ToolResultPayload::Json(value) => value.to_string(),
            };
            out.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": content,
            }));
        }
        Ok(out)
    }

    /// 将 `assistant` role 消息转换为包含 `tool_calls` 数组（可选）的对象。
    fn to_assistant_chat_message(&self, message: &Message) -> Result<serde_json::Value, LlmError> {
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for part in &message.content {
            match part {
                ContentPart::Text(value) => text.push_str(&value.text),
                ContentPart::ToolCall(call) => {
                    tool_calls.push(serde_json::json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.arguments.to_string(),
                        }
                    }));
                }
                _ => {
                    return Err(unsupported_multimodal_error(
                        &self.http_context,
                        message.role,
                    ));
                }
            }
        }

        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), "assistant".into());
        obj.insert("content".into(), text.into());
        if let Some(name) = &message.name {
            obj.insert("name".into(), name.clone().into());
        }
        if !tool_calls.is_empty() {
            obj.insert("tool_calls".into(), tool_calls.into());
        }
        Ok(serde_json::Value::Object(obj))
    }

    /// 将 `user` / `system` role 消息转换为多模态 content 数组对象。
    async fn to_user_or_system_chat_message(
        &self,
        message: &Message,
    ) -> Result<serde_json::Value, LlmError> {
        let mut parts = Vec::with_capacity(message.content.len());
        for part in &message.content {
            match part {
                ContentPart::Text(value) => parts.push(serde_json::json!({
                    "type": "text",
                    "text": value.text,
                })),
                ContentPart::Image(image) => {
                    let image_url = resolve_media_source_to_image_url(&image.source).await?;
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": image_url,
                    }));
                }
                _ => {
                    return Err(unsupported_multimodal_error(
                        &self.http_context,
                        message.role,
                    ));
                }
            }
        }

        let role = match message.role {
            Role::User => "user",
            Role::System => "system",
            _ => unreachable!(),
        };

        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), role.into());
        obj.insert("content".into(), parts.into());
        if let Some(name) = &message.name {
            obj.insert("name".into(), name.clone().into());
        }
        Ok(serde_json::Value::Object(obj))
    }

    /// 将 [`Message`] 列表转换为 Chat Completions `messages` 数组。
    pub(super) async fn to_chat_messages(
        &self,
        messages: &[Message],
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out = Vec::with_capacity(messages.len());
        for message in messages {
            match message.role {
                Role::Tool => out.extend(self.to_tool_role_messages(message)?),
                Role::Assistant => out.push(self.to_assistant_chat_message(message)?),
                Role::User | Role::System => {
                    out.push(self.to_user_or_system_chat_message(message).await?)
                }
            }
        }
        Ok(out)
    }

    /// 将 [`ToolChoice`] 转换为 Chat Completions `tool_choice` 值。
    pub(super) fn build_tool_choice_payload(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto => "auto".into(),
            ToolChoice::None => "none".into(),
            ToolChoice::Required => "required".into(),
            ToolChoice::Specific(name) => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }

    /// 构造完整的 Chat Completions 请求体。
    ///
    /// `stream` 参数控制是否开启 SSE 流式输出。
    /// `req.extensions` 中的字段最后写入，可覆盖结构化字段（用于特殊场景调试）。
    pub(super) async fn to_request_body(
        &self,
        model: &str,
        req: &ChatRequest,
        stream: bool,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), model.to_owned().into());
        body.insert(
            "messages".into(),
            self.to_chat_messages(&req.messages).await?.into(),
        );
        body.insert("stream".into(), stream.into());

        if let Some(max_tokens) = req.max_tokens {
            body.insert("max_tokens".into(), max_tokens.into());
        }
        if let Some(temperature) = req.temperature {
            body.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = req.top_p {
            body.insert("top_p".into(), top_p.into());
        }
        if req.thinking || req.thinking_budget.is_some() {
            body.insert("thinking".into(), true.into());
        }
        if let Some(thinking_budget) = req.thinking_budget {
            body.insert("thinking_budget".into(), thinking_budget.into());
        }
        if let Some(reasoning_effort) = req.reasoning_effort {
            body.insert("reasoning_effort".into(), reasoning_effort.as_str().into());
        }
        if !req.tools.is_empty() {
            body.insert(
                "tools".into(),
                req.tools
                    .iter()
                    .map(|tool| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            }
                        })
                    })
                    .collect::<Vec<_>>()
                    .into(),
            );
            body.insert(
                "tool_choice".into(),
                Self::build_tool_choice_payload(&req.tool_choice),
            );
            if let Some(parallel_tool_calls) = req.parallel_tool_calls {
                body.insert("parallel_tool_calls".into(), parallel_tool_calls.into());
            }
        }

        // extensions 最后写入，允许覆盖结构化字段
        for (key, value) in &req.extensions {
            body.insert(key.clone(), value.clone());
        }

        Ok(serde_json::Value::Object(body))
    }

    /// 解析 Chat Completions 非流式响应。
    pub(super) fn parse_response(
        &self,
        raw: serde_json::Value,
        raw_field: Option<serde_json::Value>,
    ) -> Result<ChatResponse, LlmError> {
        let choice = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .ok_or_else(|| LlmError::ProviderResponse {
                provider: self.http_context.provider_name().into(),
                code: None,
                message: "缺少 choices[0]".into(),
            })?;

        let message = choice
            .get("message")
            .and_then(|value| value.as_object())
            .ok_or_else(|| LlmError::ProviderResponse {
                provider: self.http_context.provider_name().into(),
                code: None,
                message: "缺少 message".into(),
            })?;
        let text = match message.get("content") {
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(serde_json::Value::Null) | None => String::new(),
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .map(|item| {
                    item.get("text")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned)
                        .ok_or_else(|| LlmError::ProviderResponse {
                            provider: self.http_context.provider_name().into(),
                            code: None,
                            message: "message.content 数组项缺少 text".into(),
                        })
                })
                .collect::<Result<Vec<_>, LlmError>>()?
                .join(""),
            Some(_) => {
                return Err(LlmError::ProviderResponse {
                    provider: self.http_context.provider_name().into(),
                    code: None,
                    message: "message.content 形态不受支持".into(),
                });
            }
        };

        let tool_calls = message
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        let id =
                            item.get("id")
                                .and_then(|value| value.as_str())
                                .ok_or_else(|| LlmError::ToolProtocol {
                                    message: "tool call 缺少 id".into(),
                                })?;
                        let function = item
                            .get("function")
                            .and_then(|value| value.as_object())
                            .ok_or_else(|| LlmError::ToolProtocol {
                                message: "tool call 缺少 function".into(),
                            })?;
                        let tool_name = function
                            .get("name")
                            .and_then(|value| value.as_str())
                            .ok_or_else(|| LlmError::ToolProtocol {
                                message: "tool call 缺少 name".into(),
                            })?;
                        let arguments_raw = match function.get("arguments") {
                            Some(serde_json::Value::String(arguments)) => arguments,
                            Some(_) => {
                                return Err(LlmError::ToolProtocol {
                                    message: "tool call arguments 不是字符串".into(),
                                });
                            }
                            None => {
                                return Err(LlmError::ToolProtocol {
                                    message: "tool call 缺少 arguments".into(),
                                });
                            }
                        };
                        let arguments = serde_json::from_str(arguments_raw).map_err(|err| {
                            LlmError::ToolProtocol {
                                message: format!("tool arguments 解析失败: {err}"),
                            }
                        })?;

                        Ok(ToolCall {
                            id: id.to_owned(),
                            tool_name: tool_name.to_owned(),
                            arguments,
                        })
                    })
                    .collect::<Result<Vec<_>, LlmError>>()
            })
            .transpose()?
            .unwrap_or_default();

        // 兼容两种字段命名：DeepSeek/Qwen 使用 `reasoning_content`，
        // vLLM >= 0.9 使用 `reasoning`，优先取前者。
        let thinking = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(|value| value.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        Ok(ChatResponse {
            id: raw
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned(),
            model: raw
                .get("model")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned(),
            text,
            thinking,
            tool_calls,
            finish_reason: parse_finish_reason(
                choice.get("finish_reason").and_then(|value| value.as_str()),
            ),
            usage: parse_usage(raw.get("usage")),
            raw: raw_field,
        })
    }

    /// 执行非流式聊天请求，返回完整 [`ChatResponse`]。
    pub(super) async fn execute_chat(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        let request_started_at = std::time::Instant::now();
        let message_count = req.messages.len();
        let tool_count = req.tools.len();
        let body = self.to_request_body(model, &req, false).await?;
        let raw =
            send_json_request(self, self.post_json(CHAT_COMPLETIONS_PATH).json(&body)).await?;
        let parsed = self.parse_response(raw.clone(), Some(raw))?;
        tracing::debug!(
            provider = self.http_context.provider_name(),
            model,
            message_count,
            tool_count,
            finish_reason = ?parsed.finish_reason,
            elapsed_ms = request_started_at.elapsed().as_millis() as u64,
            "chat 完成"
        );
        Ok(parsed)
    }
}

// ── ProviderAdapter 实现 ──────────────────────────────────────────────────────

#[async_trait]
impl ProviderAdapter for ChatCompletionsAdapter {
    fn name(&self) -> &'static str {
        self.http_context.provider_name()
    }

    async fn chat(&self, model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.execute_chat(model, req).await
    }

    async fn chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError> {
        self.execute_chat_stream(model, req).await
    }

    async fn embed(
        &self,
        model: &str,
        req: crate::types::request::EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        embedding::execute_embed(self, model, req).await
    }

    async fn speech_to_text(
        &self,
        model: &str,
        req: crate::types::request::SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, LlmError> {
        audio::execute_speech_to_text(self, model, req).await
    }

    async fn text_to_speech(
        &self,
        model: &str,
        req: crate::types::request::TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, LlmError> {
        audio::execute_text_to_speech(self, model, req).await
    }

    async fn generate_image(
        &self,
        model: &str,
        req: crate::types::request::ImageGenRequest,
    ) -> Result<ImageGenResponse, LlmError> {
        image::execute_generate_image(self, model, req).await
    }

    async fn generate_video(
        &self,
        model: &str,
        req: crate::types::request::VideoGenRequest,
    ) -> Result<VideoGenResponse, LlmError> {
        image::execute_generate_video(self, model, req).await
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        middleware::{RetryConfig, Transport, TransportConfig},
        types::{
            content::{ContentPart, Message, Tool},
            request::ReasoningEffort,
        },
    };

    use super::stream::PartialToolCall;
    use super::*;
    use crate::types::response::FinishReason;

    fn test_adapter() -> ChatCompletionsAdapter {
        ChatCompletionsAdapter::new(HttpContext::new(
            "compatible",
            "test-key",
            "https://example.com",
            Transport::new(TransportConfig {
                timeout_secs: 600,
                connect_timeout_secs: 5,
                read_timeout_secs: 60,
                retry: RetryConfig::default(),
                rate_limit: None,
            }),
        ))
    }

    #[tokio::test]
    async fn request_body_preserves_extensions_and_message_name() {
        let adapter = test_adapter();
        let req = ChatRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentPart::text("杭州天气")],
                name: Some("alice".into()),
            }],
            extensions: serde_json::Map::from_iter([(
                "reasoning".into(),
                serde_json::json!({ "effort": "medium" }),
            )]),
            ..ChatRequest::default()
        };

        let body = adapter
            .to_request_body("gpt-4o-mini", &req, false)
            .await
            .unwrap();

        assert_eq!(
            body.get("reasoning"),
            Some(&serde_json::json!({ "effort": "medium" }))
        );
        assert_eq!(body["messages"][0]["name"], "alice");
    }

    #[tokio::test]
    async fn request_body_includes_structured_reasoning_and_tool_controls() {
        let adapter = test_adapter();
        let req = ChatRequest::builder()
            .user_text("杭州天气")
            .thinking_budget(2048)
            .reasoning_effort(ReasoningEffort::Medium)
            .tools(vec![Tool::function(
                "get_weather",
                "查询天气",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"],
                }),
            )])
            .parallel_tool_calls(false)
            .build();

        let body = adapter
            .to_request_body("gpt-4o-mini", &req, false)
            .await
            .unwrap();

        assert_eq!(body.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(body.get("thinking_budget"), Some(&serde_json::json!(2048)));
        assert_eq!(
            body.get("reasoning_effort"),
            Some(&serde_json::json!("medium"))
        );
        assert_eq!(
            body.get("parallel_tool_calls"),
            Some(&serde_json::json!(false))
        );
    }

    #[tokio::test]
    async fn extensions_override_structured_request_controls() {
        let adapter = test_adapter();
        let req = ChatRequest::builder()
            .user_text("杭州天气")
            .thinking(true)
            .reasoning_effort(ReasoningEffort::Low)
            .parallel_tool_calls(true)
            .tools(vec![Tool::function(
                "get_weather",
                "查询天气",
                serde_json::json!({ "type": "object" }),
            )])
            .extension("thinking", serde_json::json!(false))
            .extension("reasoning_effort", serde_json::json!("high"))
            .extension("parallel_tool_calls", serde_json::json!(false))
            .build();

        let body = adapter
            .to_request_body("gpt-4o-mini", &req, false)
            .await
            .unwrap();

        assert_eq!(body.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(
            body.get("reasoning_effort"),
            Some(&serde_json::json!("high"))
        );
        assert_eq!(
            body.get("parallel_tool_calls"),
            Some(&serde_json::json!(false))
        );
    }

    #[test]
    fn parse_response_rejects_missing_tool_arguments() {
        let adapter = test_adapter();
        let error = adapter
            .parse_response(
                serde_json::json!({
                    "id": "chatcmpl_123",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }),
                None,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ToolProtocol { ref message } if message == "tool call 缺少 arguments"
        ));
    }

    #[test]
    fn parse_response_rejects_missing_message() {
        let adapter = test_adapter();
        let error = adapter
            .parse_response(
                serde_json::json!({
                    "id": "chatcmpl_123",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "finish_reason": "completed"
                    }]
                }),
                None,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ProviderResponse { ref message, .. } if message == "缺少 message"
        ));
    }

    #[test]
    fn parse_response_rejects_non_text_content_array_items() {
        let adapter = test_adapter();
        let error = adapter
            .parse_response(
                serde_json::json!({
                    "id": "chatcmpl_123",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "message": {
                            "content": [{
                                "type": "image_url",
                                "image_url": { "url": "https://example.com/image.png" }
                            }]
                        },
                        "finish_reason": "completed"
                    }]
                }),
                None,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ProviderResponse { ref message, .. }
                if message == "message.content 数组项缺少 text"
        ));
    }

    #[test]
    fn partial_tool_call_rejects_missing_arguments() {
        let error = PartialToolCall {
            id: Some("call_1".into()),
            tool_name: Some("get_weather".into()),
            arguments: String::new(),
            arguments_seen: false,
        }
        .finalize()
        .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ToolProtocol { ref message } if message == "stream tool call 缺少 arguments"
        ));
    }

    #[test]
    fn parse_stream_event_rejects_non_string_arguments() {
        let error = ChatCompletionsAdapter::parse_stream_event(
            "compatible",
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": { "city": "Hangzhou" }
                            }
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            &mut BTreeMap::new(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ToolProtocol { ref message }
                if message == "stream tool call arguments 不是字符串"
        ));
    }

    #[test]
    fn parse_stream_event_finishes_with_missing_arguments_error() {
        let error = ChatCompletionsAdapter::parse_stream_event(
            "compatible",
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "get_weather"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }),
            &mut BTreeMap::new(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LlmError::ToolProtocol { ref message }
                if message == "stream tool call 缺少 arguments"
        ));
    }

    #[test]
    fn parse_stream_event_keeps_finish_reason_for_valid_tool_call() {
        let chunks = ChatCompletionsAdapter::parse_stream_event(
            "compatible",
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Hangzhou\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }),
            &mut BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(matches!(
            chunks[0].finish_reason,
            Some(FinishReason::ToolCalls)
        ));
        assert_eq!(chunks[0].tool_calls.len(), 1);
    }

    #[test]
    fn parse_response_extracts_reasoning_content_field() {
        let adapter = test_adapter();
        let resp = adapter
            .parse_response(
                serde_json::json!({
                    "id": "chatcmpl_1",
                    "model": "deepseek-reasoner",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "先分析一下问题。",
                            "content": "答案是 42。"
                        },
                        "finish_reason": "completed"
                    }]
                }),
                None,
            )
            .unwrap();

        assert_eq!(resp.thinking.as_deref(), Some("先分析一下问题。"));
        assert_eq!(resp.text, "答案是 42。");
    }

    #[test]
    fn parse_response_extracts_reasoning_field_as_fallback() {
        let adapter = test_adapter();
        let resp = adapter
            .parse_response(
                serde_json::json!({
                    "id": "chatcmpl_2",
                    "model": "local-model",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "reasoning": "思考过程在此。",
                            "content": "最终回答。"
                        },
                        "finish_reason": "completed"
                    }]
                }),
                None,
            )
            .unwrap();

        assert_eq!(resp.thinking.as_deref(), Some("思考过程在此。"));
        assert_eq!(resp.text, "最终回答。");
    }

    #[test]
    fn parse_stream_event_emits_thinking_delta_before_text() {
        let chunks = ChatCompletionsAdapter::parse_stream_event(
            "compatible",
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "reasoning_content": "正在思考…",
                        "content": ""
                    },
                    "finish_reason": null
                }]
            }),
            &mut BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].thinking_delta.as_deref(), Some("正在思考…"));
        assert!(chunks[0].text_delta.is_none());
    }

    #[test]
    fn parse_stream_event_emits_thinking_delta_via_reasoning_field() {
        let chunks = ChatCompletionsAdapter::parse_stream_event(
            "compatible",
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "reasoning": "vLLM 新字段。",
                        "content": null
                    },
                    "finish_reason": null
                }]
            }),
            &mut BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].thinking_delta.as_deref(), Some("vLLM 新字段。"));
    }
}
