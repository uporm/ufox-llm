//! Responses 协议 adapter。
//!
//! 实现 [`ProviderAdapter`] trait，使用 OpenAI Responses API
//! (`POST /responses`) 处理对话请求。
//!
//! 与 Chat Completions 协议的主要差异：
//! - 请求体使用 `input` 数组（而非 `messages`），消息格式更结构化
//! - 支持 `reasoning`（思考链）字段，可携带摘要输出
//! - 响应的 `output` 数组可同时包含文本、tool call、推理内容
//! - `store: false` — 默认不在 OpenAI 服务端存储会话
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
            ChatChunk, ChatResponse, EmbeddingResponse, FinishReason, ImageGenResponse,
            SpeechToTextResponse, TextToSpeechResponse, VideoGenResponse,
        },
    },
};

use super::{
    RESPONSES_PATH, audio, embedding,
    http::{HttpContext, OpenAiRequestBuilder, parse_usage, send_json_request},
    image,
    media::resolve_media_source_to_image_url,
    normalize_chat_request,
    unsupported_multimodal_error,
};
use crate::provider::ProviderAdapter;

// ── Adapter struct ────────────────────────────────────────────────────────────

/// Responses 协议 adapter。
///
/// 实现全部 [`ProviderAdapter`] 方法：聊天、流式聊天、embedding、语音、图片生成。
pub(super) struct ResponsesAdapter {
    pub(super) http_context: HttpContext,
}

impl ResponsesAdapter {
    /// 使用共享 HTTP 上下文构造 adapter。
    pub(super) fn new(http_context: HttpContext) -> Self {
        Self { http_context }
    }
}

impl OpenAiRequestBuilder for ResponsesAdapter {
    fn http_context(&self) -> &HttpContext {
        &self.http_context
    }
}

// ── 消息转换 ──────────────────────────────────────────────────────────────────

impl ResponsesAdapter {
    /// 将 `user` / `system` role 消息转换为 Responses API `input_text` / `input_image` 内容项。
    async fn build_user_or_system_input_item(
        &self,
        message: &Message,
    ) -> Result<serde_json::Value, LlmError> {
        let mut parts = Vec::with_capacity(message.content.len());
        for part in &message.content {
            match part {
                ContentPart::Text(value) => parts.push(serde_json::json!({
                    "type": "input_text",
                    "text": value.text,
                })),
                ContentPart::Image(image) => {
                    let image_url = resolve_media_source_to_image_url(&image.source).await?;
                    parts.push(serde_json::json!({
                        "type": "input_image",
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

        Ok(serde_json::json!({
            "role": match message.role {
                Role::User => "user",
                Role::System => "system",
                _ => unreachable!(),
            },
            "content": parts,
        }))
    }

    /// 将 `tool` role 消息转换为 `function_call_output` 条目列表。
    fn build_tool_result_input_items(
        &self,
        message: &Message,
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out = Vec::with_capacity(message.content.len());
        for part in &message.content {
            let ContentPart::ToolResult(result) = part else {
                return Err(LlmError::ToolProtocol {
                    message: "tool role 仅允许 ToolResult".into(),
                });
            };

            let output = match &result.payload {
                ToolResultPayload::Text(text) => text.clone(),
                ToolResultPayload::Json(value) => value.to_string(),
            };
            out.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": result.tool_call_id,
                "output": output,
            }));
        }
        Ok(out)
    }

    /// 将 `assistant` role 消息转换为 Responses API 格式的输出条目。
    ///
    /// 文本内容包装为 `{ role: "assistant", content: [{ type: "output_text" }] }`；
    /// tool call 包装为独立的 `function_call` 条目。
    fn build_assistant_input_items(
        &self,
        message: &Message,
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out = Vec::new();
        let mut text = String::new();
        // 将已累积文本刷入 out 并清空缓冲
        let flush_text = |out: &mut Vec<serde_json::Value>, text: &mut String| {
            if text.is_empty() {
                return;
            }
            out.push(serde_json::json!({
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": std::mem::take(text),
                }],
            }));
        };

        for part in &message.content {
            match part {
                ContentPart::Text(value) => text.push_str(&value.text),
                ContentPart::ToolCall(call) => {
                    flush_text(&mut out, &mut text);
                    out.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.tool_name,
                        "arguments": call.arguments.to_string(),
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
        flush_text(&mut out, &mut text);
        Ok(out)
    }

    /// 将 [`Message`] 列表转换为 Responses API `input` 数组。
    async fn build_input_items(
        &self,
        messages: &[Message],
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out = Vec::with_capacity(messages.len());
        for message in messages {
            match message.role {
                Role::Tool => out.extend(self.build_tool_result_input_items(message)?),
                Role::Assistant => out.extend(self.build_assistant_input_items(message)?),
                Role::User | Role::System => {
                    out.push(self.build_user_or_system_input_item(message).await?)
                }
            }
        }
        Ok(out)
    }

    /// 将 [`ToolChoice`] 转换为 Responses API `tool_choice` 值。
    fn build_tool_choice_payload(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto => "auto".into(),
            ToolChoice::None => "none".into(),
            ToolChoice::Required => "required".into(),
            ToolChoice::Specific(name) => serde_json::json!({
                "type": "function",
                "name": name,
            }),
        }
    }

    /// 从 [`ChatRequest`] 构造 Responses API `reasoning` 对象。
    ///
    /// 仅在有 reasoning_effort 或 thinking 请求时返回 `Some`；否则不插入字段。
    fn build_reasoning_payload(req: &ChatRequest) -> Option<serde_json::Value> {
        let mut reasoning = serde_json::Map::new();
        if let Some(reasoning_effort) = req.reasoning_effort {
            reasoning.insert("effort".into(), reasoning_effort.as_str().into());
        }
        if req.thinking || req.thinking_budget.is_some() {
            reasoning.insert("summary".into(), "auto".into());
        }
        (!reasoning.is_empty()).then_some(serde_json::Value::Object(reasoning))
    }

    /// 构造完整的 Responses API 请求体。
    ///
    /// `stream` 参数控制是否开启 SSE 流式输出。
    /// `store: false` 确保不在 OpenAI 侧存储会话历史。
    /// `req.extensions` 最后写入，可覆盖结构化字段。
    pub(super) async fn build_request_body(
        &self,
        model: &str,
        req: &ChatRequest,
        stream: bool,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), model.to_owned().into());
        body.insert(
            "input".into(),
            self.build_input_items(&req.messages).await?.into(),
        );
        body.insert("stream".into(), stream.into());
        body.insert("store".into(), false.into());

        if let Some(max_tokens) = req.max_tokens {
            body.insert("max_output_tokens".into(), max_tokens.into());
        }
        if let Some(temperature) = req.temperature {
            body.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = req.top_p {
            body.insert("top_p".into(), top_p.into());
        }
        if let Some(reasoning) = Self::build_reasoning_payload(req) {
            body.insert("reasoning".into(), reasoning);
        }
        if !req.tools.is_empty() {
            body.insert(
                "tools".into(),
                req.tools
                    .iter()
                    .map(|tool| {
                        serde_json::json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
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
}

// ── 响应解析 ──────────────────────────────────────────────────────────────────

impl ResponsesAdapter {
    /// 从 `status` 字段推断 [`FinishReason`]，tool calls 优先。
    fn parse_finish_reason(
        raw: &serde_json::Value,
        tool_calls: &[ToolCall],
    ) -> Option<FinishReason> {
        if !tool_calls.is_empty() {
            return Some(FinishReason::ToolCalls);
        }

        match raw.get("status").and_then(|value| value.as_str()) {
            Some("completed") => Some(FinishReason::Completed),
            Some("incomplete") => match raw
                .get("incomplete_details")
                .and_then(|value| value.get("reason"))
                .and_then(|value| value.as_str())
            {
                Some("max_output_tokens") | Some("max_tokens") | Some("length") => Some(FinishReason::MaxOutputTokens),
                Some("content_filter") => Some(FinishReason::ContentFilter),
                Some(_) | None => Some(FinishReason::Failed),
            },
            Some("failed") | Some(_) => Some(FinishReason::Failed),
            None => None,
        }
    }

    /// 从 `content` 数组中提取所有 `output_text` / `text` 类型的文本。
    fn parse_response_text_parts(
        provider_name: &str,
        content: &[serde_json::Value],
    ) -> Result<Vec<String>, LlmError> {
        content
            .iter()
            .filter(|part| {
                matches!(
                    part.get("type").and_then(|value| value.as_str()),
                    Some("output_text") | Some("text")
                )
            })
            .map(|part| {
                part.get("text")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
                    .ok_or_else(|| LlmError::ProviderResponse {
                        provider: provider_name.to_owned(),
                        code: None,
                        message: "message.content 文本项缺少 text".into(),
                    })
            })
            .collect()
    }

    /// 从 `reasoning` 输出条目中提取思考链文本。
    ///
    /// 按优先级依次尝试 `summary[].text`、`content[].text`、`text`。
    fn parse_reasoning_text(item: &serde_json::Value) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(summary) = item.get("summary").and_then(|value| value.as_array()) {
            out.extend(summary.iter().filter_map(|part| {
                part.get("text")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            }));
        }
        if let Some(content) = item.get("content").and_then(|value| value.as_array()) {
            out.extend(content.iter().filter_map(|part| {
                part.get("text")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            }));
        }
        if out.is_empty()
            && let Some(text) = item.get("text").and_then(|value| value.as_str())
        {
            out.push(text.to_owned());
        }
        out
    }

    /// 从单个 `function_call` 输出条目中解析 [`ToolCall`]。
    fn parse_tool_call(
        provider_name: &str,
        item: &serde_json::Value,
    ) -> Result<ToolCall, LlmError> {
        let id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| LlmError::ToolProtocol {
                message: "response function_call 缺少 call_id".into(),
            })?;
        let tool_name = item
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| LlmError::ToolProtocol {
                message: "response function_call 缺少 name".into(),
            })?;
        let arguments_raw = item
            .get("arguments")
            .and_then(|value| value.as_str())
            .ok_or_else(|| LlmError::ProviderResponse {
                provider: provider_name.to_owned(),
                code: None,
                message: "response function_call.arguments 不是字符串".into(),
            })?;
        let arguments =
            serde_json::from_str(arguments_raw).map_err(|err| LlmError::ToolProtocol {
                message: format!("response tool arguments 解析失败: {err}"),
            })?;

        Ok(ToolCall {
            id: id.to_owned(),
            tool_name: tool_name.to_owned(),
            arguments,
        })
    }

    /// 解析完整的 Responses API 响应，提取文本、思考链、tool calls 及 usage。
    pub(super) fn parse_response_with_provider(
        provider_name: &str,
        raw: serde_json::Value,
        raw_field: Option<serde_json::Value>,
    ) -> Result<ChatResponse, LlmError> {
        let output = raw
            .get("output")
            .and_then(|value| value.as_array())
            .ok_or_else(|| LlmError::ProviderResponse {
                provider: provider_name.to_owned(),
                code: None,
                message: "缺少 output".into(),
            })?;

        let mut text = Vec::new();
        let mut thinking = Vec::new();
        let mut tool_calls = Vec::new();
        for item in output {
            match item.get("type").and_then(|value| value.as_str()) {
                Some("message") => {
                    let content = item
                        .get("content")
                        .and_then(|value| value.as_array())
                        .ok_or_else(|| LlmError::ProviderResponse {
                            provider: provider_name.to_owned(),
                            code: None,
                            message: "response message 缺少 content".into(),
                        })?;
                    text.extend(Self::parse_response_text_parts(provider_name, content)?);
                }
                Some("function_call") => {
                    tool_calls.push(Self::parse_tool_call(provider_name, item)?);
                }
                Some("reasoning") => {
                    thinking.extend(Self::parse_reasoning_text(item));
                }
                _ => {}
            }
        }

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
            text: text.join(""),
            thinking: (!thinking.is_empty()).then(|| thinking.join("")),
            finish_reason: Self::parse_finish_reason(&raw, &tool_calls),
            tool_calls,
            usage: parse_usage(raw.get("usage")),
            raw: raw_field,
        })
    }

    /// 解析 SSE 流式事件，返回零个或多个 [`ChatChunk`]。
    ///
    /// 事件类型处理：
    /// - `response.output_text.delta` → `text_delta` chunk
    /// - `response.reasoning_text.delta` / `response.reasoning_summary_text.delta`
    ///   / `response.reasoning_summary_part.added` → `thinking_delta` chunk
    /// - `response.completed` / `response.incomplete` → 提取 tool calls / finish_reason / usage，
    ///   仅在未通过 delta 发送的情况下补充文本 / 思考链内容
    /// - `error` → 返回 [`LlmError::ProviderResponse`]
    /// - 其余事件 → 忽略，返回空列表
    pub(super) fn parse_stream_event(
        provider_name: &str,
        raw: &serde_json::Value,
        saw_text_delta: &mut bool,
        saw_thinking_delta: &mut bool,
    ) -> Result<Vec<ChatChunk>, LlmError> {
        let event_type = raw
            .get("type")
            .and_then(|value| value.as_str())
            .ok_or_else(|| LlmError::StreamProtocol {
                provider: provider_name.to_owned(),
                message: "缺少 event.type".into(),
            })?;

        match event_type {
            "response.output_text.delta" => {
                let delta = raw
                    .get("delta")
                    .and_then(|value| value.as_str())
                    .filter(|delta| !delta.is_empty());
                if let Some(delta) = delta {
                    *saw_text_delta = true;
                    Ok(vec![ChatChunk {
                        text_delta: Some(delta.to_owned()),
                        ..ChatChunk::default()
                    }])
                } else {
                    Ok(Vec::new())
                }
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                let delta = raw
                    .get("delta")
                    .and_then(|value| value.as_str())
                    .filter(|delta| !delta.is_empty());
                if let Some(delta) = delta {
                    *saw_thinking_delta = true;
                    Ok(vec![ChatChunk {
                        thinking_delta: Some(delta.to_owned()),
                        ..ChatChunk::default()
                    }])
                } else {
                    Ok(Vec::new())
                }
            }
            "response.reasoning_summary_part.added" => {
                let text = raw
                    .get("part")
                    .and_then(|value| value.get("text"))
                    .and_then(|value| value.as_str())
                    .filter(|text| !text.is_empty());
                if let Some(text) = text {
                    *saw_thinking_delta = true;
                    Ok(vec![ChatChunk {
                        thinking_delta: Some(text.to_owned()),
                        ..ChatChunk::default()
                    }])
                } else {
                    Ok(Vec::new())
                }
            }
            "response.completed" | "response.incomplete" => {
                let response =
                    raw.get("response")
                        .cloned()
                        .ok_or_else(|| LlmError::StreamProtocol {
                            provider: provider_name.to_owned(),
                            message: format!("{event_type} 缺少 response"),
                        })?;
                let parsed = Self::parse_response_with_provider(
                    provider_name,
                    response.clone(),
                    Some(response),
                )?;
                let mut chunk = ChatChunk {
                    tool_calls: parsed.tool_calls,
                    finish_reason: parsed.finish_reason,
                    usage: parsed.usage,
                    ..ChatChunk::default()
                };
                // 若未通过 delta 事件发送过文本，则在此补充（非流式 provider 回退场景）
                if !*saw_text_delta && !parsed.text.is_empty() {
                    chunk.text_delta = Some(parsed.text);
                    *saw_text_delta = true;
                }
                if !*saw_thinking_delta && let Some(thinking) = parsed.thinking {
                    chunk.thinking_delta = Some(thinking);
                    *saw_thinking_delta = true;
                }
                Ok(vec![chunk])
            }
            "error" => Err(LlmError::ProviderResponse {
                provider: provider_name.to_owned(),
                code: raw
                    .get("error")
                    .and_then(|value| value.get("code"))
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                message: raw
                    .get("error")
                    .and_then(|value| value.get("message"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("stream 返回 error 事件")
                    .to_owned(),
            }),
            _ => Ok(Vec::new()),
        }
    }

    fn parse_response(
        &self,
        raw: serde_json::Value,
        raw_field: Option<serde_json::Value>,
    ) -> Result<ChatResponse, LlmError> {
        Self::parse_response_with_provider(self.http_context.provider_name(), raw, raw_field)
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
        let body = self.build_request_body(model, &req, false).await?;
        let raw = send_json_request(self, self.post_json(RESPONSES_PATH).json(&body)).await?;
        let parsed = self.parse_response(raw.clone(), Some(raw))?;
        tracing::debug!(
            provider = self.http_context.provider_name(),
            model,
            message_count,
            tool_count,
            finish_reason = ?parsed.finish_reason,
            elapsed_ms = request_started_at.elapsed().as_millis() as u64,
            "responses 完成"
        );
        Ok(parsed)
    }
}

// ── ProviderAdapter 实现 ──────────────────────────────────────────────────────

#[async_trait]
impl ProviderAdapter for ResponsesAdapter {
    fn name(&self) -> &'static str {
        self.http_context.provider_name()
    }

    async fn chat(&self, model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        let req = normalize_chat_request(self.http_context.provider_name(), req);
        self.execute_chat(model, req).await
    }

    async fn chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError> {
        let req = normalize_chat_request(self.http_context.provider_name(), req);
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
        super::video::execute_generate_video(self, model, req).await
    }

    async fn poll_video_task(&self, task_id: &str) -> Result<VideoGenResponse, LlmError> {
        super::video::execute_poll_video_task(self, task_id).await
    }

    async fn download_video_stream(
        &self,
        task_id: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<bytes::Bytes, LlmError>> + Send>>, LlmError> {
        super::video::execute_download_video_stream(self, task_id).await
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        middleware::{RetryConfig, Transport, TransportConfig},
        types::{
            content::{ContentPart, Message},
            request::ReasoningEffort,
        },
    };

    use super::*;

    fn test_adapter() -> ResponsesAdapter {
        ResponsesAdapter::new(HttpContext::new(
            "openai",
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
    async fn responses_request_body_uses_input_items_and_reasoning_summary() {
        let adapter = test_adapter();
        let req = ChatRequest::builder()
            .messages(vec![
                Message {
                    role: Role::System,
                    content: vec![ContentPart::text("你是助手")],
                    name: None,
                },
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentPart::text("先查天气"),
                        ContentPart::tool_call(
                            "call_1",
                            "get_weather",
                            serde_json::json!({ "city": "杭州" }),
                        ),
                    ],
                    name: None,
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentPart::tool_result("call_1", "{\"temp\":26}")],
                    name: None,
                },
            ])
            .thinking(true)
            .reasoning_effort(ReasoningEffort::Medium)
            .build();

        let body = adapter
            .build_request_body("gpt-5", &req, false)
            .await
            .unwrap();

        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(body["input"][1]["content"][0]["type"], "output_text");
        assert_eq!(body["input"][2]["type"], "function_call");
        assert_eq!(body["input"][3]["type"], "function_call_output");
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["store"], false);
    }

    #[test]
    fn parse_responses_response_extracts_text_thinking_and_tool_calls() {
        let response = ResponsesAdapter::parse_response_with_provider(
            "openai",
            serde_json::json!({
                "id": "resp_123",
                "model": "gpt-5",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            { "type": "summary_text", "text": "先确定地点。" }
                        ]
                    },
                    {
                        "type": "message",
                        "content": [
                            { "type": "output_text", "text": "杭州今天多云。" }
                        ]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Hangzhou\"}"
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 12,
                    "total_tokens": 22
                }
            }),
            None,
        )
        .unwrap();

        assert_eq!(response.text, "杭州今天多云。");
        assert_eq!(response.thinking.as_deref(), Some("先确定地点。"));
        assert_eq!(response.tool_calls.len(), 1);
        assert!(matches!(
            response.finish_reason,
            Some(FinishReason::ToolCalls)
        ));
        assert_eq!(response.usage.unwrap().total_tokens, 22);
    }

    #[test]
    fn parse_responses_stream_event_emits_completion_fallback_content() {
        let mut saw_text_delta = false;
        let mut saw_thinking_delta = false;
        let chunks = ResponsesAdapter::parse_stream_event(
            "openai",
            &serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "content": [{ "type": "output_text", "text": "你好" }]
                    }, {
                        "type": "reasoning",
                        "summary": [{ "type": "summary_text", "text": "先打招呼。" }]
                    }],
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 5,
                        "total_tokens": 8
                    }
                }
            }),
            &mut saw_text_delta,
            &mut saw_thinking_delta,
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text_delta.as_deref(), Some("你好"));
        assert_eq!(chunks[0].thinking_delta.as_deref(), Some("先打招呼。"));
        assert_eq!(chunks[0].usage.as_ref().unwrap().total_tokens, 8);
        assert!(matches!(chunks[0].finish_reason, Some(FinishReason::Completed)));
    }
}
