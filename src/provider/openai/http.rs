//! OpenAI 系 adapter 共用的 HTTP 工具层。
//!
//! 本模块提供三类能力：
//!
//! 1. **[`OpenAiRequestBuilder`] trait** — `ChatCompletionsAdapter` 和 `ResponsesAdapter`
//!    都实现此 trait，使 `audio`、`embedding`、`image` 等模块可以泛型地复用请求构造逻辑。
//!
//! 2. **错误 / 响应解析工具** — HTTP 错误映射、`finish_reason` 与 `usage` 的
//!    统一解析，两套协议共用同一套实现。
//!
//! 3. **SSE 流式工具** — 字节缓冲分割 (`take_sse_event`) 与数据行提取
//!    (`parse_sse_data`)，供两套流式状态机调用。

use crate::{
    error::LlmError,
    middleware::Transport,
    types::response::{FinishReason, Usage},
};

// ── OpenAiRequestBuilder trait ───────────────────────────────────────────────

/// OpenAI 系 adapter 共用的 HTTP 上下文。
///
/// 把 provider 名称、鉴权信息和标准化后的 base URL 收拢到一起，避免不同协议
/// adapter 各自维护同一份样板字段。
pub(super) struct HttpContext {
    transport: Transport,
    api_key: String,
    base_url: String,
    provider_name: &'static str,
}

impl HttpContext {
    /// 构造标准化后的 HTTP 上下文。
    pub(super) fn new(
        provider_name: &'static str,
        api_key: &str,
        base_url: &str,
        transport: Transport,
    ) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.trim_end_matches('/').to_owned(),
            provider_name,
        }
    }

    /// 返回底层传输配置。
    pub(super) fn transport(&self) -> &Transport {
        &self.transport
    }

    /// 返回鉴权使用的 API Key。
    pub(super) fn api_key(&self) -> &str {
        &self.api_key
    }

    /// 返回标准化后的 base URL。
    pub(super) fn base_url(&self) -> &str {
        &self.base_url
    }

    /// 返回 provider 名称。
    pub(super) fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

/// 两套 adapter 共用的 HTTP 请求构造能力。
///
/// `audio`、`embedding`、`image` 模块通过泛型参数 `A: OpenAiRequestBuilder` 调用请求方法，
/// 无需为每套 adapter 各写一份实现。
pub(super) trait OpenAiRequestBuilder {
    /// 返回 adapter 持有的共享 HTTP 上下文。
    fn http_context(&self) -> &HttpContext;

    fn transport(&self) -> &Transport {
        self.http_context().transport()
    }

    fn api_key(&self) -> &str {
        self.http_context().api_key()
    }

    fn base_url(&self) -> &str {
        self.http_context().base_url()
    }

    fn provider_name(&self) -> &'static str {
        self.http_context().provider_name()
    }

    /// 构造带有 `Authorization` 和 `Content-Type: application/json` 的 POST 请求。
    fn post_json(&self, path: &str) -> reqwest::RequestBuilder {
        self.transport()
            .client()
            .post(format!(
                "{}/{}",
                self.base_url(),
                path.trim_start_matches('/')
            ))
            .bearer_auth(self.api_key())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
    }

    /// 构造带有 `Authorization` 的 GET 请求。
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.transport()
            .client()
            .get(format!(
                "{}/{}",
                self.base_url(),
                path.trim_start_matches('/')
            ))
            .bearer_auth(self.api_key())
    }

    /// 构造带有 `Authorization` 的 POST 请求（用于 multipart/form-data）。
    fn post_multipart(&self, path: &str) -> reqwest::RequestBuilder {
        self.transport()
            .client()
            .post(format!(
                "{}/{}",
                self.base_url(),
                path.trim_start_matches('/')
            ))
            .bearer_auth(self.api_key())
    }
}

// ── HTTP 错误映射 ─────────────────────────────────────────────────────────────

/// 将非 2xx HTTP 响应映射为 [`LlmError`]。
///
/// 401/403 → 认证错误；429 → 限速；其余 → [`LlmError::HttpStatus`]。
pub(super) fn map_error_response(provider_name: &str, status: u16, body_text: &str) -> LlmError {
    let mut parsed_message = None;
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_text) {
        if let Some(msg) = json.pointer("/error/message").and_then(|v| v.as_str()) {
            parsed_message = Some(msg.to_owned());
        } else if let Some(msg) = json.get("error").and_then(|v| v.as_str()) {
            parsed_message = Some(msg.to_owned());
        }
    }
    let display_body = parsed_message.unwrap_or_else(|| body_text.to_owned());

    match status {
        401 | 403 => LlmError::Authentication {
            message: format!("[{provider_name}] {display_body}"),
        },
        429 => LlmError::RateLimit {
            retry_after_secs: None,
        },
        _ => LlmError::HttpStatus {
            provider: provider_name.into(),
            status,
            body: display_body,
        },
    }
}

/// 发送请求并在非 2xx 状态时映射为 [`LlmError`]。
pub(super) async fn send_request<A: OpenAiRequestBuilder>(
    adapter: &A,
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, LlmError> {
    let response = adapter.transport().send(request).await?;
    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status().as_u16();
        let body_text = response.text().await?;
        Err(map_error_response(adapter.provider_name(), status, &body_text))
    }
}

/// 发送 JSON 请求并解析 JSON 响应。
pub(super) async fn send_json_request<A: OpenAiRequestBuilder>(
    adapter: &A,
    request: reqwest::RequestBuilder,
) -> Result<serde_json::Value, LlmError> {
    send_request(adapter, request)
        .await?
        .json()
        .await
        .map_err(Into::into)
}

/// 发送请求并读取原始字节响应。
pub(super) async fn send_bytes_request<A: OpenAiRequestBuilder>(
    adapter: &A,
    request: reqwest::RequestBuilder,
    action: &'static str,
) -> Result<bytes::Bytes, LlmError> {
    send_request(adapter, request)
        .await?
        .bytes()
        .await
        .map_err(|err| LlmError::transport(action, err))
}

// ── 响应字段解析 ──────────────────────────────────────────────────────────────

/// 将 `finish_reason` 字符串映射为 [`FinishReason`]。
pub(super) fn parse_finish_reason(raw: Option<&str>) -> Option<FinishReason> {
    match raw {
        Some("stop") | Some("completed") => Some(FinishReason::Completed),
        Some("length") | Some("max_tokens") | Some("max_output_tokens") => Some(FinishReason::MaxOutputTokens),
        Some("tool_calls") => Some(FinishReason::ToolCalls),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some(_) => Some(FinishReason::Failed),
        None => None,
    }
}

/// 从 `usage` JSON 对象中解析 token 用量。
///
/// 兼容两套字段命名：
/// - Chat Completions: `prompt_tokens` / `completion_tokens` / `total_tokens`
/// - Responses API: `input_tokens` / `output_tokens` / `total_tokens`
pub(super) fn parse_usage(raw: Option<&serde_json::Value>) -> Option<Usage> {
    let raw = raw?;
    let parse_tokens = |value: Option<&serde_json::Value>| -> Option<u32> {
        value?.as_u64()?.try_into().ok()
    };

    Some(Usage {
        prompt_tokens: parse_tokens(
            raw.get("prompt_tokens").or_else(|| raw.get("input_tokens")),
        )?,
        completion_tokens: parse_tokens(
            raw.get("completion_tokens")
                .or_else(|| raw.get("output_tokens")),
        )?,
        total_tokens: parse_tokens(raw.get("total_tokens"))?,
    })
}

// ── 流式超时错误 ──────────────────────────────────────────────────────────────

/// 构造"读取流式响应超时"错误，附带可操作的提示信息。
pub(super) fn stream_read_timeout_error(read_timeout_ms: u64) -> LlmError {
    LlmError::request_timeout(
        "读取流式响应",
        read_timeout_ms,
        "流式连接已建立，但在读取超时窗口内未收到新数据；可尝试增大 read_timeout_secs，或检查 provider 是否持续输出分片",
    )
}

/// 将流读取错误映射为 [`LlmError`]；超时错误使用专用消息，其余走通用 transport 错误。
pub(super) fn map_stream_read_error(read_timeout_ms: u64, err: reqwest::Error) -> LlmError {
    if err.is_timeout() {
        stream_read_timeout_error(read_timeout_ms)
    } else {
        LlmError::transport("读取流式响应", err)
    }
}

// ── SSE 帧解析 ────────────────────────────────────────────────────────────────

/// 从字节缓冲中取出一个完整 SSE 帧（以 `\r\n\r\n` 或 `\n\n` 为分隔符）。
///
/// 找到分隔符时，将帧内容（含分隔符）从缓冲头部 drain 出来并返回；
/// 缓冲中不足一帧时返回 `None`。
pub(super) fn take_sse_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let mut split_at = None;
    for (i, window) in buffer.windows(2).enumerate() {
        if window == b"\n\n" {
            split_at = Some((i, 2));
            break;
        } else if window == b"\r\n" && buffer.get(i..i + 4) == Some(b"\r\n\r\n") {
            split_at = Some((i, 4));
            break;
        }
    }

    if let Some((idx, len)) = split_at {
        Some(buffer.drain(..idx + len).collect())
    } else {
        None
    }
}

/// 从一个 SSE 帧字节序列中提取所有 `data:` 行的内容，拼接为字符串。
///
/// 仅包含 `event:`、`id:`、`retry:` 等非 data 行的帧返回 `Ok(None)`。
pub(super) fn parse_sse_data(
    event: &[u8],
    provider_name: &str,
) -> Result<Option<String>, LlmError> {
    let raw = String::from_utf8(event.to_vec()).map_err(|err| LlmError::StreamProtocol {
        provider: provider_name.to_owned(),
        message: format!("SSE 数据不是合法 UTF-8: {err}"),
    })?;
    let mut data_lines = Vec::new();
    for line in raw.replace("\r\n", "\n").lines() {
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_owned());
        }
    }
    if data_lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(data_lines.join("\n")))
}

// ── SSE 状态机接口 ────────────────────────────────────────────────────────────

/// SSE 流式状态机的公共接口，供两套 adapter 共用。
///
/// 通过此 trait 与具体状态机交互，消除 `chat_completions::stream`
/// 与 `responses::stream` 中重复的事件分发循环。
pub(super) trait SseState {
    fn buffer_mut(&mut self) -> &mut Vec<u8>;
    fn is_done(&self) -> bool;
    fn abort(&mut self, err: LlmError);
    fn handle_data(&mut self, provider_name: &str, data: &str) -> Result<(), LlmError>;
}

/// 扫描缓冲区中已完整到达的 SSE 事件，逐帧分发给 `state.handle_data`。
pub(super) fn process_buffered_events<S: SseState>(provider_name: &str, state: &mut S) {
    while let Some(event) = take_sse_event(state.buffer_mut()) {
        let data = match parse_sse_data(&event, provider_name) {
            Ok(Some(data)) => data,
            Ok(None) => continue,
            Err(err) => { state.abort(err); break; }
        };
        if let Err(err) = state.handle_data(provider_name, &data) {
            state.abort(err);
            break;
        }
        if state.is_done() { break; }
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_usage_accepts_u32_range_values() {
        let usage = parse_usage(Some(&serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        })))
        .unwrap();

        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn parse_usage_rejects_values_larger_than_u32() {
        assert!(
            parse_usage(Some(&serde_json::json!({
                "prompt_tokens": u64::from(u32::MAX) + 1,
                "completion_tokens": 20,
                "total_tokens": 30,
            })))
            .is_none()
        );
    }

    #[test]
    fn take_sse_event_splits_on_double_newline() {
        let mut buf = b"data: hello\n\ndata: world\n\n".to_vec();
        let event = take_sse_event(&mut buf).unwrap();
        assert_eq!(event, b"data: hello\n\n");
        assert_eq!(buf, b"data: world\n\n");
    }

    #[test]
    fn parse_sse_data_extracts_data_lines() {
        let event = b"data: {\"id\":1}\n\n";
        let result = parse_sse_data(event, "test").unwrap();
        assert_eq!(result, Some("{\"id\":1}".to_owned()));
    }

    #[test]
    fn parse_sse_data_returns_none_for_non_data_event() {
        let event = b"event: ping\n\n";
        let result = parse_sse_data(event, "test").unwrap();
        assert!(result.is_none());
    }
}
