//! 客户端模块。
//!
//! 提供 `Client`、请求构建器和请求发送主流程。

use std::{pin::Pin, sync::Arc, time::Duration};

use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt, stream};
use reqwest::{Response, StatusCode, header::RETRY_AFTER};

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool, ToolChoice,
    provider::{compatible::CompatibleAdapter, openai::OpenAiAdapter, qwen::QwenAdapter},
    types::response::ReasoningEffort,
};

pub mod builder;
mod config;

use self::config::{ProviderConfig, ThinkingCapability};

pub use self::config::RequestOptions;
pub use builder::ClientBuilder;

/// 聊天流式响应类型。
///
/// 该类型是对异步流返回值的统一封装。流中的每一项都是一次增量输出或终止片段。
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>;

/// 聊天请求构建器。
///
/// 该类型用于组装一次完整的聊天请求。请求本身拥有消息与工具定义的所有权，因此不会把
/// 生命周期暴露到公开 API 中。
#[derive(Debug, Clone)]
pub struct ChatRequestBuilder {
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
    options: RequestOptions,
}

/// 可复用的聊天请求快照。
///
/// 构建完成后可以分别交给 [`Client::chat`] 与 [`Client::chat_stream`] 执行。
#[derive(Debug, Clone)]
pub struct ChatRequest {
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
    options: RequestOptions,
}

impl ChatRequest {
    pub fn new(messages: impl AsRef<[Message]>) -> ChatRequestBuilder {
        ChatRequestBuilder {
            messages: messages.as_ref().to_vec(),
            tools: None,
            options: RequestOptions::default(),
        }
    }
}

impl ChatRequestBuilder {
    pub fn tools(mut self, tools: impl AsRef<[Tool]>) -> Self {
        self.tools = Some(tools.as_ref().to_vec());
        self
    }

    /// 添加供应商原生请求参数。
    ///
    /// `OpenAI` / `Compatible` 会把它写入请求体顶层，`Qwen` 会写入 `parameters`。
    /// 当键名与库内置字段冲突时，内置字段优先，透传值会被忽略。
    pub fn provider_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.options.provider_options.insert(key.into(), value.into());
        self
    }

    /// 批量添加供应商原生请求参数。
    ///
    /// 参数落点与 [`Self::provider_option`] 一致；若同一键重复出现，后写入的值会覆盖先前值。
    pub fn provider_options<I, K, V>(mut self, options: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.options.provider_options.extend(
            options
                .into_iter()
                .map(|(key, value)| (key.into(), value.into())),
        );
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.options.top_p = Some(top_p);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.options.max_tokens = Some(max_tokens);
        self
    }

    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.options.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.options.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn thinking(mut self, enabled: bool) -> Self {
        self.options.thinking = enabled;
        self
    }

    pub fn thinking_budget(mut self, budget: u32) -> Self {
        self.options.thinking_budget = Some(budget);
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.options.reasoning_effort = Some(effort);
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.options.tool_choice = Some(tool_choice);
        self
    }

    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.options.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn build(self) -> ChatRequest {
        ChatRequest {
            messages: self.messages,
            tools: self.tools,
            options: self.options,
        }
    }
}

impl ChatRequest {
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }

    pub fn options(&self) -> &RequestOptions {
        &self.options
    }
}

#[derive(Clone)]
pub struct Client {
    provider_config: ProviderConfig,
    http: reqwest::Client,
    adapter: Arc<dyn ProviderAdapter>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("provider", &self.provider_config.provider)
            .finish_non_exhaustive()
    }
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    fn from_builder(provider_config: ProviderConfig) -> Self {
        let adapter = make_adapter(provider_config.provider);

        Self {
            provider_config,
            http: reqwest::Client::new(),
            adapter,
        }
    }

    pub const fn provider(&self) -> Provider {
        self.provider_config.provider
    }

    /// 发送非流式聊天请求。
    /// # Errors
    /// - [`LlmError::ApiError`]：当 Provider 返回业务失败或本地关键配置缺失时触发
    /// - [`LlmError::AuthError`]：当接口返回 `401` 时触发
    /// - [`LlmError::RateLimitError`]：当接口返回 `429` 时触发
    /// - [`LlmError::NetworkError`]：当网络请求失败时触发
    /// - [`LlmError::ParseError`]：当响应体解析失败时触发
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.send_chat(request.messages(), request.tools(), request.options())
            .await
    }

    /// 发送流式聊天请求。
    /// # Errors
    /// - [`LlmError::ApiError`]：当 Provider 返回业务失败或本地关键配置缺失时触发
    /// - [`LlmError::AuthError`]：当接口返回 `401` 时触发
    /// - [`LlmError::RateLimitError`]：当接口返回 `429` 时触发
    /// - [`LlmError::NetworkError`]：当网络请求失败时触发
    pub async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream, LlmError> {
        self.start_chat_stream(request.messages(), request.tools(), request.options())
            .await
    }

    /// 使用默认请求选项发送非流式聊天请求。
    ///
    /// 当调用方无需额外配置 `thinking`、`tool_choice` 等参数时，可以直接使用该快捷方法。
    pub async fn chat_messages(
        &self,
        messages: impl AsRef<[Message]>,
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest::new(messages).build();
        self.chat(&request).await
    }

    /// 使用默认请求选项发送流式聊天请求。
    ///
    /// 当调用方无需额外配置请求选项时，可以直接使用该快捷方法。
    pub async fn chat_stream_messages(
        &self,
        messages: impl AsRef<[Message]>,
    ) -> Result<ChatStream, LlmError> {
        let request = ChatRequest::new(messages).build();
        self.chat_stream(&request).await
    }

    async fn send_chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let provider_name = self.adapter.provider_name().to_string();
        let response = self.send_request(messages, tools, false, options).await?;
        let body = response.bytes().await.map_err(LlmError::from)?;
        tracing::debug!(
            provider = provider_name.as_str(),
            response_body = %String::from_utf8_lossy(body.as_ref()),
            "LLM 非流式响应"
        );
        self.adapter.parse_chat_response(body.as_ref())
    }

    async fn start_chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> Result<ChatStream, LlmError> {
        let provider_name = self.adapter.provider_name().to_string();
        let response = self.send_request(messages, tools, true, options).await?;
        let adapter = Arc::clone(&self.adapter);
        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |event_result| match event_result {
                Ok(event) => {
                    tracing::debug!(
                        provider = provider_name.as_str(),
                        stream_event = %event.data,
                        "LLM 流式响应事件"
                    );
                    match adapter.parse_stream_chunks(&event.data) {
                        Ok(chunks) => chunks.into_iter().map(Ok).collect::<Vec<_>>(),
                        Err(error) => vec![Err(error)],
                    }
                }
                Err(error) => vec![Err(LlmError::StreamError(format!(
                    "读取 SSE 事件失败：{error}"
                )))],
            })
            .flat_map(stream::iter);

        Ok(Box::pin(stream))
    }

    async fn send_request(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        stream: bool,
        options: &RequestOptions,
    ) -> Result<Response, LlmError> {
        let url = self.request_url()?;
        let model = self.required_model()?;
        let options = self.resolve_request_options(model, tools, options);
        let body = self
            .adapter
            .build_chat_request(model, messages, tools, stream, &options)?;
        let provider_name = self.adapter.provider_name().to_string();
        let request_body_json = serde_json::to_string(&body).unwrap_or_else(|error| {
            serde_json::json!({
                "serialization_error": error.to_string()
            })
            .to_string()
        });

        tracing::debug!(
            provider = provider_name.as_str(),
            model,
            stream,
            request_url = %url,
            request_body = %request_body_json,
            "LLM 请求"
        );

        let mut request = self.http.post(url).json(&body);
        request = request.header(
            "Authorization",
            format!("Bearer {}", self.provider_config.api_key),
        );

        request = apply_stream_headers(request, self.provider(), stream);

        if let Some(timeout_secs) = self.provider_config.timeout_secs {
            request = request.timeout(Duration::from_secs(timeout_secs));
        }

        if self.provider() == Provider::OpenAI
            && let Some(organization) = self.provider_config.organization.as_deref()
        {
            request = request.header("OpenAI-Organization", organization);
        }

        for (key, value) in &self.provider_config.extra_headers {
            request = request.header(key, value);
        }

        let response = request.send().await.map_err(LlmError::from)?;
        if response.status().is_success() {
            tracing::debug!(
                provider = provider_name.as_str(),
                status = response.status().as_u16(),
                "LLM 请求成功"
            );
            Ok(response)
        } else {
            let status = response.status();
            let retry_after = parse_retry_after_header(response.headers());
            let body = response.bytes().await.map_err(LlmError::from)?;
            tracing::debug!(
                provider = provider_name.as_str(),
                status = status.as_u16(),
                retry_after_secs = retry_after.map(|duration| duration.as_secs()),
                response_body = %String::from_utf8_lossy(body.as_ref()),
                "LLM 请求失败"
            );
            Err(map_http_error(
                status,
                retry_after,
                body.as_ref(),
                self.adapter.provider_name(),
            ))
        }
    }

    fn request_url(&self) -> Result<String, LlmError> {
        let base_url = self
            .provider_config
            .base_url
            .as_deref()
            .or_else(|| self.adapter.default_base_url())
            .ok_or_else(|| LlmError::ApiError {
                status_code: 0,
                message: "当前 Provider 需要显式设置 base_url".to_string(),
                provider: self.provider().display_name().to_string(),
            })?;

        Ok(join_url(base_url, self.adapter.chat_path()))
    }

    fn required_model(&self) -> Result<&str, LlmError> {
        self.provider_config
            .default_model
            .as_deref()
            .ok_or_else(|| LlmError::ApiError {
                status_code: 0,
                message: "尚未设置默认模型，请在构建器中调用 .model(...)".to_string(),
                provider: self.provider().display_name().to_string(),
            })
    }

    fn resolve_request_options(
        &self,
        model: &str,
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> RequestOptions {
        let capability = thinking_capability(self.provider(), model);
        let mut resolved = RequestOptions {
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.max_tokens,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            provider_options: options.provider_options.clone(),
            ..RequestOptions::default()
        };

        if options.thinking {
            if capability.supports_thinking {
                resolved.thinking = true;
            } else {
                tracing::debug!(
                    provider = self.provider().display_name(),
                    model,
                    "provider / model 不支持思考模式，thinking 参数已忽略"
                );
            }
        }

        if let Some(thinking_budget) = options.thinking_budget {
            if capability.supports_thinking_budget {
                resolved.thinking = true;
                resolved.thinking_budget = Some(thinking_budget);
            } else {
                tracing::debug!(
                    provider = self.provider().display_name(),
                    model,
                    thinking_budget,
                    "provider / model 不支持 thinking_budget，参数已忽略"
                );
            }
        }

        if let Some(reasoning_effort) = options.reasoning_effort {
            if capability.supports_reasoning_effort {
                resolved.reasoning_effort = Some(reasoning_effort);
            } else {
                tracing::debug!(
                    provider = self.provider().display_name(),
                    model,
                    reasoning_effort = reasoning_effort.as_str(),
                    "provider / model 不支持 reasoning_effort，参数已忽略"
                );
            }
        }

        if let Some(tool_choice) = options.tool_choice.clone() {
            if has_tools(tools) {
                resolved.tool_choice = Some(tool_choice);
            } else {
                tracing::debug!(
                    provider = self.provider().display_name(),
                    model,
                    "当前请求未传入 tools，tool_choice 参数已忽略"
                );
            }
        }

        if let Some(parallel_tool_calls) = options.parallel_tool_calls {
            if has_tools(tools) {
                resolved.parallel_tool_calls = Some(parallel_tool_calls);
            } else {
                tracing::debug!(
                    provider = self.provider().display_name(),
                    model,
                    parallel_tool_calls,
                    "当前请求未传入 tools，parallel_tool_calls 参数已忽略"
                );
            }
        }

        resolved
    }
}

fn make_adapter(provider: Provider) -> Arc<dyn ProviderAdapter> {
    match provider {
        Provider::OpenAI => Arc::new(OpenAiAdapter::new()),
        Provider::Qwen => Arc::new(QwenAdapter::new()),
        Provider::Compatible => Arc::new(CompatibleAdapter::new()),
    }
}

fn thinking_capability(provider: Provider, model: &str) -> ThinkingCapability {
    match provider {
        Provider::OpenAI if is_openai_reasoning_model(model) => ThinkingCapability {
            supports_thinking: true,
            supports_thinking_budget: false,
            supports_reasoning_effort: true,
        },
        Provider::Qwen if is_qwen3_reasoning_model(model) => ThinkingCapability {
            supports_thinking: true,
            supports_thinking_budget: true,
            supports_reasoning_effort: false,
        },
        Provider::Compatible if is_deepseek_reasoning_model(model) => ThinkingCapability {
            supports_thinking: true,
            supports_thinking_budget: false,
            supports_reasoning_effort: false,
        },
        _ => ThinkingCapability::default(),
    }
}

fn is_openai_reasoning_model(model: &str) -> bool {
    model.starts_with("o1") || model.starts_with("o3")
}

fn is_qwen3_reasoning_model(model: &str) -> bool {
    model.starts_with("qwen3")
}

fn is_deepseek_reasoning_model(model: &str) -> bool {
    let normalized = model.rsplit_once('/').map_or(model, |(_, suffix)| suffix);
    let normalized = normalized
        .rsplit_once(':')
        .map_or(normalized, |(_, suffix)| suffix);

    normalized == "deepseek-reasoner"
}

fn has_tools(tools: Option<&[Tool]>) -> bool {
    matches!(tools, Some(items) if !items.is_empty())
}

fn apply_stream_headers(
    request: reqwest::RequestBuilder,
    provider: Provider,
    stream: bool,
) -> reqwest::RequestBuilder {
    if !stream {
        return request;
    }

    let request = request.header("Accept", "text/event-stream");
    if provider == Provider::Qwen {
        request.header("X-DashScope-SSE", "enable")
    } else {
        request
    }
}

fn join_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let secs = raw.parse::<u64>().ok()?;
    Some(Duration::from_secs(secs))
}

fn map_http_error(
    status: StatusCode,
    retry_after: Option<Duration>,
    body: &[u8],
    provider_name: &str,
) -> LlmError {
    match status {
        StatusCode::UNAUTHORIZED => LlmError::AuthError,
        StatusCode::TOO_MANY_REQUESTS => LlmError::RateLimitError { retry_after },
        _ => LlmError::ApiError {
            status_code: status.as_u16(),
            message: extract_error_message(body, status),
            provider: provider_name.to_string(),
        },
    }
}

fn extract_error_message(body: &[u8], status: StatusCode) -> String {
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(message) = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(serde_json::Value::as_str)
        {
            return message.to_string();
        }

        if let Some(message) = value.get("message").and_then(serde_json::Value::as_str) {
            return message.to_string();
        }
    }

    let text = String::from_utf8_lossy(body).trim().to_string();
    if text.is_empty() {
        format!("请求失败，HTTP 状态码：{}", status.as_u16())
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use reqwest::StatusCode;
    use serde_json::json;

    use super::{
        ChatRequest, Client, RequestOptions, apply_stream_headers, extract_error_message, join_url,
        map_http_error, parse_retry_after_header,
    };
    use crate::{JsonType, LlmError, Message, Provider, ReasoningEffort, ToolChoice};

    #[test]
    fn client_provider() {
        let client = Client::builder()
            .provider(Provider::Compatible)
            .base_url("https://api.deepseek.com/v1")
            .api_key("sk-demo")
            .model("deepseek-chat")
            .build()
            .expect("应构建成功");

        assert_eq!(client.provider(), Provider::Compatible);
    }

    #[test]
    fn chat_request_builder_owns_messages_tools_and_options() {
        let messages = vec![Message::user("hello")];
        let tools = [crate::Tool::function("get_weather")
            .param("city", JsonType::String, "城市名称", true)
            .build()];

        let request = ChatRequest::new(&messages)
            .tools(&tools)
            .provider_option("max_completion_tokens", 4096)
            .provider_option("metadata", json!({ "tier": "pro" }))
            .temperature(0.7)
            .top_p(0.9)
            .max_tokens(2048)
            .presence_penalty(0.3)
            .frequency_penalty(0.1)
            .thinking(true)
            .tool_choice(ToolChoice::function("get_weather"))
            .parallel_tool_calls(true)
            .build();

        assert_eq!(request.messages().len(), 1);
        assert_eq!(request.tools().map(<[_]>::len), Some(1));
        assert_eq!(
            request.options().provider_options.get("max_completion_tokens"),
            Some(&json!(4096))
        );
        assert_eq!(
            request.options().provider_options.get("metadata"),
            Some(&json!({ "tier": "pro" }))
        );
        assert_eq!(request.options().temperature, Some(0.7));
        assert_eq!(request.options().top_p, Some(0.9));
        assert_eq!(request.options().max_tokens, Some(2048));
        assert_eq!(request.options().presence_penalty, Some(0.3));
        assert_eq!(request.options().frequency_penalty, Some(0.1));
        assert!(request.options().thinking);
        assert_eq!(request.options().parallel_tool_calls, Some(true));
        assert_eq!(
            request
                .options()
                .tool_choice
                .as_ref()
                .and_then(ToolChoice::function_name),
            Some("get_weather")
        );
    }

    #[test]
    fn chat_request_builder_accepts_provider_options_batch() {
        let request = ChatRequest::new([Message::user("hello")])
            .provider_option("seed", 7)
            .provider_options([
                ("seed", json!(8)),
                ("metadata", json!({ "tier": "pro" })),
                ("max_completion_tokens", json!(2048)),
            ])
            .build();

        assert_eq!(request.options().provider_options.get("seed"), Some(&json!(8)));
        assert_eq!(
            request.options().provider_options.get("metadata"),
            Some(&json!({ "tier": "pro" }))
        );
        assert_eq!(
            request.options().provider_options.get("max_completion_tokens"),
            Some(&json!(2048))
        );
    }

    #[test]
    fn join_url_base_url() {
        assert_eq!(
            join_url("https://api.openai.com/v1/", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn retry_after_duration() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("12"),
        );

        assert_eq!(
            parse_retry_after_header(&headers),
            Some(Duration::from_secs(12))
        );
    }

    #[test]
    fn map_http_error_returns_rate_limit_with_retry_after() {
        let error = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(5)),
            r#"{"message":"请求过于频繁"}"#.as_bytes(),
            "openai",
        );

        match error {
            LlmError::RateLimitError { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(5)));
            }
            other => panic!("错误类型不符合预期：{other:?}"),
        }
    }

    #[test]
    fn extract_error_message_reads_nested_error_message() {
        let message = extract_error_message(
            r#"{"error":{"message":"无效请求"}}"#.as_bytes(),
            StatusCode::BAD_REQUEST,
        );

        assert_eq!(message, "无效请求");
    }

    #[test]
    fn apply_stream_headers_adds_qwen_sse_header_for_qwen_streaming() {
        let client = reqwest::Client::new();
        let request = apply_stream_headers(
            client.post("https://example.com/chat"),
            Provider::Qwen,
            true,
        )
        .build()
        .expect("请求应构建成功");

        assert_eq!(
            request.headers().get("Accept"),
            Some(&reqwest::header::HeaderValue::from_static(
                "text/event-stream"
            ))
        );
        assert_eq!(
            request.headers().get("X-DashScope-SSE"),
            Some(&reqwest::header::HeaderValue::from_static("enable"))
        );
    }

    #[test]
    fn apply_stream_headers_skips_qwen_sse_header_for_non_qwen_streaming() {
        let client = reqwest::Client::new();
        let request = apply_stream_headers(
            client.post("https://example.com/chat"),
            Provider::OpenAI,
            true,
        )
        .build()
        .expect("请求应构建成功");

        assert_eq!(
            request.headers().get("Accept"),
            Some(&reqwest::header::HeaderValue::from_static(
                "text/event-stream"
            ))
        );
        assert!(request.headers().get("X-DashScope-SSE").is_none());
    }

    #[test]
    fn resolve_request_options_ignores_reasoning_effort_for_qwen3() {
        let client = Client::builder()
            .provider(Provider::Qwen)
            .api_key("sk-demo")
            .model("qwen3-max")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "qwen3-max",
            None,
            &RequestOptions {
                thinking: true,
                thinking_budget: Some(8000),
                reasoning_effort: Some(ReasoningEffort::High),
                ..RequestOptions::default()
            },
        );

        assert!(options.thinking);
        assert_eq!(options.thinking_budget, Some(8000));
        assert_eq!(options.reasoning_effort, None);
    }

    #[test]
    fn resolve_request_options_ignores_thinking_settings_for_non_reasoning_openai_model() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "gpt-4o",
            None,
            &RequestOptions {
                thinking: true,
                thinking_budget: Some(4000),
                reasoning_effort: Some(ReasoningEffort::High),
                ..RequestOptions::default()
            },
        );

        assert!(!options.thinking);
        assert_eq!(options.thinking_budget, None);
        assert_eq!(options.reasoning_effort, None);
    }

    #[test]
    fn resolve_request_options_keeps_tool_settings_when_tools_are_provided() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");
        let tools = [crate::Tool::function("get_weather")
            .param("city", crate::JsonType::String, "城市名称", true)
            .build()];

        let options = client.resolve_request_options(
            "gpt-4o",
            Some(&tools),
            &RequestOptions {
                tool_choice: Some(ToolChoice::function("get_weather")),
                parallel_tool_calls: Some(true),
                ..RequestOptions::default()
            },
        );

        assert_eq!(options.parallel_tool_calls, Some(true));
        assert_eq!(
            options
                .tool_choice
                .as_ref()
                .and_then(ToolChoice::function_name),
            Some("get_weather")
        );
    }

    #[test]
    fn resolve_request_options_keeps_sampling_settings() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "gpt-4o",
            None,
            &RequestOptions {
                temperature: Some(0.4),
                top_p: Some(0.8),
                max_tokens: Some(1024),
                presence_penalty: Some(0.2),
                frequency_penalty: Some(0.1),
                provider_options: serde_json::Map::from_iter([(
                    "max_completion_tokens".to_string(),
                    json!(1536),
                )]),
                ..RequestOptions::default()
            },
        );

        assert_eq!(options.temperature, Some(0.4));
        assert_eq!(options.top_p, Some(0.8));
        assert_eq!(options.max_tokens, Some(1024));
        assert_eq!(options.presence_penalty, Some(0.2));
        assert_eq!(options.frequency_penalty, Some(0.1));
        assert_eq!(
            options.provider_options.get("max_completion_tokens"),
            Some(&json!(1536))
        );
    }

    #[test]
    fn resolve_request_options_drops_tool_settings_when_tools_are_missing() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "gpt-4o",
            None,
            &RequestOptions {
                tool_choice: Some(ToolChoice::Required),
                parallel_tool_calls: Some(true),
                ..RequestOptions::default()
            },
        );

        assert_eq!(options.tool_choice, None);
        assert_eq!(options.parallel_tool_calls, None);
    }

    #[test]
    fn compatible_deepseek_reasoner() {
        let client = Client::builder()
            .provider(Provider::Compatible)
            .base_url("https://example.com/v1")
            .api_key("sk-demo")
            .model("vendor/deepseek-reasoner")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "vendor/deepseek-reasoner",
            None,
            &RequestOptions {
                thinking: true,
                ..RequestOptions::default()
            },
        );

        assert!(options.thinking);
    }
}
