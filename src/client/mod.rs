//! 客户端模块。
//!
//! 提供 `Client`、请求构建器和请求发送主流程。

use std::{
    collections::HashMap,
    future::{Future, IntoFuture},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt, stream};
use reqwest::{Response, StatusCode, header::RETRY_AFTER};

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool, ToolChoice,
    provider::{compatible::CompatibleAdapter, openai::OpenAiAdapter, qwen::QwenAdapter},
    types::response::ReasoningEffort,
};

pub mod builder;

pub use builder::{
    ApiKeySet, ApiKeyUnset, ClientBuilder, ClientConfig, CompatibleConfig, OpenAiConfig,
    ProviderConfig, ProviderSet, ProviderUnset, QwenConfig,
};

/// 聊天流式响应类型。
///
/// 该类型是对异步流返回值的统一封装。流中的每一项都是一次增量输出或终止片段。
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>;

/// 单次聊天请求的附加选项。
///
/// 该类型承载由 `Client` 入口构建的请求级配置，例如思考模式、思考预算与推理强度。
/// 普通调用方通常无需直接构造它，而是通过 [`ChatRequestBuilder`] 或
/// [`ChatStreamRequestBuilder`] 链式设置。
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    thinking: bool,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<ReasoningEffort>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
}

impl RequestOptions {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            thinking: false,
            thinking_budget: None,
            reasoning_effort: None,
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }

    #[must_use]
    pub const fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    #[must_use]
    pub const fn with_thinking_budget(mut self, thinking_budget: u32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    #[must_use]
    pub const fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    #[must_use]
    pub const fn thinking(&self) -> bool {
        self.thinking
    }

    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u32> {
        self.thinking_budget
    }

    #[must_use]
    pub const fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }

    #[must_use]
    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    #[must_use]
    pub const fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ThinkingCapability {
    supports_thinking: bool,
    supports_thinking_budget: bool,
    supports_reasoning_effort: bool,
}

/// 非流式聊天请求构建器。
///
/// 该类型允许在真正发起请求前，为单次对话追加思考模式相关配置，并通过 `.await`
/// 直接执行请求。
pub struct ChatRequestBuilder<'a> {
    client: &'a Client,
    messages: &'a [Message],
    tools: Option<&'a [Tool]>,
    options: RequestOptions,
}

impl<'a> ChatRequestBuilder<'a> {
    fn new(client: &'a Client, messages: &'a [Message], tools: Option<&'a [Tool]>) -> Self {
        Self {
            client,
            messages,
            tools,
            options: RequestOptions::default(),
        }
    }

    #[must_use]
    pub fn thinking(mut self, enabled: bool) -> Self {
        self.options.thinking = enabled;
        self
    }

    #[must_use]
    pub fn thinking_budget(mut self, budget: u32) -> Self {
        self.options.thinking_budget = Some(budget);
        self
    }

    #[must_use]
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.options.reasoning_effort = Some(effort);
        self
    }

    #[must_use]
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.options.tool_choice = Some(tool_choice);
        self
    }

    #[must_use]
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.options.parallel_tool_calls = Some(enabled);
        self
    }
}

impl<'a> IntoFuture for ChatRequestBuilder<'a> {
    type Output = Result<ChatResponse, LlmError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            self.client
                .send_chat(self.messages, self.tools, &self.options)
                .await
        })
    }
}

/// 流式聊天请求构建器。
///
/// 该类型允许在启动流式请求前，为单次调用附加思考模式相关配置，并通过 `.await`
/// 直接启动流。
pub struct ChatStreamRequestBuilder<'a> {
    client: &'a Client,
    messages: &'a [Message],
    tools: Option<&'a [Tool]>,
    options: RequestOptions,
}

impl<'a> ChatStreamRequestBuilder<'a> {
    fn new(client: &'a Client, messages: &'a [Message], tools: Option<&'a [Tool]>) -> Self {
        Self {
            client,
            messages,
            tools,
            options: RequestOptions::default(),
        }
    }

    #[must_use]
    pub fn thinking(mut self, enabled: bool) -> Self {
        self.options.thinking = enabled;
        self
    }

    #[must_use]
    pub fn thinking_budget(mut self, budget: u32) -> Self {
        self.options.thinking_budget = Some(budget);
        self
    }

    #[must_use]
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.options.reasoning_effort = Some(effort);
        self
    }

    #[must_use]
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.options.tool_choice = Some(tool_choice);
        self
    }

    #[must_use]
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.options.parallel_tool_calls = Some(enabled);
        self
    }
}

impl<'a> IntoFuture for ChatStreamRequestBuilder<'a> {
    type Output = Result<ChatStream, LlmError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            self.client
                .start_chat_stream(self.messages, self.tools, &self.options)
                .await
        })
    }
}

/// `LLM` 客户端。
///
/// 该类型是 SDK 对外的主入口，负责根据构建配置发送聊天请求并解析响应。
#[derive(Clone)]
pub struct Client {
    config: ClientConfig,
    http: reqwest::Client,
    adapter: Arc<dyn ProviderAdapter>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("config", &self.config)
            .field("provider", &self.config.provider())
            .finish_non_exhaustive()
    }
}

impl Client {
    #[must_use]
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    pub(crate) fn from_config(config: ClientConfig) -> Self {
        let adapter = make_adapter(config.provider());

        Self {
            config,
            http: reqwest::Client::new(),
            adapter,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// 发送非流式聊天请求。
    /// # Errors
    /// - [`LlmError::ApiError`]：当 Provider 返回业务失败或本地关键配置缺失时触发
    /// - [`LlmError::AuthError`]：当接口返回 `401` 时触发
    /// - [`LlmError::RateLimitError`]：当接口返回 `429` 时触发
    /// - [`LlmError::NetworkError`]：当网络请求失败时触发
    /// - [`LlmError::ParseError`]：当响应体解析失败时触发
    #[must_use]
    pub fn chat<'a>(&'a self, messages: &'a [Message]) -> ChatRequestBuilder<'a> {
        ChatRequestBuilder::new(self, messages, None)
    }

    /// 发送带工具定义的非流式聊天请求。
    /// # Errors
    /// - [`LlmError::UnsupportedFeature`]：当当前 Provider 不支持工具调用时触发
    /// - 其余错误与 [`Client::chat`] 相同
    #[must_use]
    pub fn chat_with_tools<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [Tool],
    ) -> ChatRequestBuilder<'a> {
        ChatRequestBuilder::new(self, messages, Some(tools))
    }

    /// 发送流式聊天请求。
    /// # Errors
    /// - [`LlmError::ApiError`]：当 Provider 返回业务失败或本地关键配置缺失时触发
    /// - [`LlmError::AuthError`]：当接口返回 `401` 时触发
    /// - [`LlmError::RateLimitError`]：当接口返回 `429` 时触发
    /// - [`LlmError::NetworkError`]：当网络请求失败时触发
    #[must_use]
    pub fn chat_stream<'a>(&'a self, messages: &'a [Message]) -> ChatStreamRequestBuilder<'a> {
        ChatStreamRequestBuilder::new(self, messages, None)
    }

    /// 发送带工具定义的流式聊天请求。
    /// # Errors
    /// - [`LlmError::UnsupportedFeature`]：当当前 Provider 不支持工具调用时触发
    /// - 其余错误与 [`Client::chat_stream`] 相同
    #[must_use]
    pub fn chat_stream_with_tools<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [Tool],
    ) -> ChatStreamRequestBuilder<'a> {
        ChatStreamRequestBuilder::new(self, messages, Some(tools))
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
        request = request.header("Authorization", format!("Bearer {}", self.api_key()));

        request = apply_stream_headers(request, self.config.provider(), stream);

        if let Some(timeout_secs) = self.timeout_secs() {
            request = request.timeout(Duration::from_secs(timeout_secs));
        }

        if self.config.provider() == Provider::OpenAI
            && let Some(organization) = self.organization()
        {
            request = request.header("OpenAI-Organization", organization);
        }

        if let Some(extra_headers) = self.extra_headers() {
            for (key, value) in extra_headers {
                request = request.header(key, value);
            }
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
            .base_url()
            .or_else(|| self.adapter.default_base_url())
            .ok_or_else(|| LlmError::ApiError {
                status_code: 0,
                message: "当前 Provider 需要显式设置 base_url".to_string(),
                provider: self.config.provider().display_name().to_string(),
            })?;

        Ok(join_url(base_url, self.adapter.chat_path()))
    }

    fn required_model(&self) -> Result<&str, LlmError> {
        self.default_model().ok_or_else(|| LlmError::ApiError {
            status_code: 0,
            message: "尚未设置默认模型，请在构建器中调用 .model(...)".to_string(),
            provider: self.config.provider().display_name().to_string(),
        })
    }

    fn api_key(&self) -> &str {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.api_key(),
            ProviderConfig::Qwen(config) => config.api_key(),
            ProviderConfig::Compatible(config) => config.api_key(),
        }
    }

    fn base_url(&self) -> Option<&str> {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.base_url(),
            ProviderConfig::Qwen(config) => config.base_url(),
            ProviderConfig::Compatible(config) => config.base_url(),
        }
    }

    fn organization(&self) -> Option<&str> {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.organization(),
            ProviderConfig::Qwen(config) => config.organization(),
            ProviderConfig::Compatible(config) => config.organization(),
        }
    }

    fn default_model(&self) -> Option<&str> {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.default_model(),
            ProviderConfig::Qwen(config) => config.default_model(),
            ProviderConfig::Compatible(config) => config.default_model(),
        }
    }

    fn timeout_secs(&self) -> Option<u64> {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.timeout_secs(),
            ProviderConfig::Qwen(config) => config.timeout_secs(),
            ProviderConfig::Compatible(config) => config.timeout_secs(),
        }
    }

    fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        match self.config.provider_config() {
            ProviderConfig::OpenAi(config) => config.extra_headers(),
            ProviderConfig::Qwen(config) => config.extra_headers(),
            ProviderConfig::Compatible(config) => config.extra_headers(),
        }
    }

    fn resolve_request_options(
        &self,
        model: &str,
        tools: Option<&[Tool]>,
        options: &RequestOptions,
    ) -> RequestOptions {
        let capability = thinking_capability(self.config.provider(), model);
        let mut resolved = RequestOptions::default();

        if options.thinking {
            if capability.supports_thinking {
                resolved.thinking = true;
            } else {
                tracing::debug!(
                    provider = self.config.provider().display_name(),
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
                    provider = self.config.provider().display_name(),
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
                    provider = self.config.provider().display_name(),
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
                    provider = self.config.provider().display_name(),
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
                    provider = self.config.provider().display_name(),
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
    let normalized = model
        .rsplit_once('/')
        .map_or(model, |(_, suffix)| suffix);
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

    use super::{
        Client, RequestOptions, apply_stream_headers, extract_error_message, join_url,
        map_http_error, parse_retry_after_header,
    };
    use crate::{LlmError, Provider, ReasoningEffort, ToolChoice};

    #[test]
    fn client_provider() {
        let client = Client::builder()
            .provider(Provider::Compatible)
            .base_url("https://api.deepseek.com/v1")
            .api_key("sk-demo")
            .model("deepseek-chat")
            .build()
            .expect("应构建成功");

        assert_eq!(client.config().provider(), Provider::Compatible);
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

        assert_eq!(parse_retry_after_header(&headers), Some(Duration::from_secs(12)));
    }

    #[test]
    fn http() {
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
    fn json() {
        let message = extract_error_message(
            r#"{"error":{"message":"无效请求"}}"#.as_bytes(),
            StatusCode::BAD_REQUEST,
        );

        assert_eq!(message, "无效请求");
    }

    #[test]
    fn qwen_dashscope_sse() {
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
    fn qwen_dashscope_sse_2() {
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
    fn qwen3_reasoning_effort() {
        let client = Client::builder()
            .provider(Provider::Qwen)
            .api_key("sk-demo")
            .model("qwen3-max")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "qwen3-max",
            None,
            &RequestOptions::new()
                .with_thinking(true)
                .with_thinking_budget(8000)
                .with_reasoning_effort(ReasoningEffort::High),
        );

        assert!(options.thinking());
        assert_eq!(options.thinking_budget(), Some(8000));
        assert_eq!(options.reasoning_effort(), None);
    }

    #[test]
    fn mod_test() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "gpt-4o",
            None,
            &RequestOptions::new()
                .with_thinking(true)
                .with_thinking_budget(4000)
                .with_reasoning_effort(ReasoningEffort::High),
        );

        assert!(!options.thinking());
        assert_eq!(options.thinking_budget(), None);
        assert_eq!(options.reasoning_effort(), None);
    }

    #[test]
    fn tool_choice_2() {
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
            &RequestOptions::new()
                .with_tool_choice(ToolChoice::function("get_weather"))
                .with_parallel_tool_calls(true),
        );

        assert_eq!(options.parallel_tool_calls(), Some(true));
        assert_eq!(
            options.tool_choice().and_then(ToolChoice::function_name),
            Some("get_weather")
        );
    }

    #[test]
    fn tool_choice_3() {
        let client = Client::builder()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        let options = client.resolve_request_options(
            "gpt-4o",
            None,
            &RequestOptions::new()
                .with_tool_choice(ToolChoice::Required)
                .with_parallel_tool_calls(true),
        );

        assert_eq!(options.tool_choice(), None);
        assert_eq!(options.parallel_tool_calls(), None);
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
            &RequestOptions::new().with_thinking(true),
        );

        assert!(options.thinking());
    }
}
