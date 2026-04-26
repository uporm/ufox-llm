use futures::Stream;

use crate::{
    error::LlmError,
    middleware::{RateLimitConfig, RetryConfig, Transport, TransportConfig},
    provider::{ApiProtocol, Provider, ProviderAdapter},
    types::{request::*, response::*},
};

/// `Client` 构造器。
pub struct ClientBuilder {
    provider: Option<Provider>,
    api_protocol: Option<ApiProtocol>,
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    timeout_secs: u64,
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
    max_retries: u32,
    /// 本地限流的每分钟最大请求数
    rate_limit_rpm: Option<u32>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            api_protocol: None,
            base_url: None,
            api_key: None,
            model: None,
            timeout_secs: 600,
            connect_timeout_secs: 5,
            read_timeout_secs: 60,
            max_retries: 3,
            rate_limit_rpm: None,
        }
    }
}

impl ClientBuilder {
    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// 显式指定线路协议。
    ///
    /// 不设置时按 provider 给默认值：`Provider::OpenAI` 默认 `Responses`，其余默认 `ChatCompletions`。
    /// 仅 `Provider::OpenAI` 和 `Provider::Compatible` 支持 `ApiProtocol::Responses`。
    pub fn api_protocol(mut self, protocol: ApiProtocol) -> Self {
        self.api_protocol = Some(protocol);
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// 设置连接建立超时（秒）。
    pub fn connect_timeout_secs(mut self, secs: u64) -> Self {
        self.connect_timeout_secs = secs;
        self
    }

    /// 设置单次读取超时（秒）。
    pub fn read_timeout_secs(mut self, secs: u64) -> Self {
        self.read_timeout_secs = secs;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// 设置客户端本地限流的每分钟最大请求数。
    ///
    /// 该限制在 SDK 侧生效，用于主动控制请求发送频率；`None` 表示不启用本地限流。
    pub fn rate_limit_rpm(mut self, rpm: u32) -> Self {
        self.rate_limit_rpm = Some(rpm);
        self
    }

    /// 构建 `Client`。
    pub fn build(self) -> Result<Client, LlmError> {
        let provider = self
            .provider
            .ok_or(LlmError::MissingConfig { field: "provider" })?;
        let api_key = self
            .api_key
            .ok_or(LlmError::MissingConfig { field: "api_key" })?;
        let model = self
            .model
            .ok_or(LlmError::MissingConfig { field: "model" })?;
        let base_url = match self.base_url {
            Some(url) => url,
            None => provider
                .default_base_url()
                .ok_or(LlmError::MissingConfig { field: "base_url" })?
                .to_owned(),
        };

        let protocol = self.api_protocol.unwrap_or_else(|| provider.default_protocol());

        // 仅 OpenAI 和 Compatible（指向自定义 endpoint）支持 Responses API。
        if protocol == ApiProtocol::Responses
            && !matches!(provider, Provider::OpenAI | Provider::Compatible)
        {
            return Err(LlmError::InvalidConfig {
                message: format!(
                    "ApiProtocol::Responses 仅支持 Provider::OpenAI 和 Provider::Compatible，当前 provider 为 {}",
                    provider.name()
                ),
            });
        }

        let transport = Transport::new(TransportConfig {
            timeout_secs: self.timeout_secs,
            connect_timeout_secs: self.connect_timeout_secs,
            read_timeout_secs: self.read_timeout_secs,
            retry: RetryConfig {
                max_retries: self.max_retries,
                ..RetryConfig::default()
            },
            rate_limit: self.rate_limit_rpm.map(|rpm| RateLimitConfig {
                requests_per_minute: rpm,
            }),
        });
        let adapter = provider.into_adapter(protocol, &api_key, &base_url, &transport)?;

        Ok(Client {
            adapter,
            model,
            provider,
            base_url,
            timeout_secs: self.timeout_secs,
            connect_timeout_secs: self.connect_timeout_secs,
            read_timeout_secs: self.read_timeout_secs,
            max_retries: self.max_retries,
            rate_limit_rpm: self.rate_limit_rpm,
        })
    }
}

/// SDK 唯一对外入口。
pub struct Client {
    adapter: Box<dyn ProviderAdapter>,
    model: String,
    provider: Provider,
    base_url: String,
    timeout_secs: u64,
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
    max_retries: u32,
    rate_limit_rpm: Option<u32>,
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// 从环境变量或当前目录的 `.env` 文件快速构建，且 `.env` 优先级更高。
    pub fn from_env() -> Result<Self, LlmError> {
        // `.env` 需要覆盖已有环境变量，才能让本地显式配置具备最高优先级。
        let _ = dotenvy::dotenv_override();
        let raw_provider =
            std::env::var("UFOX_LLM_PROVIDER").map_err(|_| LlmError::MissingConfig {
                field: "UFOX_LLM_PROVIDER",
            })?;
        let api_key = std::env::var("UFOX_LLM_API_KEY").map_err(|_| LlmError::MissingConfig {
            field: "UFOX_LLM_API_KEY",
        })?;
        let model = std::env::var("UFOX_LLM_MODEL").map_err(|_| LlmError::MissingConfig {
            field: "UFOX_LLM_MODEL",
        })?;
        let base_url = std::env::var("UFOX_LLM_BASE_URL").ok();
        let timeout_secs = Self::read_timeout_env("UFOX_LLM_TIMEOUT_SECS")?;
        let connect_timeout_secs = Self::read_timeout_env("UFOX_LLM_CONNECT_TIMEOUT_SECS")?;
        let read_timeout_secs = Self::read_timeout_env("UFOX_LLM_READ_TIMEOUT_SECS")?;
        Self::build_from_env_vars(
            raw_provider,
            api_key,
            model,
            base_url,
            TimeoutOverrides {
                timeout_secs,
                connect_timeout_secs,
                read_timeout_secs,
            },
        )
    }

    fn read_timeout_env(name: &'static str) -> Result<Option<u64>, LlmError> {
        std::env::var(name)
            .ok()
            .map(|value| {
                value.parse::<u64>().map_err(|_| LlmError::InvalidConfig {
                    message: format!("{name} 必须是非负整数秒"),
                })
            })
            .transpose()
    }

    fn build_from_env_vars(
        raw_provider: String,
        api_key: String,
        model: String,
        base_url: Option<String>,
        timeout_overrides: TimeoutOverrides,
    ) -> Result<Self, LlmError> {
        let provider = Provider::from_name(&raw_provider).ok_or(LlmError::InvalidConfig {
            message: format!("不支持的 LLM_PROVIDER 值：{raw_provider}"),
        })?;

        let mut builder = Client::builder()
            .provider(provider)
            .api_key(api_key)
            .model(model);
        if let Some(base_url) = base_url {
            builder = builder.base_url(base_url);
        }
        if let Some(timeout_secs) = timeout_overrides.timeout_secs {
            builder = builder.timeout_secs(timeout_secs);
        }
        if let Some(connect_timeout_secs) = timeout_overrides.connect_timeout_secs {
            builder = builder.connect_timeout_secs(connect_timeout_secs);
        }
        if let Some(read_timeout_secs) = timeout_overrides.read_timeout_secs {
            builder = builder.read_timeout_secs(read_timeout_secs);
        }
        builder.build()
    }

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.adapter.chat(&self.model, req).await
    }

    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<impl Stream<Item = Result<ChatChunk, LlmError>> + Send, LlmError> {
        self.adapter.chat_stream(&self.model, req).await
    }

    pub async fn embed(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        self.adapter.embed(&self.model, req).await
    }

    pub async fn speech_to_text(
        &self,
        req: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, LlmError> {
        self.adapter.speech_to_text(&self.model, req).await
    }

    pub async fn text_to_speech(
        &self,
        req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, LlmError> {
        self.adapter.text_to_speech(&self.model, req).await
    }

    pub async fn generate_image(&self, req: ImageGenRequest) -> Result<ImageGenResponse, LlmError> {
        self.adapter.generate_image(&self.model, req).await
    }

    pub async fn generate_video(&self, req: VideoGenRequest) -> Result<VideoGenResponse, LlmError> {
        self.adapter.generate_video(&self.model, req).await
    }

    pub async fn poll_video_task(&self, task_id: &str) -> Result<VideoGenResponse, LlmError> {
        self.adapter.poll_video_task(task_id).await
    }

    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// 返回连接建立超时（秒）。
    pub fn connect_timeout_secs(&self) -> u64 {
        self.connect_timeout_secs
    }

    /// 返回单次读取超时（秒）。
    pub fn read_timeout_secs(&self) -> u64 {
        self.read_timeout_secs
    }

    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// 返回客户端本地限流的每分钟最大请求数。
    ///
    /// 返回 `None` 表示未启用本地限流。
    pub fn rate_limit_rpm(&self) -> Option<u32> {
        self.rate_limit_rpm
    }
}

#[derive(Default)]
struct TimeoutOverrides {
    timeout_secs: Option<u64>,
    connect_timeout_secs: Option<u64>,
    read_timeout_secs: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::{Client, TimeoutOverrides};

    #[test]
    fn build_from_env_vars_builds_client_from_values() {
        let client = Client::build_from_env_vars(
            "openai".into(),
            "test-key".into(),
            "gpt-4o-mini".into(),
            None,
            TimeoutOverrides::default(),
        )
        .expect("should build client from env values");

        assert_eq!(client.provider().name(), "openai");
        assert_eq!(client.model(), "gpt-4o-mini");
        assert_eq!(client.base_url(), "https://api.openai.com/v1");
        assert_eq!(client.timeout_secs(), 600);
        assert_eq!(client.connect_timeout_secs(), 5);
        assert_eq!(client.read_timeout_secs(), 60);
    }

    #[test]
    fn build_from_env_vars_prefers_explicit_base_url_when_present() {
        let client = Client::build_from_env_vars(
            "compatible".into(),
            "test-key".into(),
            "custom-model".into(),
            Some("https://example.com/v1".into()),
            TimeoutOverrides::default(),
        )
        .expect("should build client with explicit base url");

        assert_eq!(client.provider().name(), "compatible");
        assert_eq!(client.base_url(), "https://example.com/v1");
    }

    #[test]
    fn build_from_env_vars_applies_timeout_overrides() {
        let client = Client::build_from_env_vars(
            "openai".into(),
            "test-key".into(),
            "gpt-4o-mini".into(),
            None,
            TimeoutOverrides {
                timeout_secs: Some(120),
                connect_timeout_secs: Some(3),
                read_timeout_secs: Some(15),
            },
        )
        .expect("should apply timeout overrides");

        assert_eq!(client.timeout_secs(), 120);
        assert_eq!(client.connect_timeout_secs(), 3);
        assert_eq!(client.read_timeout_secs(), 15);
    }
}
