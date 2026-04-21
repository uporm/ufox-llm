//! Client 构建器。
//!
//! 提供客户端构建器，并直接组装 `Client` 的运行时配置字段。

use std::{collections::HashMap, time::Duration};

use crate::Provider;

use super::{Client, config::ProviderConfig};

/// 客户端构建器。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientBuilder {
    provider_config: ProviderConfig,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            provider_config: ProviderConfig {
                provider: Provider::OpenAI,
                api_key: "".to_string(),
                base_url: None,
                organization: None,
                default_model: None,
                timeout_secs: None,
                extra_headers: HashMap::new(),
            },
        }
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider_config.provider = provider;
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.provider_config.api_key = api_key.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.provider_config.base_url = Some(base_url.into());
        self
    }

    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.provider_config.organization = Some(organization.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.provider_config.default_model = Some(model.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.provider_config.timeout_secs = Some(duration_to_timeout_secs(timeout));
        self
    }

    /// 添加一个额外请求头。
    pub fn extra_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.provider_config
            .extra_headers
            .insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> Result<Client, crate::LlmError> {
        let provider_config = self.provider_config;
        if provider_config.api_key.is_empty() {
            return Err(crate::LlmError::ValidationError(
                "尚未设置 api_key，请在构建器中调用 .api_key(...)".to_string(),
            ));
        }

        Ok(Client::from_builder(provider_config))
    }
}

fn duration_to_timeout_secs(duration: Duration) -> u64 {
    let secs = duration.as_secs();
    if secs == 0 && duration.subsec_nanos() > 0 {
        1
    } else {
        secs
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ClientBuilder;
    use crate::{LlmError, Provider};

    #[test]
    fn build_persists_basic_openai_fields() {
        let client = ClientBuilder::new()
            .provider(Provider::OpenAI)
            .api_key("sk-demo")
            .model("gpt-4o")
            .build()
            .expect("应构建成功");

        assert_eq!(client.provider(), Provider::OpenAI);
        assert_eq!(client.provider_config.api_key, "sk-demo");
        assert_eq!(
            client.provider_config.default_model.as_deref(),
            Some("gpt-4o")
        );
    }

    #[test]
    fn build_uses_openai_as_default_provider() {
        let client = ClientBuilder::new()
            .api_key("sk-demo")
            .build()
            .expect("应构建成功");

        assert_eq!(client.provider(), Provider::OpenAI);
    }

    #[test]
    fn build_requires_api_key() {
        let missing_api_key = ClientBuilder::new().provider(Provider::OpenAI).build();

        assert!(matches!(
            missing_api_key,
            Err(LlmError::ValidationError(message)) if message.contains("api_key")
        ));
    }

    #[test]
    fn build_persists_qwen_headers_and_model() {
        let client = ClientBuilder::new()
            .provider(Provider::Qwen)
            .api_key("sk-qwen")
            .model("qwen-max")
            .extra_header("X-Trace-Id", "demo-request")
            .build()
            .expect("应构建成功");

        assert_eq!(client.provider(), Provider::Qwen);
        assert_eq!(client.provider_config.api_key, "sk-qwen");
        assert_eq!(
            client.provider_config.default_model.as_deref(),
            Some("qwen-max")
        );
        assert_eq!(
            client
                .provider_config
                .extra_headers
                .get("X-Trace-Id")
                .map(String::as_str),
            Some("demo-request")
        );
    }

    #[test]
    fn timeout_rounds_up_subsecond_duration() {
        let client = ClientBuilder::new()
            .provider(Provider::OpenAI)
            .api_key("sk-openai")
            .timeout(Duration::from_millis(500))
            .build()
            .expect("应构建成功");

        assert_eq!(client.provider_config.timeout_secs, Some(1));
    }
}
