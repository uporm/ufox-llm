//! `ClientBuilder` 模块。
//!
//! 该模块负责定义客户端构建器、Provider 独立配置结构体以及 typestate 编译期校验逻辑。
//!
//! 设计上采用“统一构建入口 + 编译期状态收敛”的方式：
//! 1. 调用方始终从同一个 [`ClientBuilder`] 入口链式配置；
//! 2. `provider` 与 `api_key` 通过 typestate 在编译期建模，确保缺少关键配置时
//!    `build()` 方法根本不可见，而不是等到运行时再报错；
//! 3. `OpenAI`、`Qwen`、`Compatible` 各自拥有独立配置结构体，便于后续 `client`
//!    层在不引入大量分支判断的情况下读取对应 Provider 的配置。
//!
//! 该模块依赖上级 `client` 模块中的最小 `Client` 骨架，并依赖 `provider` 模块中的
//! [`crate::Provider`] 枚举表达供应商类型。

use std::{collections::HashMap, marker::PhantomData, time::Duration};

use crate::Provider;

use super::Client;

/// 构建器尚未设置 `provider` 的状态标记。
///
/// 该标记仅用于 typestate 编译期约束，不承载运行时数据。
///
/// # 示例
/// ```rust
/// use ufox_llm::ProviderUnset;
///
/// let state = ProviderUnset;
/// assert_eq!(format!("{state:?}"), "ProviderUnset");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ProviderUnset;

/// 构建器已经设置 `provider` 的状态标记。
///
/// 该标记仅用于 typestate 编译期约束，不承载运行时数据。
///
/// # 示例
/// ```rust
/// use ufox_llm::ProviderSet;
///
/// let state = ProviderSet;
/// assert_eq!(format!("{state:?}"), "ProviderSet");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ProviderSet;

/// 构建器尚未设置 `api_key` 的状态标记。
///
/// 该标记仅用于 typestate 编译期约束，不承载运行时数据。
///
/// # 示例
/// ```rust
/// use ufox_llm::ApiKeyUnset;
///
/// let state = ApiKeyUnset;
/// assert_eq!(format!("{state:?}"), "ApiKeyUnset");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ApiKeyUnset;

/// 构建器已经设置 `api_key` 的状态标记。
///
/// 该标记仅用于 typestate 编译期约束，不承载运行时数据。
///
/// # 示例
/// ```rust
/// use ufox_llm::ApiKeySet;
///
/// let state = ApiKeySet;
/// assert_eq!(format!("{state:?}"), "ApiKeySet");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ApiKeySet;

/// `OpenAI` Provider 配置。
///
/// 该配置结构体描述 `OpenAI` 客户端实例的静态配置项。虽然 `organization` 仅对
/// `OpenAI` 生效，但其余字段语义与其他 Provider 保持对齐，便于上层统一建模。
///
/// # 示例
/// ```rust
/// use ufox_llm::OpenAiConfig;
///
/// let config = OpenAiConfig::new("sk-demo");
/// assert_eq!(config.api_key(), "sk-demo");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiConfig {
    api_key: String,
    base_url: Option<String>,
    organization: Option<String>,
    default_model: Option<String>,
    timeout_secs: Option<u64>,
    extra_headers: Option<HashMap<String, String>>,
}

impl OpenAiConfig {
    /// 创建 `OpenAI` 配置。
    ///
    /// # Arguments
    /// * `api_key` - `OpenAI` API Key
    ///
    /// # Returns
    /// 仅包含 `api_key` 的基础配置对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::OpenAiConfig;
    ///
    /// let config = OpenAiConfig::new("sk-openai");
    /// assert_eq!(config.api_key(), "sk-openai");
    /// ```
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            organization: None,
            default_model: None,
            timeout_secs: None,
            extra_headers: None,
        }
    }

    /// 返回 API Key。
    ///
    /// # Returns
    /// 当前配置中的 API Key。
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// 返回基础地址覆盖项。
    ///
    /// # Returns
    /// 若调用方显式设置了 `base_url`，则返回该值。
    #[must_use]
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// 返回组织标识。
    ///
    /// # Returns
    /// 若设置了 `organization`，则返回该值。
    #[must_use]
    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    /// 返回默认模型名称。
    ///
    /// # Returns
    /// 若设置了默认模型，则返回该值。
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// 返回超时时间秒数。
    ///
    /// # Returns
    /// 若设置了超时时间，则返回秒数。
    #[must_use]
    pub const fn timeout_secs(&self) -> Option<u64> {
        self.timeout_secs
    }

    /// 返回额外请求头集合。
    ///
    /// # Returns
    /// 若设置了额外请求头，则返回只读映射。
    #[must_use]
    pub fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        self.extra_headers.as_ref()
    }
}

/// `Qwen` Provider 配置。
///
/// 该配置结构体描述 `Qwen` / `DashScope` 客户端实例的静态配置项。`extra_headers`
/// 常用于配置 `Qwen` 特有的鉴权或流式协商头。
///
/// # 示例
/// ```rust
/// use ufox_llm::QwenConfig;
///
/// let config = QwenConfig::new("sk-demo");
/// assert_eq!(config.api_key(), "sk-demo");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenConfig {
    api_key: String,
    base_url: Option<String>,
    organization: Option<String>,
    default_model: Option<String>,
    timeout_secs: Option<u64>,
    extra_headers: Option<HashMap<String, String>>,
}

impl QwenConfig {
    /// 创建 `Qwen` 配置。
    ///
    /// # Arguments
    /// * `api_key` - `Qwen` API Key
    ///
    /// # Returns
    /// 仅包含 `api_key` 的基础配置对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::QwenConfig;
    ///
    /// let config = QwenConfig::new("sk-qwen");
    /// assert_eq!(config.api_key(), "sk-qwen");
    /// ```
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            organization: None,
            default_model: None,
            timeout_secs: None,
            extra_headers: None,
        }
    }

    /// 返回 API Key。
    ///
    /// # Returns
    /// 当前配置中的 API Key。
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// 返回基础地址覆盖项。
    ///
    /// # Returns
    /// 若调用方显式设置了 `base_url`，则返回该值。
    #[must_use]
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// 返回组织标识。
    ///
    /// # Returns
    /// 若设置了 `organization`，则返回该值。
    #[must_use]
    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    /// 返回默认模型名称。
    ///
    /// # Returns
    /// 若设置了默认模型，则返回该值。
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// 返回超时时间秒数。
    ///
    /// # Returns
    /// 若设置了超时时间，则返回秒数。
    #[must_use]
    pub const fn timeout_secs(&self) -> Option<u64> {
        self.timeout_secs
    }

    /// 返回额外请求头集合。
    ///
    /// # Returns
    /// 若设置了额外请求头，则返回只读映射。
    #[must_use]
    pub fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        self.extra_headers.as_ref()
    }
}

/// 兼容 `OpenAI` 协议 Provider 配置。
///
/// 该配置结构体面向 `DeepSeek`、`Ollama`、自建兼容网关等服务。与官方 `OpenAI`
/// 不同，这类服务通常要求调用方显式设置 `base_url`。
///
/// # 示例
/// ```rust
/// use ufox_llm::CompatibleConfig;
///
/// let config = CompatibleConfig::new("sk-demo");
/// assert_eq!(config.api_key(), "sk-demo");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibleConfig {
    api_key: String,
    base_url: Option<String>,
    organization: Option<String>,
    default_model: Option<String>,
    timeout_secs: Option<u64>,
    extra_headers: Option<HashMap<String, String>>,
}

impl CompatibleConfig {
    /// 创建兼容协议配置。
    ///
    /// # Arguments
    /// * `api_key` - 兼容协议服务的 API Key
    ///
    /// # Returns
    /// 仅包含 `api_key` 的基础配置对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::CompatibleConfig;
    ///
    /// let config = CompatibleConfig::new("sk-compatible");
    /// assert_eq!(config.api_key(), "sk-compatible");
    /// ```
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            organization: None,
            default_model: None,
            timeout_secs: None,
            extra_headers: None,
        }
    }

    /// 返回 API Key。
    ///
    /// # Returns
    /// 当前配置中的 API Key。
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// 返回基础地址覆盖项。
    ///
    /// # Returns
    /// 若调用方显式设置了 `base_url`，则返回该值。
    #[must_use]
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// 返回组织标识。
    ///
    /// # Returns
    /// 若设置了 `organization`，则返回该值。
    #[must_use]
    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    /// 返回默认模型名称。
    ///
    /// # Returns
    /// 若设置了默认模型，则返回该值。
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// 返回超时时间秒数。
    ///
    /// # Returns
    /// 若设置了超时时间，则返回秒数。
    #[must_use]
    pub const fn timeout_secs(&self) -> Option<u64> {
        self.timeout_secs
    }

    /// 返回额外请求头集合。
    ///
    /// # Returns
    /// 若设置了额外请求头，则返回只读映射。
    #[must_use]
    pub fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        self.extra_headers.as_ref()
    }
}

/// 统一的 Provider 配置枚举。
///
/// 该枚举为后续 `client` 层提供统一的配置读取入口，同时保留各 Provider 的独立配置结构。
///
/// # 示例
/// ```rust
/// use ufox_llm::{OpenAiConfig, ProviderConfig};
///
/// let config = ProviderConfig::OpenAi(OpenAiConfig::new("sk-demo"));
/// assert!(matches!(config, ProviderConfig::OpenAi(_)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderConfig {
    /// `OpenAI` 配置。
    OpenAi(OpenAiConfig),
    /// `Qwen` 配置。
    Qwen(QwenConfig),
    /// 兼容 `OpenAI` 协议的配置。
    Compatible(CompatibleConfig),
}

impl ProviderConfig {
    /// 返回 Provider 类型。
    ///
    /// # Returns
    /// 当前配置所对应的 [`Provider`]。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{OpenAiConfig, Provider, ProviderConfig};
    ///
    /// let config = ProviderConfig::OpenAi(OpenAiConfig::new("sk-demo"));
    /// assert_eq!(config.provider(), Provider::OpenAI);
    /// ```
    #[must_use]
    pub const fn provider(&self) -> Provider {
        match self {
            Self::OpenAi(_) => Provider::OpenAI,
            Self::Qwen(_) => Provider::Qwen,
            Self::Compatible(_) => Provider::Compatible,
        }
    }
}

/// 已解析的客户端配置。
///
/// 该结构体是 `ClientBuilder` 的最终产物，会在后续 `Client` 主体实现中作为运行时
/// 请求发送与协议适配的基础配置载体。
///
/// # 示例
/// ```rust
/// use ufox_llm::{Client, Provider};
///
/// let client = Client::builder()
///     .provider(Provider::OpenAI)
///     .api_key("sk-demo")
///     .model("gpt-4o")
///     .build()
///     .expect("应构建成功");
///
/// assert_eq!(client.config().provider(), Provider::OpenAI);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientConfig {
    provider: Provider,
    provider_config: ProviderConfig,
}

impl ClientConfig {
    /// 返回 Provider 类型。
    ///
    /// # Returns
    /// 当前客户端配置对应的 [`Provider`]。
    #[must_use]
    pub const fn provider(&self) -> Provider {
        self.provider
    }

    /// 返回统一 Provider 配置。
    ///
    /// # Returns
    /// 当前客户端配置中保存的 Provider 专属配置。
    #[must_use]
    pub const fn provider_config(&self) -> &ProviderConfig {
        &self.provider_config
    }
}

/// 带 typestate 的客户端构建器。
///
/// 该构建器通过两个状态参数表达是否已设置 `provider` 和 `api_key`。只有当二者都
/// 已就绪时，`build()` 方法才会出现在类型系统中。
///
/// # 示例
/// ```rust
/// use ufox_llm::{Client, Provider};
///
/// let client = Client::builder()
///     .provider(Provider::Qwen)
///     .api_key("sk-demo")
///     .model("qwen-max")
///     .extra_header("X-Trace-Id", "demo-request")
///     .build()
///     .expect("应构建成功");
///
/// assert_eq!(client.config().provider(), Provider::Qwen);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientBuilder<ProviderState = ProviderUnset, ApiKeyState = ApiKeyUnset> {
    provider: Option<Provider>,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    default_model: Option<String>,
    timeout_secs: Option<u64>,
    extra_headers: HashMap<String, String>,
    _provider_state: PhantomData<ProviderState>,
    _api_key_state: PhantomData<ApiKeyState>,
}

impl ClientBuilder<ProviderUnset, ApiKeyUnset> {
    /// 创建空的客户端构建器。
    ///
    /// # Returns
    /// 尚未设置 `provider` 与 `api_key` 的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new();
    /// assert_eq!(format!("{builder:?}").contains("provider: None"), true);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            api_key: None,
            base_url: None,
            organization: None,
            default_model: None,
            timeout_secs: None,
            extra_headers: HashMap::new(),
            _provider_state: PhantomData,
            _api_key_state: PhantomData,
        }
    }
}

impl Default for ClientBuilder<ProviderUnset, ApiKeyUnset> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, A> ClientBuilder<P, A> {
    /// 设置目标 Provider。
    ///
    /// # Arguments
    /// * `provider` - 目标供应商类型
    ///
    /// # Returns
    /// 已进入“Provider 已设置”状态的新构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{ClientBuilder, Provider};
    ///
    /// let builder = ClientBuilder::new().provider(Provider::OpenAI);
    /// assert_eq!(builder.configured_provider(), Some(Provider::OpenAI));
    /// ```
    #[must_use]
    pub fn provider(self, provider: Provider) -> ClientBuilder<ProviderSet, A> {
        ClientBuilder {
            provider: Some(provider),
            api_key: self.api_key,
            base_url: self.base_url,
            organization: self.organization,
            default_model: self.default_model,
            timeout_secs: self.timeout_secs,
            extra_headers: self.extra_headers,
            _provider_state: PhantomData,
            _api_key_state: PhantomData,
        }
    }

    /// 设置 API Key。
    ///
    /// # Arguments
    /// * `api_key` - Provider API Key
    ///
    /// # Returns
    /// 已进入“API Key 已设置”状态的新构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().api_key("sk-demo");
    /// assert_eq!(builder.configured_api_key(), Some("sk-demo"));
    /// ```
    #[must_use]
    pub fn api_key(self, api_key: impl Into<String>) -> ClientBuilder<P, ApiKeySet> {
        ClientBuilder {
            provider: self.provider,
            api_key: Some(api_key.into()),
            base_url: self.base_url,
            organization: self.organization,
            default_model: self.default_model,
            timeout_secs: self.timeout_secs,
            extra_headers: self.extra_headers,
            _provider_state: PhantomData,
            _api_key_state: PhantomData,
        }
    }

    /// 设置基础地址覆盖项。
    ///
    /// # Arguments
    /// * `base_url` - 覆盖 Provider 默认地址的基础 URL
    ///
    /// # Returns
    /// 设置基础地址后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().base_url("https://api.deepseek.com/v1");
    /// assert_eq!(builder.configured_base_url(), Some("https://api.deepseek.com/v1"));
    /// ```
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// 设置组织标识。
    ///
    /// # Arguments
    /// * `organization` - 组织标识
    ///
    /// # Returns
    /// 设置组织标识后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().organization("org-demo");
    /// assert_eq!(builder.configured_organization(), Some("org-demo"));
    /// ```
    #[must_use]
    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// 设置默认模型名称。
    ///
    /// # Arguments
    /// * `model` - 默认模型名称
    ///
    /// # Returns
    /// 设置模型后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().model("gpt-4o");
    /// assert_eq!(builder.configured_model(), Some("gpt-4o"));
    /// ```
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// 设置请求超时时间。
    ///
    /// 内部最终以秒存储。之所以对带小数部分的 `Duration` 做向上取整，是为了避免调用方
    /// 传入 `500ms` 这类值时被静默截断为 `0s`，从而得到与预期不符的超时配置。
    ///
    /// # Arguments
    /// * `timeout` - 超时时长
    ///
    /// # Returns
    /// 设置超时后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use std::time::Duration;
    ///
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().timeout(Duration::from_millis(500));
    /// assert_eq!(builder.timeout_secs(), Some(1));
    /// ```
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout_secs = Some(duration_to_timeout_secs(timeout));
        self
    }

    /// 添加一个额外请求头。
    ///
    /// # Arguments
    /// * `key` - 请求头名称
    /// * `value` - 请求头值
    ///
    /// # Returns
    /// 添加请求头后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ClientBuilder;
    ///
    /// let builder = ClientBuilder::new().extra_header("X-Trace-Id", "demo-request");
    /// assert_eq!(
    ///     builder.extra_headers().and_then(|headers| headers.get("X-Trace-Id")).map(String::as_str),
    ///     Some("demo-request")
    /// );
    /// ```
    #[must_use]
    pub fn extra_header(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_headers.insert(key.into(), value.into());
        self
    }

    /// 返回当前已选择的 Provider。
    ///
    /// # Returns
    /// 若已设置 Provider，则返回对应值。
    #[must_use]
    pub const fn configured_provider(&self) -> Option<Provider> {
        self.provider
    }

    /// 返回当前 API Key。
    ///
    /// # Returns
    /// 若已设置 API Key，则返回对应值。
    #[must_use]
    pub fn configured_api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// 返回基础地址覆盖项。
    ///
    /// # Returns
    /// 若已设置 `base_url`，则返回对应值。
    #[must_use]
    pub fn configured_base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// 返回组织标识。
    ///
    /// # Returns
    /// 若已设置 `organization`，则返回对应值。
    #[must_use]
    pub fn configured_organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    /// 返回默认模型名称。
    ///
    /// # Returns
    /// 若已设置模型，则返回对应值。
    #[must_use]
    pub fn configured_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// 返回超时时间秒数。
    ///
    /// # Returns
    /// 若已设置超时，则返回秒数。
    #[must_use]
    pub const fn timeout_secs(&self) -> Option<u64> {
        self.timeout_secs
    }

    /// 返回额外请求头集合。
    ///
    /// # Returns
    /// 若已设置额外请求头，则返回只读映射。
    #[must_use]
    pub fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        (!self.extra_headers.is_empty()).then_some(&self.extra_headers)
    }
}

impl ClientBuilder<ProviderSet, ApiKeySet> {
    /// 构建客户端实例。
    ///
    /// 该方法仅在 `provider` 与 `api_key` 都已设置时可用，因此缺少关键配置的情况会在
    /// 编译期被拒绝，而不是留到运行时。
    ///
    /// # Returns
    /// 构建完成的客户端实例。
    ///
    /// # 示例
    /// ```rust
    /// use std::time::Duration;
    ///
    /// use ufox_llm::{Client, Provider};
    ///
    /// let client = Client::builder()
    ///     .provider(Provider::Compatible)
    ///     .base_url("https://api.deepseek.com/v1")
    ///     .api_key("sk-demo")
    ///     .model("deepseek-chat")
    ///     .timeout(Duration::from_secs(30))
    ///     .build()
    ///     .expect("应构建成功");
    ///
    /// assert_eq!(client.config().provider(), Provider::Compatible);
    /// ```
    pub fn build(self) -> Result<Client, crate::LlmError> {
        let provider = self
            .provider
            .expect("ProviderSet 状态必须保证 provider 已存在");
        let api_key = self
            .api_key
            .expect("ApiKeySet 状态必须保证 api_key 已存在");
        let extra_headers = (!self.extra_headers.is_empty()).then_some(self.extra_headers);

        let provider_config = match provider {
            Provider::OpenAI => ProviderConfig::OpenAi(OpenAiConfig {
                api_key,
                base_url: self.base_url,
                organization: self.organization,
                default_model: self.default_model,
                timeout_secs: self.timeout_secs,
                extra_headers,
            }),
            Provider::Qwen => ProviderConfig::Qwen(QwenConfig {
                api_key,
                base_url: self.base_url,
                organization: self.organization,
                default_model: self.default_model,
                timeout_secs: self.timeout_secs,
                extra_headers,
            }),
            Provider::Compatible => ProviderConfig::Compatible(CompatibleConfig {
                api_key,
                base_url: self.base_url,
                organization: self.organization,
                default_model: self.default_model,
                timeout_secs: self.timeout_secs,
                extra_headers,
            }),
        };

        Ok(Client::from_config(ClientConfig {
            provider,
            provider_config,
        }))
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

    use super::{ApiKeySet, ClientBuilder, ProviderSet, ProviderUnset};
    use crate::Provider;

    #[test]
    fn typestate() {
        let builder: ClientBuilder<ProviderUnset, _> = ClientBuilder::new();
        let builder: ClientBuilder<ProviderSet, ApiKeySet> =
            builder.provider(Provider::OpenAI).api_key("sk-demo");

        assert_eq!(builder.configured_provider(), Some(Provider::OpenAI));
        assert_eq!(builder.configured_api_key(), Some("sk-demo"));
    }

    #[test]
    fn build_provider() {
        let client = ClientBuilder::new()
            .provider(Provider::Qwen)
            .api_key("sk-qwen")
            .model("qwen-max")
            .extra_header("X-Trace-Id", "demo-request")
            .build()
            .expect("应构建成功");

        assert_eq!(client.config().provider(), Provider::Qwen);

        let crate::ProviderConfig::Qwen(config) = client.config().provider_config() else {
            panic!("预期为 Qwen 配置");
        };

        assert_eq!(config.api_key(), "sk-qwen");
        assert_eq!(config.default_model(), Some("qwen-max"));
        assert_eq!(
            config
                .extra_headers()
                .and_then(|headers| headers.get("X-Trace-Id"))
                .map(String::as_str),
            Some("demo-request")
        );
    }

    #[test]
    fn builder_test() {
        let client = ClientBuilder::new()
            .provider(Provider::OpenAI)
            .api_key("sk-openai")
            .timeout(Duration::from_millis(500))
            .build()
            .expect("应构建成功");

        let crate::ProviderConfig::OpenAi(config) = client.config().provider_config() else {
            panic!("预期为 OpenAI 配置");
        };

        assert_eq!(config.timeout_secs(), Some(1));
    }
}
