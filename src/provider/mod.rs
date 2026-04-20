//! Provider 抽象模块。
//!
//! 该模块负责定义 SDK 对不同 `LLM Provider` 的统一抽象边界，包括：
//! 1. [`Provider`]：用于表达当前选择的供应商类型；
//! 2. [`ProviderAdapter`]：用于约束不同供应商在请求构建、响应解析与流式解析上的公共能力。
//!
//! 设计上将“网络发送”与“协议适配”分离：
//! 1. `client` 层负责管理 `HTTP` 客户端、超时、鉴权头与实际请求发送；
//! 2. `provider` 层只负责把公共类型转换为各家私有协议格式，并把私有协议再解析回公共类型；
//! 3. 这种分层可以让 `Compatible` 供应商复用 `OpenAI` 协议逻辑，同时只覆盖默认地址等少量差异。
//!
//! 该模块依赖 `types` 模块中的消息、工具与响应类型，并依赖 `serde_json::Value`
//! 作为协议无关的中间请求体表示。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatResponse, LlmError, Message, StreamChunk, Tool, client::RequestOptions};

pub mod openai;
pub mod qwen;
pub mod compatible;

/// `LLM` 供应商类型。
///
/// 该枚举用于描述当前客户端选择的供应商族别。后续 `builder` 会基于该值选择默认端点、
/// 构建对应的适配器实现，并决定是否允许某些供应商特有配置。
///
/// # 示例
/// ```rust
/// use ufox_llm::Provider;
///
/// assert_eq!(Provider::OpenAI.as_str(), "openai");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// `OpenAI` 官方协议。
    OpenAI,
    /// 阿里云 `Qwen` / `DashScope` 协议。
    Qwen,
    /// 与 `OpenAI` `Chat Completions` 协议兼容的第三方服务。
    Compatible,
}

impl Provider {
    /// 返回供应商的稳定字符串表示。
    ///
    /// # Returns
    /// 适合用于日志、错误上下文和配置序列化的小写供应商名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Provider;
    ///
    /// assert_eq!(Provider::Qwen.as_str(), "qwen");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Qwen => "qwen",
            Self::Compatible => "compatible",
        }
    }

    /// 返回供应商的展示名称。
    ///
    /// # Returns
    /// 适合展示给用户或写入错误消息的人类可读名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Provider;
    ///
    /// assert_eq!(Provider::Compatible.display_name(), "Compatible");
    /// ```
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::OpenAI => "OpenAI",
            Self::Qwen => "Qwen",
            Self::Compatible => "Compatible",
        }
    }

    /// 返回供应商默认基础地址。
    ///
    /// `Compatible` 类型不提供默认基础地址，因为这类服务的接入点由调用方自行指定。
    ///
    /// # Returns
    /// 若供应商存在内建默认地址，则返回该地址。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Provider;
    ///
    /// assert_eq!(
    ///     Provider::OpenAI.default_base_url(),
    ///     Some("https://api.openai.com/v1")
    /// );
    /// assert_eq!(Provider::Compatible.default_base_url(), None);
    /// ```
    #[must_use]
    pub const fn default_base_url(self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("https://api.openai.com/v1"),
            Self::Qwen => Some("https://dashscope.aliyuncs.com"),
            Self::Compatible => None,
        }
    }

    /// 返回供应商是否要求调用方显式提供基础地址。
    ///
    /// # Returns
    /// 如果当前供应商没有内建默认地址，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Provider;
    ///
    /// assert!(Provider::Compatible.requires_explicit_base_url());
    /// assert!(!Provider::OpenAI.requires_explicit_base_url());
    /// ```
    #[must_use]
    pub const fn requires_explicit_base_url(self) -> bool {
        self.default_base_url().is_none()
    }
}

/// 供应商协议适配器。
///
/// 该 trait 用于描述不同供应商在“公共类型 <-> 私有协议”之间的双向转换能力。
/// 它不负责发送网络请求，只负责：
/// - 将消息、模型名和工具声明转换为供应商私有请求体；
/// - 将非流式响应体解析为统一的 [`ChatResponse`]；
/// - 将单条流式事件数据解析为统一的 [`StreamChunk`]。
///
/// 这种设计的原因是不同供应商的网络层行为高度相似，但请求和响应格式差异较大。
/// 因此让 `client` 统一处理请求发送、让适配器专注协议转换，可以显著减少重复逻辑。
pub trait ProviderAdapter: Send + Sync {
    /// 返回当前适配器对应的供应商类型。
    ///
    /// # Returns
    /// 当前协议适配器归属的 [`Provider`]。
    fn provider(&self) -> Provider;

    /// 返回适配器对应供应商的稳定名称。
    ///
    /// 默认实现直接复用 [`Provider::as_str`]，这样错误上下文和日志输出保持一致。
    ///
    /// # Returns
    /// 当前供应商的小写稳定名称。
    fn provider_name(&self) -> &'static str {
        self.provider().as_str()
    }

    /// 返回供应商默认基础地址。
    ///
    /// # Returns
    /// 若当前供应商内建默认基础地址，则返回该地址。
    fn default_base_url(&self) -> Option<&'static str> {
        self.provider().default_base_url()
    }

    /// 返回聊天接口相对路径。
    ///
    /// `client` 层会将其与基础地址拼接得到最终请求地址。把路径放在适配器上，
    /// 可以避免 `client` 层为每个供应商硬编码分支。
    ///
    /// # Returns
    /// 聊天请求的相对路径。
    fn chat_path(&self) -> &'static str;

    /// 构建供应商私有请求体。
    ///
    /// # Arguments
    /// * `model` - 调用使用的模型名称
    /// * `messages` - 发送给模型的消息序列
    /// * `tools` - 可选的工具定义列表
    /// * `stream` - 是否启用流式输出
    ///
    /// # Returns
    /// 供应商私有格式的请求体 `JSON`。
    ///
    /// # Errors
    /// - [`LlmError::UnsupportedFeature`]：当当前供应商不支持某项请求能力时触发
    /// - [`LlmError::StreamError`]：当流式请求体需要额外校验且校验失败时触发
    fn build_chat_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
        stream: bool,
        options: &RequestOptions,
    ) -> Result<Value, LlmError>;

    /// 解析非流式响应体。
    ///
    /// # Arguments
    /// * `body` - 原始响应体字节
    ///
    /// # Returns
    /// 统一后的完整聊天响应。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当响应体不是合法 `JSON` 时触发
    /// - [`LlmError::ApiError`]：当响应体内部表示业务失败时触发
    fn parse_chat_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError>;

    /// 解析单条流式事件的数据部分。
    ///
    /// 当流式协议使用类似 `data: [DONE]` 的终止标记时，应返回 `Ok(None)`，
    /// 表示该事件不产生可消费的增量片段。
    ///
    /// # Arguments
    /// * `event_data` - `SSE` 事件中 `data:` 字段对应的原始文本
    ///
    /// # Returns
    /// 成功时返回可选的统一流式增量片段。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件数据格式非法或语义不完整时触发
    fn parse_stream_chunk(&self, event_data: &str) -> Result<Option<StreamChunk>, LlmError>;

    /// 解析单条流式事件，并返回该事件内包含的所有公共增量片段。
    ///
    /// 某些 Provider 可能在同一条 `SSE` 事件中同时返回思考过程与正式回复。默认实现会
    /// 退化为最多返回一个片段；若供应商支持单事件多片段，或需要在同一事件中同时保留
    /// 正文、`finish_reason`、`usage`、`tool_calls` 等尾信息，应覆盖该方法。
    ///
    /// 对 SDK 内置适配器来说，`Client` 主路径始终优先调用该方法；因此新 Provider 若
    /// 存在 mixed-event 语义，建议把真实解析逻辑实现于此，再让 `parse_stream_chunk()`
    /// 退化为兼容包装。
    ///
    /// # Arguments
    /// * `event_data` - `SSE` 事件中 `data:` 字段对应的原始文本
    ///
    /// # Returns
    /// 当前事件对应的零个或多个流式片段。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件数据格式非法或语义不完整时触发
    fn parse_stream_chunks(&self, event_data: &str) -> Result<Vec<StreamChunk>, LlmError> {
        Ok(self.parse_stream_chunk(event_data)?.into_iter().collect())
    }

    /// 返回当前供应商是否支持工具调用。
    ///
    /// # Returns
    /// 如果当前供应商支持函数工具调用，则返回 `true`。
    fn supports_tools(&self) -> bool {
        true
    }

    /// 返回当前供应商是否支持多模态输入。
    ///
    /// # Returns
    /// 如果当前供应商支持图片等多模态输入，则返回 `true`。
    fn supports_multimodal(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Provider;

    #[test]
    fn provider_2() {
        assert_eq!(Provider::OpenAI.display_name(), "OpenAI");
        assert_eq!(
            Provider::OpenAI.default_base_url(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            Provider::Qwen.default_base_url(),
            Some("https://dashscope.aliyuncs.com")
        );
        assert!(Provider::Compatible.requires_explicit_base_url());
    }
}
