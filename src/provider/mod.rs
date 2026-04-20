//! Provider 抽象。
//!
//! 定义 `Provider` 与 `ProviderAdapter`，统一协议构建和响应解析接口。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatResponse, LlmError, Message, StreamChunk, Tool, client::RequestOptions};

pub mod compatible;
pub mod openai;
pub mod qwen;

/// `LLM` 供应商类型。
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
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Qwen => "qwen",
            Self::Compatible => "compatible",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::OpenAI => "OpenAI",
            Self::Qwen => "Qwen",
            Self::Compatible => "Compatible",
        }
    }

    pub const fn default_base_url(self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("https://api.openai.com/v1"),
            Self::Qwen => Some("https://dashscope.aliyuncs.com"),
            Self::Compatible => None,
        }
    }

    pub const fn requires_explicit_base_url(self) -> bool {
        self.default_base_url().is_none()
    }
}

/// 供应商协议适配器。
pub trait ProviderAdapter: Send + Sync {
    fn provider(&self) -> Provider;

    fn provider_name(&self) -> &'static str {
        self.provider().as_str()
    }

    fn default_base_url(&self) -> Option<&'static str> {
        self.provider().default_base_url()
    }

    fn chat_path(&self) -> &'static str;

    /// 构建供应商私有请求体。
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
    /// # Errors
    /// - [`LlmError::ParseError`]：当响应体不是合法 `JSON` 时触发
    /// - [`LlmError::ApiError`]：当响应体内部表示业务失败时触发
    fn parse_chat_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError>;

    /// 解析单条流式事件的数据部分。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件数据格式非法或语义不完整时触发
    fn parse_stream_chunk(&self, event_data: &str) -> Result<Option<StreamChunk>, LlmError>;

    /// 解析单条流式事件，并返回该事件内包含的所有公共增量片段。
    /// # Errors
    /// - [`LlmError::ParseError`]：当事件数据不是合法 `JSON` 时触发
    /// - [`LlmError::StreamError`]：当事件数据格式非法或语义不完整时触发
    fn parse_stream_chunks(&self, event_data: &str) -> Result<Vec<StreamChunk>, LlmError> {
        Ok(self.parse_stream_chunk(event_data)?.into_iter().collect())
    }

    fn supports_tools(&self) -> bool {
        true
    }

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
