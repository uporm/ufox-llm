//! 兼容 `OpenAI` 协议的供应商模块。
//!
//! 该模块负责聚合与 `OpenAI Chat Completions` 协议兼容的第三方服务接入能力，
//! 并对外提供统一的 [`CompatibleAdapter`] 入口。
//!
//! 设计上采用“协议实现完全复用、接入配置独立声明”的方式：
//! 1. 请求序列化直接复用 `openai::request`；
//! 2. 非流式响应解析直接复用 `openai::response`；
//! 3. 流式事件解析通过本目录下的 `stream` 模块复用 `OpenAI` 解析器；
//! 4. 适配器层只负责声明自己属于 [`Provider::Compatible`]，并保持 `base_url`
//!    由上层调用方显式提供。
//!
//! 这种设计可以让 `DeepSeek`、`Ollama`、自建网关等兼容接口直接共享协议实现，
//! 同时不把它们误标识为官方 `OpenAI` Provider。

use std::sync::Mutex;

use serde_json::Value;

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool,
    client::RequestOptions,
};

pub mod stream;

pub use stream::{CompatibleStreamParser, is_done_event};

/// 兼容 `OpenAI` 协议的适配器。
///
/// 该适配器面向所有请求体与响应体遵循 `OpenAI Chat Completions` 语义，但基础地址
/// 由调用方自行指定的服务。
///
/// # 示例
/// ```rust
/// use ufox_llm::{Provider, ProviderAdapter};
/// use ufox_llm::provider::compatible::CompatibleAdapter;
///
/// let adapter = CompatibleAdapter::new();
/// assert_eq!(adapter.provider(), Provider::Compatible);
/// ```
#[derive(Debug, Default)]
pub struct CompatibleAdapter {
    stream_parser: Mutex<CompatibleStreamParser>,
}

impl CompatibleAdapter {
    /// 创建兼容 `OpenAI` 协议的适配器。
    ///
    /// # Returns
    /// 可用于构建请求体和解析响应的兼容协议适配器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::compatible::CompatibleAdapter;
    /// use ufox_llm::{Provider, ProviderAdapter};
    ///
    /// let adapter = CompatibleAdapter::new();
    /// assert_eq!(adapter.provider(), Provider::Compatible);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_stream_parser(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, CompatibleStreamParser>, LlmError> {
        self.stream_parser.lock().map_err(|_| {
            LlmError::StreamError("Compatible 流式解析器状态已损坏，无法继续解析".to_string())
        })
    }
}

impl ProviderAdapter for CompatibleAdapter {
    fn provider(&self) -> Provider {
        Provider::Compatible
    }

    fn chat_path(&self) -> &'static str {
        "/chat/completions"
    }

    fn build_chat_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
        stream: bool,
        options: &RequestOptions,
    ) -> Result<Value, LlmError> {
        if stream {
            self.lock_stream_parser()?.reset();
        }

        crate::provider::openai::request::build_chat_request(
            model, messages, tools, stream, options,
        )
    }

    fn parse_chat_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError> {
        crate::provider::openai::response::parse_chat_response_with_provider(body, "Compatible")
    }

    fn parse_stream_chunk(&self, event_data: &str) -> Result<Option<StreamChunk>, LlmError> {
        self.lock_stream_parser()?.parse_event(event_data)
    }

    fn parse_stream_chunks(&self, event_data: &str) -> Result<Vec<StreamChunk>, LlmError> {
        self.lock_stream_parser()?.parse_event_chunks(event_data)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::CompatibleAdapter;
    use crate::{LlmError, Message, Provider, ProviderAdapter, client::RequestOptions};

    #[test]
    fn adapter_exposes_compatible_provider_info() {
        let adapter = CompatibleAdapter::new();

        assert_eq!(adapter.provider(), Provider::Compatible);
        assert_eq!(adapter.chat_path(), "/chat/completions");
        assert_eq!(adapter.default_base_url(), None);
        assert_eq!(adapter.provider_name(), "compatible");
    }

    #[test]
    fn adapter() {
        let adapter = CompatibleAdapter::new();
        let request = adapter
            .build_chat_request(
                "deepseek-chat",
                &[Message::user("你好")],
                None,
                true,
                &RequestOptions::default(),
            )
            .expect("请求体应构建成功");

        assert_eq!(request["stream"], true);

        let chunk = adapter
            .parse_stream_chunk(
                &json!({
                    "choices": [
                        {
                            "delta": {
                                "content": "你"
                            },
                            "finish_reason": null
                        }
                    ]
                })
                .to_string(),
            )
            .expect("流式事件应解析成功")
            .expect("应产出流式增量");

        assert_eq!(chunk.delta(), "你");
    }

    #[test]
    fn adapter_openai() {
        let adapter = CompatibleAdapter::new();
        let error = adapter
            .parse_chat_response(r#"{"error":{"message":"上游服务失败"}}"#.as_bytes())
            .expect_err("应返回错误");

        match error {
            LlmError::ApiError { provider, .. } => assert_eq!(provider, "Compatible"),
            other => panic!("错误类型不符合预期：{other:?}"),
        }
    }
}
