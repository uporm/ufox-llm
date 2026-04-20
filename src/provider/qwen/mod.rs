//! `Qwen` 协议适配模块。
//!
//! 该模块负责聚合 `DashScope` / `Qwen` 协议的请求序列化、非流式响应解析、
//! 流式事件解析，并对外提供统一的 [`QwenAdapter`] 入口。
//!
//! 设计上采用“子模块拆分 + 适配器聚合”的方式：
//! 1. `request` 专注把公共类型序列化为 `Qwen` 私有请求体；
//! 2. `response` 专注把完整响应体解析为公共 [`ChatResponse`]；
//! 3. `stream` 专注把单条 `SSE` 事件解析为公共 [`StreamChunk`]，并在内部处理
//!    工具调用碎片的跨事件聚合；
//! 4. `QwenAdapter` 则负责将这些能力组合为 [`ProviderAdapter`] 的统一实现。
//!
//! 该模块依赖上级 `provider` 抽象，并复用 `types` 模块中的消息、工具与响应类型。

use std::sync::Mutex;

use serde_json::Value;

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool,
    client::RequestOptions,
};

pub mod request;
pub mod response;
pub mod stream;

pub use stream::{QwenStreamParser, is_done_event};

/// `Qwen` 协议适配器。
///
/// 该适配器将 `Qwen` 私有协议封装为统一的 [`ProviderAdapter`] 接口。
/// 对于流式请求，它内部持有一个带锁的 [`QwenStreamParser`]，用于在 trait 的
/// `&self` 方法签名下继续安全地维护跨事件解析状态。
///
/// # 示例
/// ```rust
/// use ufox_llm::{Provider, ProviderAdapter};
/// use ufox_llm::provider::qwen::QwenAdapter;
///
/// let adapter = QwenAdapter::new();
/// assert_eq!(adapter.provider(), Provider::Qwen);
/// ```
#[derive(Debug, Default)]
pub struct QwenAdapter {
    stream_parser: Mutex<QwenStreamParser>,
}

impl QwenAdapter {
    /// 创建 `Qwen` 协议适配器。
    ///
    /// # Returns
    /// 可用于构建请求体和解析响应的 `Qwen` 适配器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::provider::qwen::QwenAdapter;
    /// use ufox_llm::{Provider, ProviderAdapter};
    ///
    /// let adapter = QwenAdapter::new();
    /// assert_eq!(adapter.provider(), Provider::Qwen);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_stream_parser(&self) -> Result<std::sync::MutexGuard<'_, QwenStreamParser>, LlmError> {
        self.stream_parser.lock().map_err(|_| {
            LlmError::StreamError("Qwen 流式解析器状态已损坏，无法继续解析".to_string())
        })
    }
}

impl ProviderAdapter for QwenAdapter {
    fn provider(&self) -> Provider {
        Provider::Qwen
    }

    fn chat_path(&self) -> &'static str {
        "/services/aigc/text-generation/generation"
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

        request::build_chat_request(model, messages, tools, stream, options)
    }

    fn parse_chat_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError> {
        response::parse_chat_response(body)
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

    use super::QwenAdapter;
    use crate::{Message, Provider, ProviderAdapter, client::RequestOptions};

    #[test]
    fn adapter_exposes_qwen_provider_info() {
        let adapter = QwenAdapter::new();

        assert_eq!(adapter.provider(), Provider::Qwen);
        assert_eq!(adapter.chat_path(), "/services/aigc/text-generation/generation");
        assert_eq!(
            adapter.default_base_url(),
            Some("https://dashscope.aliyuncs.com")
        );
    }

    #[test]
    fn adapter() {
        let adapter = QwenAdapter::new();
        let request = adapter
            .build_chat_request(
                "qwen-max",
                &[Message::user("你好")],
                None,
                true,
                &RequestOptions::default(),
            )
            .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["incremental_output"], true);

        let chunk = adapter
            .parse_stream_chunk(
                &json!({
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": "你"
                                },
                                "finish_reason": null
                            }
                        ]
                    }
                })
                .to_string(),
            )
            .expect("流式事件应解析成功")
            .expect("应产出流式增量");

        assert_eq!(chunk.delta(), "你");
    }
}
