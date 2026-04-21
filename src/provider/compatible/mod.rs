//! Compatible 适配器。
//!
//! 复用 OpenAI-compatible 协议实现并暴露兼容适配器入口。

use std::sync::Mutex;

use serde_json::Value;

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool,
    client::RequestOptions,
};

pub mod stream;

pub use stream::{CompatibleStreamParser, is_done_event};

/// 兼容 `OpenAI` 协议的适配器。
#[derive(Debug, Default)]
pub struct CompatibleAdapter {
    stream_parser: Mutex<CompatibleStreamParser>,
}

impl CompatibleAdapter {
    /// 创建兼容 `OpenAI` 协议的适配器。
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
    fn adapter_keeps_provider_options_in_openai_compatible_body() {
        let adapter = CompatibleAdapter::new();
        let request = adapter
            .build_chat_request(
                "deepseek-chat",
                &[Message::user("你好")],
                None,
                false,
                &RequestOptions {
                    provider_options: serde_json::Map::from_iter([
                        ("seed".to_string(), json!(42)),
                        ("temperature".to_string(), json!(0.1)),
                    ]),
                    temperature: Some(0.6),
                    ..RequestOptions::default()
                },
            )
            .expect("请求体应构建成功");

        assert_eq!(request["seed"], 42);
        assert!(
            (request["temperature"]
                .as_f64()
                .expect("temperature 应为数字")
                - 0.6)
                .abs()
                < 1e-6
        );
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
