//! Qwen 适配器。
//!
//! 组合 Qwen 协议下的请求构建、响应解析与流式解析能力。

use std::sync::Mutex;

use serde_json::Value;

use crate::{
    ChatResponse, LlmError, Message, Provider, ProviderAdapter, StreamChunk, Tool,
    client::RequestOptions,
};

mod request;
mod response;
mod stream;

pub use stream::QwenStreamParser;

/// `Qwen` 协议适配器。
#[derive(Debug, Default)]
pub struct QwenAdapter {
    stream_parser: Mutex<QwenStreamParser>,
}

impl QwenAdapter {
    /// 创建 `Qwen` 协议适配器。
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
        assert_eq!(adapter.chat_path(), "/chat/completions");
        assert_eq!(
            adapter.default_base_url(),
            Some("https://dashscope-intl.aliyuncs.com/api/v1")
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

        assert_eq!(chunk.delta, "你");
    }
}
