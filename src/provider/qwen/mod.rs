use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{
    error::LlmError,
    middleware::Transport,
    types::{request::ChatRequest, response::ChatChunk, response::ChatResponse},
};

use super::{ProviderAdapter, ApiProtocol, openai};

/// 通义千问适配器。
struct QwenAdapter {
    inner: Box<dyn ProviderAdapter>,
}

impl QwenAdapter {
    fn new(inner: Box<dyn ProviderAdapter>) -> Self {
        Self { inner }
    }

    /// 将通用请求重写为 Qwen 兼容参数，避免把 provider 差异泄漏到共享协议层。
    fn rewrite_chat_request(&self, mut req: ChatRequest) -> ChatRequest {
        if !req.extensions.contains_key("enable_thinking") {
            req.extensions
                .insert("enable_thinking".into(), req.thinking.into());
        }
        if let Some(thinking_budget) = req.thinking_budget
            && !req.extensions.contains_key("thinking_budget")
        {
            req.extensions
                .insert("thinking_budget".into(), thinking_budget.into());
        }

        // 清空通用字段，避免共享 OpenAI 兼容层继续产出不被 Qwen 接受的 `thinking`。
        req.thinking = false;
        req.thinking_budget = None;
        req
    }
}

#[async_trait]
impl ProviderAdapter for QwenAdapter {
    fn name(&self) -> &'static str {
        "qwen"
    }

    async fn chat(&self, model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner.chat(model, self.rewrite_chat_request(req)).await
    }

    async fn chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError> {
        self.inner
            .chat_stream(model, self.rewrite_chat_request(req))
            .await
    }
}

/// 构造通义千问 provider adapter。
///
/// 当前先复用 DashScope 的 OpenAI 兼容 chat 协议，仅开放 `chat` 与 `chat_stream`，
/// 避免把尚未逐项验证的兼容能力一并声明为已支持。
pub(crate) fn build(
    api_key: &str,
    base_url: &str,
    transport: &Transport,
) -> Result<Box<dyn ProviderAdapter>, LlmError> {
    let compatible_chat = openai::build("qwen", ApiProtocol::ChatCompletions, api_key, base_url, transport)?;
    Ok(Box::new(QwenAdapter::new(compatible_chat)))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::{
        middleware::{RetryConfig, Transport, TransportConfig},
        types::{
            request::{ChatRequest, EmbeddingRequest},
            response::ChatResponse,
        },
    };

    use super::*;
    use futures::stream;

    fn test_transport() -> Transport {
        Transport::new(TransportConfig {
            timeout_secs: 600,
            connect_timeout_secs: 5,
            read_timeout_secs: 60,
            retry: RetryConfig::default(),
            rate_limit: None,
        })
    }

    struct CapturingAdapter {
        seen: Arc<Mutex<Option<ChatRequest>>>,
    }

    #[async_trait]
    impl ProviderAdapter for CapturingAdapter {
        fn name(&self) -> &'static str {
            "capturing"
        }

        async fn chat(&self, _model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError> {
            *self.seen.lock().unwrap() = Some(req);
            Ok(ChatResponse {
                id: String::new(),
                model: String::new(),
                text: String::new(),
                thinking: None,
                tool_calls: Vec::new(),
                finish_reason: None,
                usage: None,
                raw: None,
            })
        }

        async fn chat_stream(
            &self,
            _model: &str,
            req: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError> {
            *self.seen.lock().unwrap() = Some(req);
            Ok(Box::pin(stream::empty()))
        }
    }

    #[tokio::test]
    async fn qwen_adapter_only_enables_chat_apis() {
        let adapter = build("test-key", "https://example.com/v1", &test_transport()).unwrap();

        assert_eq!(adapter.name(), "qwen");
        assert!(matches!(
            adapter
                .embed(
                    "text-embedding-v1",
                    EmbeddingRequest {
                        inputs: vec!["hello".into()],
                        dimensions: None,
                        extensions: serde_json::Map::new(),
                    },
                )
                .await
                .unwrap_err(),
            LlmError::UnsupportedCapability {
                provider: Some(ref provider),
                ref capability,
            } if provider == "qwen" && capability == "embed"
        ));
    }

    #[tokio::test]
    async fn qwen_adapter_rewrites_thinking_controls_before_forwarding() {
        let seen = Arc::new(Mutex::new(None));
        let adapter = QwenAdapter::new(Box::new(CapturingAdapter { seen: seen.clone() }));

        adapter
            .chat(
                "qwen-plus",
                ChatRequest::builder()
                    .user_text("hello")
                    .thinking_budget(2048)
                    .build(),
            )
            .await
            .unwrap();

        let forwarded = seen.lock().unwrap().clone().unwrap();
        assert!(!forwarded.thinking);
        assert_eq!(forwarded.thinking_budget, None);
        assert_eq!(
            forwarded.extensions.get("enable_thinking"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            forwarded.extensions.get("thinking_budget"),
            Some(&serde_json::json!(2048))
        );
    }
}
