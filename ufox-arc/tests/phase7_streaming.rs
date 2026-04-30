/// Phase 7 流式集成测试：验证 chat_stream() 走完整推理循环。
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use ufox_arc::tools::{Tool, ToolError, ToolMetadata};
use ufox_arc::{Agent, ArcError, ExecutionState};
use ufox_llm::{Provider, ToolResultPayload};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── helpers ──────────────────────────────────────────────────────────────────

/// SSE 正文：纯文字响应（finish_reason: stop）。
fn sse_text(content: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{content_json}}},\"finish_reason\":null}}]}}\n\n\
data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}}}\n\n\
data: [DONE]\n\n",
        content_json = serde_json::to_string(content).unwrap()
    )
}

/// SSE 正文：触发一次工具调用后结束（arguments 分两帧）。
fn sse_tool_call(tool_name: &str, args: &str) -> String {
    // 首帧：带 id/type/name，arguments 为空串
    // 次帧：追加完整 arguments + finish_reason + usage
    let mid = args.len() / 2;
    let (args1, args2) = args.split_at(mid);
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_001\",\"type\":\"function\",\"function\":{{\"name\":\"{tool_name}\",\"arguments\":\"\"}}}}]}},\"finish_reason\":null}}]}}\n\n\
data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{a1}}}}}]}},\"finish_reason\":null}}]}}\n\n\
data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{a2}}}}}]}},\"finish_reason\":\"tool_calls\"}}],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":10,\"total_tokens\":30}}}}\n\n\
data: [DONE]\n\n",
        a1 = serde_json::to_string(args1).unwrap(),
        a2 = serde_json::to_string(args2).unwrap(),
    )
}

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .append_header("content-type", "text/event-stream")
        .set_body_string(body.to_string())
}

// ─── echo 工具 ────────────────────────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn metadata(&self) -> &ToolMetadata {
        static META: std::sync::OnceLock<ToolMetadata> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolMetadata {
            name: "echo".into(),
            description: "Echo the input text.".into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
            requires_confirmation: false,
            timeout: Duration::from_secs(5),
        })
    }

    async fn execute(&self, params: Value) -> Result<ToolResultPayload, ToolError> {
        let text = params["text"].as_str().unwrap_or("").to_string();
        Ok(ToolResultPayload::text(format!("echo: {text}")))
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

/// 无工具的流式对话：chunk 正确发出，最后收到 Completed 事件。
#[tokio::test]
async fn stream_simple_yields_chunks_and_completed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(&sse_text("你好世界")))
        .mount(&server)
        .await;

    let agent = Agent::builder()
        .llm(
            ufox_llm::Client::builder()
                .provider(Provider::Compatible)
                .api_key("test-key")
                .base_url(server.uri())
                .model("test-model")
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let mut session = agent.session("u1", "s1").await.unwrap();
    let mut stream = session.chat_stream("hi").await.unwrap();

    let mut text_chunks = 0u32;
    let mut completed = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if event.chunk.is_some() {
            text_chunks += 1;
        }
        if matches!(event.state_change, Some(ExecutionState::Completed)) {
            completed = true;
        }
    }

    assert!(text_chunks > 0, "should receive at least one text chunk");
    assert!(completed, "should receive Completed event");
}

/// 流式模式下工具调用循环：Act step 事件正确发出，最终收到 Completed。
#[tokio::test]
async fn stream_tool_call_emits_act_step_and_completes() {
    let server = MockServer::start().await;

    // 第一次请求：SSE 工具调用
    let args = r#"{"text":"hello"}"#;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(&sse_tool_call("echo", args)))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // 第二次请求：SSE 最终文本
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(&sse_text("工具已执行完毕。")))
        .mount(&server)
        .await;

    let agent = Agent::builder()
        .llm(
            ufox_llm::Client::builder()
                .provider(Provider::Compatible)
                .api_key("test-key")
                .base_url(server.uri())
                .model("test-model")
                .build()
                .unwrap(),
        )
        .tool(EchoTool)
        .build()
        .unwrap();

    let mut session = agent.session("u1", "stream-tool").await.unwrap();
    let mut stream = session.chat_stream("call echo").await.unwrap();

    let mut saw_act_step = false;
    let mut completed = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if let Some(step) = &event.step {
            if matches!(step.kind, ufox_arc::StepKind::Act) {
                saw_act_step = true;
            }
        }
        if matches!(event.state_change, Some(ExecutionState::Completed)) {
            completed = true;
        }
    }

    assert!(saw_act_step, "should receive an Act step event");
    assert!(completed, "should receive Completed event");
}

/// 流式模式 SessionBusy：第二个并发请求立即返回错误。
#[tokio::test]
async fn stream_session_busy_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(&sse_text("slow")).set_delay(Duration::from_millis(200)))
        .mount(&server)
        .await;

    let agent = Agent::builder()
        .llm(
            ufox_llm::Client::builder()
                .provider(Provider::Compatible)
                .api_key("test-key")
                .base_url(server.uri())
                .model("test-model")
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let mut s1 = agent.session("u1", "busy-stream").await.unwrap();
    let mut s2 = s1.clone();

    let _stream1 = s1.chat_stream("first").await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let result = s2.chat_stream("second").await;
    assert!(matches!(result, Err(ArcError::SessionBusy)));
}
