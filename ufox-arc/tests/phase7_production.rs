/// Phase 7 集成测试：工具调用循环、HITL、超时、MaxIterations、Session 持久化。
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use ufox_arc::tools::{Tool, ToolError, ToolMetadata};
use ufox_arc::{
    Agent, AgentConfig, ArcError, AutoApproveHandler, ExecutionState, InMemorySessionStore,
    SessionStore, StepKind,
};
use ufox_llm::{Provider, ToolResultPayload};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── helpers ──────────────────────────────────────────────────────────────────

fn make_agent(base_url: &str) -> Agent {
    Agent::builder()
        .llm(
            ufox_llm::Client::builder()
                .provider(Provider::Compatible)
                .api_key("test-key")
                .base_url(base_url)
                .model("test-model")
                .build()
                .unwrap(),
        )
        .system("你是测试助手。")
        .build()
        .unwrap()
}

fn text_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890u64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    })
}

fn tool_call_response(tool_name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1234567890u64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": serde_json::to_string(&args).unwrap()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30 }
    })
}

// ─── 简单 echo 工具 ────────────────────────────────────────────────────────────

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

// ─── 需要确认的工具 ───────────────────────────────────────────────────────────

struct ConfirmedTool;

#[async_trait]
impl Tool for ConfirmedTool {
    fn metadata(&self) -> &ToolMetadata {
        static META: std::sync::OnceLock<ToolMetadata> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolMetadata {
            name: "confirmed_op".into(),
            description: "An operation that requires confirmation.".into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
            requires_confirmation: true,
            timeout: Duration::from_secs(5),
        })
    }

    async fn execute(&self, params: Value) -> Result<ToolResultPayload, ToolError> {
        let value = params["value"].as_str().unwrap_or("").to_string();
        Ok(ToolResultPayload::text(format!("confirmed: {value}")))
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

/// 工具调用完整循环：mock 先返回 tool_calls，再返回最终文本。
#[tokio::test]
async fn tool_call_loop_executes_and_returns_final_response() {
    let server = MockServer::start().await;

    // 第一次响应：触发工具调用
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response(
            "echo",
            serde_json::json!({"text": "hello"}),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // 第二次响应：LLM 收到工具结果后返回最终答复
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(text_response("工具已成功执行，返回：echo: hello")),
        )
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
        .system("你是测试助手。")
        .tool(EchoTool)
        .build()
        .unwrap();

    let mut session = agent.session("user1", "tool-loop").await.unwrap();
    let result = session.chat("请调用 echo 工具").await.unwrap();

    assert!(matches!(result.trace.state, ExecutionState::Completed));
    let kinds: Vec<_> = result.trace.steps.iter().map(|s| &s.kind).collect();
    assert!(kinds.iter().any(|k| matches!(k, StepKind::Think)));
    assert!(kinds.iter().any(|k| matches!(k, StepKind::Act)));
    assert!(kinds.iter().any(|k| matches!(k, StepKind::Completion)));
    assert!(result.response.text.contains("echo: hello"));

    // 总 token 数应累计两次调用
    assert_eq!(result.trace.total_usage.total_tokens, 45);
}

/// HITL：AutoApproveHandler 自动批准需要确认的工具，循环正常完成。
#[tokio::test]
async fn hitl_auto_approve_allows_confirmed_tool() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response(
            "confirmed_op",
            serde_json::json!({"value": "test"}),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("操作已确认并完成。")))
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
        .tool(ConfirmedTool)
        .interrupt_handler(AutoApproveHandler)
        .build()
        .unwrap();

    let mut session = agent.session("user1", "hitl-auto").await.unwrap();
    let result = session.chat("执行受保护操作").await.unwrap();

    assert!(matches!(result.trace.state, ExecutionState::Completed));
    assert!(result.response.text.contains("确认"));
}

/// MaxIterations：超过最大迭代次数时返回 ArcError::MaxIterations。
#[tokio::test]
async fn max_iterations_returns_error() {
    let server = MockServer::start().await;

    // 每次都返回工具调用，造成无限循环，触发 MaxIterations
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response(
            "echo",
            serde_json::json!({"text": "loop"}),
        )))
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
        .max_iterations(3)
        .build()
        .unwrap();

    let mut session = agent.session("user1", "max-iter").await.unwrap();
    let err = session.chat("一直调用工具").await.unwrap_err();

    assert!(matches!(err, ArcError::MaxIterations(3)));
}

/// Timeout：请求耗时超过配置的 timeout 时返回 ArcError::Timeout。
#[tokio::test]
async fn timeout_returns_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(text_response("slow"))
                .set_delay(Duration::from_millis(300)),
        )
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
        .config(AgentConfig {
            timeout: Duration::from_millis(50),
            ..Default::default()
        })
        .build()
        .unwrap();

    let mut session = agent.session("user1", "timeout").await.unwrap();
    let err = session.chat("slow request").await.unwrap_err();

    assert!(matches!(err, ArcError::Timeout(_)));
}

/// SessionBusy：同一会话发起并发写时，第二个立即返回 SessionBusy。
#[tokio::test]
async fn concurrent_chat_returns_session_busy() {
    let server = MockServer::start().await;

    // 第一个请求故意延迟，让第二个并发请求先检查到 Running 状态
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(text_response("done"))
                .set_delay(Duration::from_millis(200)),
        )
        .mount(&server)
        .await;

    let agent = make_agent(&server.uri());
    let mut s1 = agent.session("user1", "busy-test").await.unwrap();
    // Clone 共享同一内部状态
    let mut s2 = s1.clone();

    // 第一个请求在后台跑
    let h = tokio::spawn(async move { s1.chat("first").await });

    // 给第一个请求一点时间进入 Running 状态
    tokio::time::sleep(Duration::from_millis(20)).await;

    // 第二个请求应立即返回 SessionBusy
    let err = s2.chat("second").await.unwrap_err();
    assert!(matches!(err, ArcError::SessionBusy));

    // 等第一个完成
    let _ = h.await.unwrap();
}

/// Session 持久化：保存、恢复、确认历史完整。
#[tokio::test]
async fn session_persist_and_restore() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("我记住了。")))
        .mount(&server)
        .await;

    let agent = make_agent(&server.uri());
    let store = InMemorySessionStore::default();

    // 第一次会话
    let mut session1 = agent.session("user1", "persist-sess").await.unwrap();
    let _ = session1.chat("你好").await.unwrap();
    session1.persist(&store).await.unwrap();

    // 新建会话对象，恢复历史
    let session2 = agent.session("user1", "persist-sess").await.unwrap();
    session2.restore(&store).await.unwrap();

    // 验证历史消息数量：1 user + 1 assistant = 2
    let msgs = store.load(&session2.session_id).await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].text(), "你好");
}
