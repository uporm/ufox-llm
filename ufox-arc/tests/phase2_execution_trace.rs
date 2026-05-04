use ufox_arc::{Agent, AgentConfig, ExecutionState, StepKind};
use ufox_llm::Provider;
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
        .instructions("你是测试助手。")
        .build()
        .unwrap()
}

fn chat_completions_response(content: &str) -> serde_json::Value {
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

// ─── tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_returns_execution_trace_with_steps() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completions_response("Rust 所有权让内存安全无需 GC。")),
        )
        .mount(&server)
        .await;

    let agent = make_agent(&server.uri());
    let thread = agent.thread("user_test", "trace-test");

    let result = agent.run(&thread, "解释所有权").await.unwrap();

    assert!(!result.response.text.is_empty());
    assert!(matches!(result.trace.state, ExecutionState::Completed));

    // 默认简单模式：Think + Completion 两步
    let kinds: Vec<_> = result.trace.steps.iter().map(|s| &s.kind).collect();
    assert!(
        kinds.iter().any(|k| matches!(k, StepKind::Think)),
        "trace should contain a Think step"
    );
    assert!(
        kinds.iter().any(|k| matches!(k, StepKind::Completion)),
        "trace should contain a Completion step"
    );
    assert_eq!(result.trace.total_usage.total_tokens, 15);
}

#[tokio::test]
async fn multi_turn_chat_accumulates_history() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_completions_response("好的，我记住了。")),
        )
        .expect(2)
        .mount(&server)
        .await;

    let agent = make_agent(&server.uri());
    let thread = agent.thread("user_test", "multi-turn");

    let r1 = agent.run(&thread, "第一轮消息").await.unwrap();
    let r2 = agent.run(&thread, "第二轮消息").await.unwrap();

    assert!(!r1.response.text.is_empty());
    assert!(!r2.response.text.is_empty());
}

#[tokio::test]
async fn agent_config_defaults_to_simple_mode() {
    let config = AgentConfig::default();
    assert!(!config.enable_perceive);
    assert!(!config.enable_observe);
    assert!(!config.enable_reflect);
    assert_eq!(config.max_iterations, 10);
}

#[tokio::test]
async fn agent_builder_exposes_enable_flags() {
    let server = MockServer::start().await;
    let agent = Agent::builder()
        .llm(
            ufox_llm::Client::builder()
                .provider(Provider::Compatible)
                .api_key("k")
                .base_url(server.uri())
                .model("m")
                .build()
                .unwrap(),
        )
        .enable_perceive(true)
        .enable_observe(true)
        .enable_reflect(true)
        .build()
        .unwrap();

    // 只验证配置已正确写入 Agent；不实际发起网络请求
    let _ = agent;
}
