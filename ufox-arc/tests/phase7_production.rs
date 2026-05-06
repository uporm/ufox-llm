/// Phase 7 集成测试：工具调用循环、HITL、超时、MaxIterations、Thread 持久化。
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use ufox_arc::interrupt::{InterruptDecision, InterruptHandler, InterruptReason};
use ufox_arc::tools::{Confirm, Tool, ToolError, ToolSpec};
use ufox_arc::{
    Agent, AgentConfig, ArcError, AutoApproveHandler, ExecutionState, InMemoryThreadStore,
    StepKind, ThreadId, ThreadStore, UserId,
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
        .instructions("你是测试助手。")
        // 这些用例依赖精确的 mock 次数与轨迹，关闭 reflect 避免额外模型往返。
        .config(AgentConfig {
            reflect: None,
            ..Default::default()
        })
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
    fn spec(&self) -> &ToolSpec {
        static META: std::sync::OnceLock<ToolSpec> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolSpec {
            name: "echo".into(),
            description: "Echo the input text.".into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
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
    fn spec(&self) -> &ToolSpec {
        static META: std::sync::OnceLock<ToolSpec> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolSpec {
            name: "confirmed_op".into(),
            description: "An operation that requires confirmation.".into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
            timeout: Duration::from_secs(5),
        })
    }

    fn confirm(&self, _params: &Value) -> Confirm {
        Ok(Some("该工具显式要求人工确认".into()))
    }

    async fn execute(&self, params: Value) -> Result<ToolResultPayload, ToolError> {
        let value = params["value"].as_str().unwrap_or("").to_string();
        Ok(ToolResultPayload::text(format!("confirmed: {value}")))
    }
}

struct ConditionalTool;

#[async_trait]
impl Tool for ConditionalTool {
    fn spec(&self) -> &ToolSpec {
        static META: std::sync::OnceLock<ToolSpec> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolSpec {
            name: "conditional_op".into(),
            description: "Conditionally requires confirmation.".into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "mode": { "type": "string" } },
                "required": ["mode"]
            }),
            timeout: Duration::from_secs(5),
        })
    }

    fn confirm(&self, params: &Value) -> Confirm {
        let mode = params["mode"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "conditional_op".into(),
                message: "missing 'mode' parameter".into(),
            })?;

        Ok(match mode {
            "safe" => None,
            "dangerous" => Some("mode=dangerous 会触发保护确认".into()),
            _ => Some("未知模式需要人工确认".into()),
        })
    }

    async fn execute(&self, params: Value) -> Result<ToolResultPayload, ToolError> {
        let mode = params["mode"].as_str().unwrap_or("").to_string();
        Ok(ToolResultPayload::text(format!("conditional: {mode}")))
    }
}

#[derive(Clone)]
struct CountingInterruptHandler {
    count: Arc<AtomicUsize>,
    modify_to_safe: bool,
}

#[async_trait]
impl InterruptHandler for CountingInterruptHandler {
    async fn handle_interrupt(
        &self,
        _reason: InterruptReason,
        _user_id: &UserId,
        _thread_id: &ThreadId,
    ) -> Result<InterruptDecision, ArcError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(if self.modify_to_safe {
            InterruptDecision::ModifyAndContinue(serde_json::json!({ "mode": "safe" }))
        } else {
            InterruptDecision::Continue
        })
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
        .instructions("你是测试助手。")
        .config(AgentConfig {
            reflect: None,
            ..Default::default()
        })
        .tool(EchoTool)
        .build()
        .unwrap();

    let thread = agent.thread("user1", "tool-loop");
    let result = agent.run(&thread, "请调用 echo 工具").await.unwrap();

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
        .config(AgentConfig {
            reflect: None,
            ..Default::default()
        })
        .tool(ConfirmedTool)
        .interrupt_handler(AutoApproveHandler)
        .build()
        .unwrap();

    let thread = agent.thread("user1", "hitl-auto");
    let result = agent.run(&thread, "执行受保护操作").await.unwrap();

    assert!(matches!(result.trace.state, ExecutionState::Completed));
    assert!(result.response.text.contains("确认"));
}

/// 动态确认策略：安全参数不触发 HITL。
#[tokio::test]
async fn dynamic_confirmation_policy_skips_hitl_for_safe_args() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response(
            "conditional_op",
            serde_json::json!({"mode": "safe"}),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(text_response("安全模式执行完成：conditional: safe")),
        )
        .mount(&server)
        .await;

    let count = Arc::new(AtomicUsize::new(0));
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
            reflect: None,
            ..Default::default()
        })
        .tool(ConditionalTool)
        .interrupt_handler(CountingInterruptHandler {
            count: count.clone(),
            modify_to_safe: false,
        })
        .build()
        .unwrap();

    let thread = agent.thread("user1", "conditional-safe");
    let result = agent.run(&thread, "执行安全模式").await.unwrap();

    assert!(matches!(result.trace.state, ExecutionState::Completed));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(result.response.text.contains("conditional: safe"));
}

/// 动态确认策略：危险参数会触发 HITL，且改参后会重新评估策略。
#[tokio::test]
async fn dynamic_confirmation_policy_rechecks_after_modify() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response(
            "conditional_op",
            serde_json::json!({"mode": "dangerous"}),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(text_response("参数已改为安全模式：conditional: safe")),
        )
        .mount(&server)
        .await;

    let count = Arc::new(AtomicUsize::new(0));
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
            reflect: None,
            ..Default::default()
        })
        .tool(ConditionalTool)
        .interrupt_handler(CountingInterruptHandler {
            count: count.clone(),
            modify_to_safe: true,
        })
        .build()
        .unwrap();

    let thread = agent.thread("user1", "conditional-modify");
    let result = agent.run(&thread, "执行危险模式").await.unwrap();

    assert!(matches!(result.trace.state, ExecutionState::Completed));
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(result.response.text.contains("conditional: safe"));
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
        .config(AgentConfig {
            max_iterations: 3,
            reflect: None,
            ..Default::default()
        })
        .build()
        .unwrap();

    let thread = agent.thread("user1", "max-iter");
    let err = agent.run(&thread, "一直调用工具").await.unwrap_err();

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
            reflect: None,
            ..Default::default()
        })
        .build()
        .unwrap();

    let thread = agent.thread("user1", "timeout");
    let err = agent.run(&thread, "slow request").await.unwrap_err();

    assert!(matches!(err, ArcError::Timeout(_)));
}

/// ThreadBusy：同一线程发起并发写时，第二个立即返回 ThreadBusy。
#[tokio::test]
async fn concurrent_chat_returns_thread_busy() {
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
    let t1 = agent.thread("user1", "busy-test");
    let t2 = t1.clone();
    let agent2 = agent.clone();

    // 第一个请求在后台跑
    let h = tokio::spawn(async move { agent2.run(&t1, "first").await });

    // 给第一个请求一点时间进入 Running 状态
    tokio::time::sleep(Duration::from_millis(20)).await;

    // 第二个请求应立即返回 ThreadBusy
    let err = agent.run(&t2, "second").await.unwrap_err();
    assert!(matches!(err, ArcError::ThreadBusy));

    // 等第一个完成
    let _ = h.await.unwrap();
}

/// Thread 持久化：保存、恢复、确认历史完整。
#[tokio::test]
async fn thread_persist_and_restore() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("我记住了。")))
        .mount(&server)
        .await;

    let agent = make_agent(&server.uri());
    let store = InMemoryThreadStore::default();

    // 第一次线程
    let thread1 = agent.thread("user1", "persist-sess");
    let _ = agent.run(&thread1, "你好").await.unwrap();
    thread1.save(&store).await.unwrap();

    // 新建线程对象，恢复历史
    let thread2 = agent.thread("user1", "persist-sess");
    thread2.load(&store).await.unwrap();

    // 验证历史消息数量：1 user + 1 assistant = 2
    let snapshot = store.load(&thread2.thread_id).await.unwrap().unwrap();
    assert_eq!(snapshot.messages.len(), 2);
    assert_eq!(snapshot.messages[0].text(), "你好");
}
