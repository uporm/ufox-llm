pub mod session_store;
pub mod store;

pub use session_store::{InMemorySessionStore, SqliteSessionStore};
pub use store::SessionStore;

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::stream;
use futures::{Stream, StreamExt};
use ufox_llm::{ChatResponse, ContentPart, FinishReason, Message, Role, ToolCall, Usage};

use crate::agent::Agent;
use crate::agent::loop_::{LoopCtx, LoopResult, MemoryCtx, build_request, execute_tools, run_loop};
use crate::agent::step::{ExecutionState, ExecutionStep, StepInput, StepKind, StepOutput};
use crate::error::ArcError;
use crate::memory::{Memory, MemoryFilter, MemoryId, MemoryScope, strategy};
use crate::multimodal::{DefaultExtractor, MediaExtractor, MediaRef, Modality};
use ufox_llm::MediaSource;

/// 用户的唯一标识，用于跨会话记忆归属。
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct UserId(pub String);

impl From<String> for UserId {
    fn from(s: String) -> Self {
        UserId(s)
    }
}

impl From<&str> for UserId {
    fn from(s: &str) -> Self {
        UserId(s.to_string())
    }
}

impl std::fmt::Display for UserId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 会话的唯一标识，用于多轮对话与会话恢复。
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SessionId(pub String);

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        SessionId(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        SessionId(s.to_string())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `chat()` / `chat_stream()` 的统一输入类型。
pub enum SessionInput {
    Text(String),
    Message(Message),
}

impl From<String> for SessionInput {
    fn from(s: String) -> Self {
        SessionInput::Text(s)
    }
}

impl From<&str> for SessionInput {
    fn from(s: &str) -> Self {
        SessionInput::Text(s.to_string())
    }
}

impl From<Message> for SessionInput {
    fn from(m: Message) -> Self {
        SessionInput::Message(m)
    }
}

impl SessionInput {
    fn into_message(self) -> Message {
        match self {
            SessionInput::Text(text) => Message {
                role: Role::User,
                content: vec![ContentPart::text(text)],
                name: None,
            },
            SessionInput::Message(m) => m,
        }
    }
}

/// 会话的运行状态；同一会话禁止并发写入。
#[derive(Debug)]
enum SessionState {
    Idle,
    Running {
        // Phase 7 会话超时检测时使用
        #[allow(dead_code)]
        started_at: Instant,
    },
}

/// 完整执行轨迹，每次 `chat()` 生成一份。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionTrace {
    pub user_id: UserId,
    pub session_id: SessionId,
    pub steps: Vec<ExecutionStep>,
    pub state: ExecutionState,
    pub total_duration: Duration,
    pub total_usage: Usage,
}

/// `chat()` 的完整返回结果。
#[derive(Debug)]
pub struct ExecutionResult {
    pub user_id: UserId,
    pub session_id: SessionId,
    pub response: ChatResponse,
    pub trace: ExecutionTrace,
}

/// `chat_stream()` 的单个流式事件。
#[derive(Debug)]
pub struct ExecutionEvent {
    pub user_id: UserId,
    pub session_id: SessionId,
    /// 流式文本/工具调用增量块；流结束时为 `None`。
    pub chunk: Option<ufox_llm::ChatChunk>,
    pub step: Option<ExecutionStep>,
    pub state_change: Option<ExecutionState>,
}

/// `chat_stream()` 返回的事件流。
pub struct ExecutionEventStream {
    inner: Pin<Box<dyn Stream<Item = Result<ExecutionEvent, ArcError>> + Send>>,
}

impl Stream for ExecutionEventStream {
    type Item = Result<ExecutionEvent, ArcError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// 会话内部共享状态，通过 `Arc` 在会话对象与流之间传递。
pub(crate) struct SessionShared {
    agent: Arc<Agent>,
    messages: tokio::sync::Mutex<Vec<Message>>,
    state: tokio::sync::Mutex<SessionState>,
}

/// 单个用户会话，持有消息历史和运行状态。
///
/// 同一 `session_id` 不允许并发写；新的写请求立即返回 `SessionBusy`。
/// `Clone` 克隆共享同一 `Arc` 内部状态，适用于多处持有同一会话引用的场景。
pub struct Session {
    pub user_id: UserId,
    pub session_id: SessionId,
    pub(crate) shared: Arc<SessionShared>,
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Session {
    pub(crate) fn new(user_id: UserId, session_id: SessionId, agent: Arc<Agent>) -> Self {
        Self {
            user_id,
            session_id,
            shared: Arc::new(SessionShared {
                agent,
                messages: tokio::sync::Mutex::new(Vec::new()),
                state: tokio::sync::Mutex::new(SessionState::Idle),
            }),
        }
    }

    /// 从 `SessionStore` 恢复历史消息，覆盖当前会话内容。
    ///
    /// 典型用法：进程重启后用同一 `session_id` 重新创建 `Session`，
    /// 调用 `restore()` 把历史加载回来，再继续对话。
    pub async fn restore(&self, store: &dyn SessionStore) -> Result<(), ArcError> {
        let msgs = store.load(&self.session_id).await?;
        *self.shared.messages.lock().await = msgs;
        Ok(())
    }

    /// 将当前消息历史保存到 `SessionStore`。
    pub async fn persist(&self, store: &dyn SessionStore) -> Result<(), ArcError> {
        let msgs = self.shared.messages.lock().await;
        store.save(&self.session_id, &msgs).await
    }

    /// 非流式对话；返回带完整步骤轨迹的执行结果。
    pub async fn chat<I>(&mut self, input: I) -> Result<ExecutionResult, ArcError>
    where
        I: Into<SessionInput>,
    {
        let wall_start = Instant::now();

        {
            let mut state = self.shared.state.lock().await;
            if matches!(*state, SessionState::Running { .. }) {
                return Err(ArcError::SessionBusy);
            }
            *state = SessionState::Running {
                started_at: Instant::now(),
            };
        }

        let user_message = input.into().into_message();
        let loop_result = self.run_chat(user_message).await;

        *self.shared.state.lock().await = SessionState::Idle;

        let LoopResult {
            steps,
            final_response,
            state,
            total_usage,
        } = loop_result?;

        let trace = ExecutionTrace {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            steps,
            state,
            total_duration: wall_start.elapsed(),
            total_usage,
        };

        Ok(ExecutionResult {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            response: final_response,
            trace,
        })
    }

    /// 流式对话；调用方通过 `StreamExt::next()` 消费增量事件。
    ///
    /// 完整支持工具调用循环、记忆检索、HITL、速率限制；
    /// Think 步骤以流式 chunk 推送，Act 步骤以 `step` 事件推送。
    pub async fn chat_stream<I>(&mut self, input: I) -> Result<ExecutionEventStream, ArcError>
    where
        I: Into<SessionInput>,
    {
        {
            let mut state = self.shared.state.lock().await;
            if matches!(*state, SessionState::Running { .. }) {
                return Err(ArcError::SessionBusy);
            }
            *state = SessionState::Running {
                started_at: Instant::now(),
            };
        }

        let user_message = input.into().into_message();
        self.shared.messages.lock().await.push(user_message);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ExecutionEvent, ArcError>>(32);
        let shared = Arc::clone(&self.shared);
        let user_id = self.user_id.clone();
        let session_id = self.session_id.clone();

        tokio::spawn(async move {
            if let Err(e) = run_streaming_loop(
                Arc::clone(&shared),
                user_id.clone(),
                session_id.clone(),
                &tx,
            )
            .await
            {
                let _ = tx.send(Err(e)).await;
            }
            *shared.state.lock().await = SessionState::Idle;
        });

        let event_stream = stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };

        Ok(ExecutionEventStream {
            inner: Box::pin(event_stream),
        })
    }

    /// 将用户消息追加进历史，然后委托给推理循环执行。
    async fn run_chat(&self, user_message: Message) -> Result<LoopResult, ArcError> {
        let mut messages = self.shared.messages.lock().await;
        messages.push(user_message);
        let agent = &self.shared.agent;
        run_loop(
            LoopCtx {
                llm: &agent.llm,
                system: agent.system.as_deref(),
                config: &agent.config,
                tools: agent.tools.as_deref(),
                memory: agent.memory.as_deref().map(|store| MemoryCtx {
                    store,
                    user_id: &self.user_id,
                    session_id: &self.session_id,
                }),
                interrupt: agent
                    .interrupt_handler
                    .as_deref()
                    .map(|h| (h, &self.user_id, &self.session_id)),
            },
            &mut messages,
        )
        .await
    }

    /// 向会话记忆写入一条记录。
    pub async fn remember_session(
        &self,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, ArcError> {
        let store = self
            .shared
            .agent
            .memory
            .as_deref()
            .ok_or_else(|| ArcError::Memory("no memory store configured".into()))?;
        let memory = Memory::new_session(self.session_id.clone(), content).with_tags(tags);
        store.insert(memory).await
    }

    /// 向用户记忆写入一条记录（跨会话持久）。
    pub async fn remember_user(
        &self,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, ArcError> {
        let store = self
            .shared
            .agent
            .memory
            .as_deref()
            .ok_or_else(|| ArcError::Memory("no memory store configured".into()))?;
        let memory = Memory::new_user(self.user_id.clone(), content).with_tags(tags);
        store.insert(memory).await
    }

    /// 按过滤条件检索记忆。
    pub async fn search_memory(&self, filter: MemoryFilter) -> Result<Vec<Memory>, ArcError> {
        let store = self
            .shared
            .agent
            .memory
            .as_deref()
            .ok_or_else(|| ArcError::Memory("no memory store configured".into()))?;
        store.find(filter).await
    }

    /// 查询当前会话的所有记忆。
    pub async fn session_memories(&self) -> Result<Vec<Memory>, ArcError> {
        self.search_memory(MemoryFilter {
            scope: Some(MemoryScope::Session {
                session_id: self.session_id.clone(),
            }),
            ..Default::default()
        })
        .await
    }

    /// 查询当前用户的所有用户级记忆。
    pub async fn user_memories(&self) -> Result<Vec<Memory>, ArcError> {
        self.search_memory(MemoryFilter {
            scope: Some(MemoryScope::User {
                user_id: self.user_id.clone(),
            }),
            ..Default::default()
        })
        .await
    }

    /// 将媒体内容附加到当前会话：
    /// 1. 用 `DefaultExtractor` 提取 `ContentPart` 列表；
    /// 2. 将提取结果作为 User 消息插入对话历史（后续轮次无需重复上传）；
    /// 3. 若会话配置了记忆存储，将来源元信息写入会话记忆以便跨轮检索。
    ///
    /// 返回 `MediaRef` 可用于标识本次附加的媒体。
    pub async fn attach(
        &self,
        source: MediaSource,
        modality: Modality,
        tags: Vec<String>,
    ) -> Result<MediaRef, ArcError> {
        let extractor = DefaultExtractor;
        let extracted = extractor.extract(source, modality).await?;

        // 将提取内容加入对话历史
        {
            let mut messages = self.shared.messages.lock().await;
            messages.push(Message {
                role: Role::User,
                content: extracted.parts.clone(),
                name: Some("attached_media".to_string()),
            });
        }

        let media_ref = MediaRef::new();

        // 将来源元信息持久化到会话记忆
        if let Some(store) = self.shared.agent.memory.as_deref() {
            let mut mem_meta = extracted.metadata.clone();
            mem_meta.insert(
                "media_ref".into(),
                serde_json::Value::String(media_ref.0.to_string()),
            );
            mem_meta.insert(
                "modality".into(),
                serde_json::to_value(modality).unwrap_or_default(),
            );

            // 对于文本/文档内容，直接保存提取的文本；否则保存来源描述
            let content = match modality {
                Modality::Text | Modality::Document => extracted
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text(t) = p {
                            Some(t.text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => format!("[{modality:?} media attached]"),
            };

            let memory = Memory::new_session(self.session_id.clone(), content)
                .with_tags(tags)
                .with_metadata(
                    "media_source".to_string(),
                    serde_json::to_value(&mem_meta).unwrap_or_default(),
                );

            store.insert(memory).await?;
        }

        Ok(media_ref)
    }
}

/// 完整推理循环的流式版本：每个 Think 步骤以 chunk 事件推送，
/// Act 步骤以 step 事件推送，结束时发送 `state_change: Completed`。
///
/// 函数持有对 `SessionShared` 的 `Arc` 引用，在 `tokio::spawn` 中运行。
async fn run_streaming_loop(
    shared: Arc<SessionShared>,
    user_id: UserId,
    session_id: SessionId,
    tx: &tokio::sync::mpsc::Sender<Result<ExecutionEvent, ArcError>>,
) -> Result<(), ArcError> {
    let agent = &shared.agent;
    let config = &agent.config;
    let deadline = Instant::now() + config.timeout;

    // Perceive（可选）：检索记忆并注入上下文
    if config.enable_perceive
        && let Some(store) = agent.memory.as_deref()
    {
        let retrieved = strategy::retrieve_context(store, &session_id, &user_id, 10).await;
        let context_text = strategy::format_context(&retrieved);
        if !context_text.is_empty() {
            shared.messages.lock().await.insert(
                0,
                Message {
                    role: Role::User,
                    content: vec![ContentPart::text(context_text)],
                    name: Some("memory_context".to_string()),
                },
            );
        }
    }

    let mut iteration = 0usize;
    let mut step_idx = 0usize;

    loop {
        if iteration >= config.max_iterations {
            return Err(ArcError::MaxIterations(config.max_iterations));
        }
        iteration += 1;

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ArcError::Timeout(config.timeout));
        }

        // Build request from current history (lock then release before streaming)
        let req = {
            let messages = shared.messages.lock().await;
            build_request(
                agent.system.as_deref(),
                &messages,
                config,
                agent.tools.as_deref(),
            )
        };

        // Think（流式）
        let mut accumulated_text = String::new();
        let mut accumulated_calls: Vec<ToolCall> = Vec::new();
        let mut last_finish_reason: Option<FinishReason> = None;
        let mut step_usage = None;

        let stream_result = tokio::time::timeout(remaining, agent.llm.chat_stream(req)).await;

        let mut llm_stream = match stream_result {
            Err(_) => return Err(ArcError::Timeout(config.timeout)),
            Ok(Err(e)) => return Err(ArcError::Llm(e)),
            Ok(Ok(s)) => s,
        };

        while let Some(chunk_result) = llm_stream.next().await {
            let chunk = match chunk_result {
                Err(e) => return Err(ArcError::Llm(e)),
                Ok(c) => c,
            };
            if let Some(t) = &chunk.text_delta {
                accumulated_text.push_str(t);
            }
            accumulated_calls.extend(chunk.tool_calls.clone());
            if chunk.finish_reason.is_some() {
                last_finish_reason = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                step_usage = chunk.usage.clone();
            }

            if tx
                .send(Ok(ExecutionEvent {
                    user_id: user_id.clone(),
                    session_id: session_id.clone(),
                    chunk: Some(chunk),
                    step: None,
                    state_change: None,
                }))
                .await
                .is_err()
            {
                // 接收端已丢弃，静默退出
                return Ok(());
            }
        }

        let finish_reason = last_finish_reason.unwrap_or(FinishReason::Completed);
        let has_tool_calls =
            !accumulated_calls.is_empty() || finish_reason == FinishReason::ToolCalls;

        // 将 assistant 消息追加进历史
        {
            let mut content: Vec<ContentPart> = Vec::new();
            if !accumulated_text.is_empty() {
                content.push(ContentPart::text(accumulated_text));
            }
            for call in &accumulated_calls {
                content.push(ContentPart::ToolCall(call.clone()));
            }
            if !content.is_empty() {
                shared.messages.lock().await.push(Message {
                    role: Role::Assistant,
                    content,
                    name: None,
                });
            }
        }
        step_idx += 1;

        if !has_tool_calls {
            let _ = tx
                .send(Ok(ExecutionEvent {
                    user_id: user_id.clone(),
                    session_id: session_id.clone(),
                    chunk: None,
                    step: None,
                    state_change: Some(ExecutionState::Completed),
                }))
                .await;
            return Ok(());
        }

        // Act：执行工具调用
        let interrupt = agent
            .interrupt_handler
            .as_deref()
            .map(|h| (h, &user_id, &session_id));
        let act_start = Instant::now();
        let tool_results =
            execute_tools(&accumulated_calls, agent.tools.as_deref(), interrupt).await?;

        for result in &tool_results {
            shared.messages.lock().await.push(Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult(result.clone())],
                name: None,
            });
        }

        let act_step = ExecutionStep {
            index: step_idx,
            kind: StepKind::Act,
            input: StepInput::ToolCalls(accumulated_calls),
            output: StepOutput::ToolResults(tool_results),
            duration: act_start.elapsed(),
            usage: step_usage,
        };
        step_idx += 1;

        if tx
            .send(Ok(ExecutionEvent {
                user_id: user_id.clone(),
                session_id: session_id.clone(),
                chunk: None,
                step: Some(act_step),
                state_change: None,
            }))
            .await
            .is_err()
        {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionId, SessionInput, UserId};
    use ufox_llm::{ContentPart, Message, Role};

    #[test]
    fn user_id_from_str() {
        let id = UserId::from("user_123");
        assert_eq!(id.0, "user_123");
    }

    #[test]
    fn session_id_from_string() {
        let id = SessionId::from("session-abc".to_string());
        assert_eq!(id.0, "session-abc");
    }

    #[test]
    fn session_input_text_becomes_user_message() {
        let input = SessionInput::from("hello");
        let msg = input.into_message();
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), "hello");
    }

    #[test]
    fn session_input_message_passes_through() {
        let original = Message {
            role: Role::User,
            content: vec![ContentPart::text("hi")],
            name: None,
        };
        let input = SessionInput::from(original.clone());
        let msg = input.into_message();
        assert_eq!(msg.text(), original.text());
    }
}
