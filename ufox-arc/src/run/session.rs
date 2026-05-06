use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use ufox_llm::{ChatChunk, ChatResponse, ContentPart, Message, Role, Usage};

use super::trace::{ExecutionState, ExecutionStep};
use crate::error::ArcError;
use crate::thread::{ThreadId, UserId};

/// 单次执行的唯一标识。
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RunId(pub String);

impl RunId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

/// `run()` / `run_stream()` 的统一输入类型。
pub enum RunInput {
    Text(String),
    Message(Message),
}

impl From<String> for RunInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for RunInput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Message> for RunInput {
    fn from(m: Message) -> Self {
        Self::Message(m)
    }
}

impl RunInput {
    pub fn into_message(self) -> Message {
        match self {
            Self::Text(text) => Message {
                role: Role::User,
                content: vec![ContentPart::text(text)],
                name: None,
            },
            Self::Message(message) => message,
        }
    }
}

/// 单次运行的完整轨迹。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunTrace {
    pub run_id: RunId,
    pub user_id: UserId,
    pub thread_id: ThreadId,
    pub steps: Vec<ExecutionStep>,
    pub state: ExecutionState,
    pub total_duration: Duration,
    pub total_usage: Usage,
}

/// 单次运行的完整结果。
#[derive(Debug)]
pub struct RunResult {
    pub run_id: RunId,
    pub user_id: UserId,
    pub thread_id: ThreadId,
    pub response: ChatResponse,
    pub trace: RunTrace,
}

/// `run_stream()` 的单个流式事件。
#[derive(Debug)]
pub struct RunEvent {
    pub run_id: RunId,
    pub user_id: UserId,
    pub thread_id: ThreadId,
    pub chunk: Option<ChatChunk>,
    pub step: Option<ExecutionStep>,
    pub state_change: Option<ExecutionState>,
}

/// `run_stream()` 返回的事件流。
pub struct RunEventStream {
    pub(crate) inner: Pin<Box<dyn Stream<Item = Result<RunEvent, ArcError>> + Send>>,
}

impl Stream for RunEventStream {
    type Item = Result<RunEvent, ArcError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
