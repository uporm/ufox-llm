pub mod attachment;
pub mod store;

pub use attachment::{AttachmentKind, AttachmentRef};
pub use store::{InMemoryThreadStore, SqliteThreadStore, ThreadStore};

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use ufox_llm::MediaSource;
use ufox_llm::{Message, Role};

use self::attachment::{DefaultExtractor, MediaExtractor};
use crate::error::ArcError;

// 保留既有消息名，避免影响依赖该标记的历史数据或下游逻辑。
const ATTACHED_ATTACHMENT_MESSAGE_NAME: &str = "attached_media";

/// 用户的唯一标识，用于跨线程记忆归属。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// 线程的唯一标识，用于多轮对话与线程恢复。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub String);

impl ThreadId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for ThreadId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ThreadId {
    fn from(s: String) -> Self {
        ThreadId(s)
    }
}

impl From<&str> for ThreadId {
    fn from(s: &str) -> Self {
        ThreadId(s.to_string())
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 线程运行状态；同一线程禁止并发写入。
#[derive(Debug)]
pub(crate) enum ThreadState {
    Idle,
    Busy {
        #[allow(dead_code)]
        started_at: Instant,
    },
}

/// 线程快照，用于持久化当前线程状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadSnapshot {
    pub user_id: UserId,
    pub thread_id: ThreadId,
    pub messages: Vec<Message>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// 线程内部共享状态，通过 `Arc` 在运行与调用方之间传递。
pub(crate) struct ThreadShared {
    pub messages: Mutex<Vec<Message>>,
    pub metadata: RwLock<serde_json::Map<String, serde_json::Value>>,
    pub state: Mutex<ThreadState>,
}

/// 单个用户线程，只持有消息历史和运行状态。
#[derive(Clone)]
pub struct Thread {
    pub user_id: UserId,
    pub thread_id: ThreadId,
    pub(crate) shared: Arc<ThreadShared>,
}

impl Thread {
    pub fn new(user_id: UserId, thread_id: ThreadId) -> Self {
        Self {
            user_id,
            thread_id,
            shared: Arc::new(ThreadShared {
                messages: Mutex::new(Vec::new()),
                metadata: RwLock::new(serde_json::Map::new()),
                state: Mutex::new(ThreadState::Idle),
            }),
        }
    }

    /// 返回当前线程的消息历史快照。
    pub async fn messages(&self) -> Vec<Message> {
        self.shared.messages.lock().await.clone()
    }

    /// 以完整快照覆盖当前线程内容。
    pub async fn load(&self, store: &dyn ThreadStore) -> Result<(), ArcError> {
        let Some(snapshot) = store.load(&self.thread_id).await? else {
            return Ok(());
        };
        *self.shared.messages.lock().await = snapshot.messages;
        *self.shared.metadata.write().await = snapshot.metadata;
        Ok(())
    }

    /// 将当前线程保存到存储。
    pub async fn save(&self, store: &dyn ThreadStore) -> Result<(), ArcError> {
        store.save(&self.snapshot().await).await
    }

    /// 生成当前线程快照。
    pub async fn snapshot(&self) -> ThreadSnapshot {
        ThreadSnapshot {
            user_id: self.user_id.clone(),
            thread_id: self.thread_id.clone(),
            messages: self.shared.messages.lock().await.clone(),
            metadata: self.shared.metadata.read().await.clone(),
            updated_at: chrono::Utc::now(),
        }
    }

    /// 清空线程消息历史。
    pub async fn clear(&self) {
        self.shared.messages.lock().await.clear();
    }

    /// 用于兼容层或高级调用方直接追加消息。
    pub async fn append_message(&self, message: Message) {
        self.shared.messages.lock().await.push(message);
    }

    /// 将附件内容附加到当前线程。
    pub async fn attach(
        &self,
        source: MediaSource,
        kind: AttachmentKind,
        _tags: Vec<String>,
    ) -> Result<AttachmentRef, ArcError> {
        let extractor = DefaultExtractor;
        let extracted = extractor.extract(source, kind).await?;

        self.shared.messages.lock().await.push(Message {
            role: Role::User,
            content: extracted.parts,
            name: Some(ATTACHED_ATTACHMENT_MESSAGE_NAME.to_string()),
        });

        Ok(AttachmentRef::new())
    }

    pub(crate) async fn try_start_run(&self) -> Result<(), ArcError> {
        let mut state = self.shared.state.lock().await;
        if matches!(*state, ThreadState::Busy { .. }) {
            return Err(ArcError::ThreadBusy);
        }
        *state = ThreadState::Busy {
            started_at: Instant::now(),
        };
        Ok(())
    }

    pub(crate) async fn finish_run(&self) {
        *self.shared.state.lock().await = ThreadState::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::{ThreadId, UserId};

    #[test]
    fn user_id_from_str() {
        let id = UserId::from("user_123");
        assert_eq!(id.0, "user_123");
    }

    #[test]
    fn thread_id_from_string() {
        let id = ThreadId::from("thread-abc".to_string());
        assert_eq!(id.0, "thread-abc");
    }
}
