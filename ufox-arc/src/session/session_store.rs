use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use ufox_llm::Message;

use crate::error::ArcError;
use crate::session::SessionId;
use crate::session::store::SessionStore;

/// 内存版会话存储；仅用于测试与短生命周期场景。
#[derive(Default)]
pub struct InMemorySessionStore {
    data: RwLock<HashMap<String, Vec<Message>>>,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn save(&self, session_id: &SessionId, messages: &[Message]) -> Result<(), ArcError> {
        self.data
            .write()
            .await
            .insert(session_id.0.clone(), messages.to_vec());
        Ok(())
    }

    async fn load(&self, session_id: &SessionId) -> Result<Vec<Message>, ArcError> {
        Ok(self
            .data
            .read()
            .await
            .get(&session_id.0)
            .cloned()
            .unwrap_or_default())
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), ArcError> {
        self.data.write().await.remove(&session_id.0);
        Ok(())
    }
}

/// SQLite 版会话存储；消息列表以 JSON 整体存储，支持跨进程恢复。
pub struct SqliteSessionStore {
    pool: sqlx::SqlitePool,
}

impl SqliteSessionStore {
    /// 连接已有数据库文件；不存在则创建。
    pub async fn open(path: &str) -> Result<Self, ArcError> {
        Self::from_url(&format!("sqlite:{path}?mode=rwc")).await
    }

    /// 使用内存数据库（测试专用）。
    pub async fn in_memory() -> Result<Self, ArcError> {
        Self::from_url("sqlite::memory:").await
    }

    async fn from_url(url: &str) -> Result<Self, ArcError> {
        let pool = sqlx::SqlitePool::connect(url)
            .await
            .map_err(|e| ArcError::Session(e.to_string()))?;

        sqlx::query(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS session_messages (
                 session_id TEXT PRIMARY KEY,
                 messages_json TEXT NOT NULL,
                 updated_at TEXT NOT NULL
             )",
        )
        .execute(&pool)
        .await
        .map_err(|e| ArcError::Session(e.to_string()))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn save(&self, session_id: &SessionId, messages: &[Message]) -> Result<(), ArcError> {
        let json = serde_json::to_string(messages)
            .map_err(|e| ArcError::Session(format!("serialize messages: {e}")))?;
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO session_messages (session_id, messages_json, updated_at)
             VALUES (?, ?, ?)
             ON CONFLICT(session_id) DO UPDATE SET
                 messages_json = excluded.messages_json,
                 updated_at    = excluded.updated_at",
        )
        .bind(&session_id.0)
        .bind(&json)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| ArcError::Session(e.to_string()))?;

        Ok(())
    }

    async fn load(&self, session_id: &SessionId) -> Result<Vec<Message>, ArcError> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT messages_json FROM session_messages WHERE session_id = ?")
                .bind(&session_id.0)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| ArcError::Session(e.to_string()))?;

        match row {
            None => Ok(vec![]),
            Some((json,)) => serde_json::from_str(&json)
                .map_err(|e| ArcError::Session(format!("deserialize messages: {e}"))),
        }
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), ArcError> {
        sqlx::query("DELETE FROM session_messages WHERE session_id = ?")
            .bind(&session_id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| ArcError::Session(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ufox_llm::{ContentPart, Role};

    fn make_messages() -> Vec<Message> {
        vec![
            Message {
                role: Role::User,
                content: vec![ContentPart::text("hello")],
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::text("hi there")],
                name: None,
            },
        ]
    }

    #[tokio::test]
    async fn in_memory_save_load_delete() {
        let store = InMemorySessionStore::default();
        let sid = SessionId("sess-1".into());
        let msgs = make_messages();

        store.save(&sid, &msgs).await.unwrap();
        let loaded = store.load(&sid).await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text(), "hello");

        store.delete(&sid).await.unwrap();
        assert!(store.load(&sid).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn sqlite_save_load_overwrite_delete() {
        let store = SqliteSessionStore::in_memory().await.unwrap();
        let sid = SessionId("sess-sqlite".into());
        let msgs = make_messages();

        store.save(&sid, &msgs).await.unwrap();
        let loaded = store.load(&sid).await.unwrap();
        assert_eq!(loaded.len(), 2);

        // overwrite with fewer messages
        let truncated = vec![msgs[0].clone()];
        store.save(&sid, &truncated).await.unwrap();
        let loaded2 = store.load(&sid).await.unwrap();
        assert_eq!(loaded2.len(), 1);

        store.delete(&sid).await.unwrap();
        assert!(store.load(&sid).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn load_nonexistent_returns_empty() {
        let store = SqliteSessionStore::in_memory().await.unwrap();
        let sid = SessionId("no-such-session".into());
        assert!(store.load(&sid).await.unwrap().is_empty());
    }
}
