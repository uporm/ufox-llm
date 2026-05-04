use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::ArcError;

use super::{ThreadId, ThreadSnapshot, ThreadStore};

/// 内存版线程存储；仅用于测试与短生命周期场景。
#[derive(Default)]
pub struct InMemoryThreadStore {
    data: RwLock<HashMap<String, ThreadSnapshot>>,
}

#[async_trait]
impl ThreadStore for InMemoryThreadStore {
    async fn save(&self, snapshot: &ThreadSnapshot) -> Result<(), ArcError> {
        self.data
            .write()
            .await
            .insert(snapshot.thread_id.0.clone(), snapshot.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<ThreadSnapshot>, ArcError> {
        Ok(self.data.read().await.get(&thread_id.0).cloned())
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<(), ArcError> {
        self.data.write().await.remove(&thread_id.0);
        Ok(())
    }
}

/// SQLite 版线程存储；线程快照以 JSON 整体存储，支持跨进程恢复。
pub struct SqliteThreadStore {
    pool: sqlx::SqlitePool,
}

impl SqliteThreadStore {
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
            .map_err(|e| ArcError::Thread(e.to_string()))?;

        sqlx::query(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS thread_snapshots (
                 thread_id TEXT PRIMARY KEY,
                 user_id TEXT NOT NULL,
                 snapshot_json TEXT NOT NULL,
                 updated_at TEXT NOT NULL
             )",
        )
        .execute(&pool)
        .await
        .map_err(|e| ArcError::Thread(e.to_string()))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl ThreadStore for SqliteThreadStore {
    async fn save(&self, snapshot: &ThreadSnapshot) -> Result<(), ArcError> {
        let json = serde_json::to_string(snapshot)
            .map_err(|e| ArcError::Thread(format!("serialize thread snapshot: {e}")))?;
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO thread_snapshots (thread_id, user_id, snapshot_json, updated_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(thread_id) DO UPDATE SET
                 user_id = excluded.user_id,
                 snapshot_json = excluded.snapshot_json,
                 updated_at = excluded.updated_at",
        )
        .bind(&snapshot.thread_id.0)
        .bind(&snapshot.user_id.0)
        .bind(&json)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| ArcError::Thread(e.to_string()))?;

        Ok(())
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<ThreadSnapshot>, ArcError> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT snapshot_json FROM thread_snapshots WHERE thread_id = ?")
                .bind(&thread_id.0)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| ArcError::Thread(e.to_string()))?;

        match row {
            None => Ok(None),
            Some((json,)) => serde_json::from_str(&json)
                .map(Some)
                .map_err(|e| ArcError::Thread(format!("deserialize thread snapshot: {e}"))),
        }
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<(), ArcError> {
        sqlx::query("DELETE FROM thread_snapshots WHERE thread_id = ?")
            .bind(&thread_id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| ArcError::Thread(e.to_string()))?;
        Ok(())
    }
}
