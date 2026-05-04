use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;

use crate::error::ArcError;
use crate::memory::{Memory, MemoryFilter, MemoryId, MemoryScope, MemoryStore};
use crate::thread::{ThreadId, UserId};

/// SQLite 持久化后端；使用 WAL 模式提升并发读性能。
pub struct SqliteMemory {
    pool: SqlitePool,
}

impl SqliteMemory {
    /// 打开（或创建）位于 `path` 的 SQLite 数据库，并运行 schema 迁移。
    pub async fn open(path: &str) -> Result<Self, ArcError> {
        let opts = SqliteConnectOptions::from_str(path)
            .map_err(|e| ArcError::Memory(e.to_string()))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .map_err(|e| ArcError::Memory(e.to_string()))?;

        Self::migrate(&pool).await?;
        Ok(Self { pool })
    }

    /// 内存数据库，仅用于测试。
    pub async fn in_memory() -> Result<Self, ArcError> {
        let pool = SqlitePool::connect("sqlite::memory:")
            .await
            .map_err(|e| ArcError::Memory(e.to_string()))?;
        Self::migrate(&pool).await?;
        Ok(Self { pool })
    }

    async fn migrate(pool: &SqlitePool) -> Result<(), ArcError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY NOT NULL,
                scope_kind  TEXT NOT NULL,
                scope_id    TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                timestamp   TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(scope_kind, scope_id);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp);
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| ArcError::Memory(e.to_string()))?;
        Ok(())
    }

    fn scope_to_row(scope: &MemoryScope) -> (&'static str, String) {
        match scope {
            MemoryScope::Thread { thread_id } => ("thread", thread_id.0.clone()),
            MemoryScope::User { user_id } => ("user", user_id.0.clone()),
        }
    }

    fn row_to_scope(kind: &str, id: &str) -> MemoryScope {
        match kind {
            "user" => MemoryScope::User {
                user_id: UserId(id.to_string()),
            },
            _ => MemoryScope::Thread {
                thread_id: ThreadId(id.to_string()),
            },
        }
    }
}

#[async_trait]
impl MemoryStore for SqliteMemory {
    async fn insert(&self, memory: Memory) -> Result<MemoryId, ArcError> {
        let id = memory.id.to_string();
        let (kind, scope_id) = Self::scope_to_row(&memory.scope);
        let metadata =
            serde_json::to_string(&memory.metadata).map_err(|e| ArcError::Memory(e.to_string()))?;
        let tags =
            serde_json::to_string(&memory.tags).map_err(|e| ArcError::Memory(e.to_string()))?;
        let timestamp = memory.timestamp.to_rfc3339();

        sqlx::query(
            "INSERT INTO memories (id, scope_kind, scope_id, content, metadata, timestamp, tags)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(kind)
        .bind(&scope_id)
        .bind(&memory.content)
        .bind(&metadata)
        .bind(&timestamp)
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(|e| ArcError::Memory(e.to_string()))?;

        Ok(memory.id)
    }

    async fn find(&self, filter: MemoryFilter) -> Result<Vec<Memory>, ArcError> {
        // 构建基础查询；复杂过滤在 Rust 侧完成，避免动态 SQL 拼接。
        let mut sql =
            "SELECT id, scope_kind, scope_id, content, metadata, timestamp, tags FROM memories"
                .to_string();
        let mut conditions: Vec<String> = Vec::new();

        if let Some(ref scope) = filter.scope {
            let (kind, _) = Self::scope_to_row(scope);
            conditions.push(format!("scope_kind = '{kind}'"));
            let id = match scope {
                MemoryScope::Thread { thread_id } => &thread_id.0,
                MemoryScope::User { user_id } => &user_id.0,
            };
            conditions.push(format!("scope_id = '{}'", id.replace('\'', "''")));
        }
        if let Some(since) = filter.since {
            conditions.push(format!("timestamp >= '{}'", since.to_rfc3339()));
        }
        if let Some(until) = filter.until {
            conditions.push(format!("timestamp <= '{}'", until.to_rfc3339()));
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY timestamp DESC");
        if let Some(limit) = filter.limit {
            // over-fetch to allow post-filtering by tags
            let fetch = limit * 4;
            sql.push_str(&format!(" LIMIT {fetch}"));
        }

        let rows = sqlx::query_as::<_, MemoryRow>(&sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ArcError::Memory(e.to_string()))?;

        let mut results: Vec<Memory> = rows
            .into_iter()
            .filter_map(|row| row.try_into_memory().ok())
            .filter(|m| {
                for tag in &filter.tags {
                    if !m.tags.contains(tag) {
                        return false;
                    }
                }
                true
            })
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    async fn replace(&self, id: MemoryId, memory: Memory) -> Result<(), ArcError> {
        let id_str = id.to_string();
        let (kind, scope_id) = Self::scope_to_row(&memory.scope);
        let metadata =
            serde_json::to_string(&memory.metadata).map_err(|e| ArcError::Memory(e.to_string()))?;
        let tags =
            serde_json::to_string(&memory.tags).map_err(|e| ArcError::Memory(e.to_string()))?;
        let timestamp = memory.timestamp.to_rfc3339();

        let rows_affected = sqlx::query(
            "UPDATE memories
             SET scope_kind=?, scope_id=?, content=?, metadata=?, timestamp=?, tags=?
             WHERE id=?",
        )
        .bind(kind)
        .bind(&scope_id)
        .bind(&memory.content)
        .bind(&metadata)
        .bind(&timestamp)
        .bind(&tags)
        .bind(&id_str)
        .execute(&self.pool)
        .await
        .map_err(|e| ArcError::Memory(e.to_string()))?
        .rows_affected();

        if rows_affected == 0 {
            return Err(ArcError::Memory(format!("memory {id} not found")));
        }
        Ok(())
    }

    async fn remove(&self, id: MemoryId) -> Result<(), ArcError> {
        sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| ArcError::Memory(e.to_string()))?;
        Ok(())
    }
}

#[derive(sqlx::FromRow)]
struct MemoryRow {
    id: String,
    scope_kind: String,
    scope_id: String,
    content: String,
    metadata: String,
    timestamp: String,
    tags: String,
}

impl MemoryRow {
    fn try_into_memory(self) -> Result<Memory, ArcError> {
        let id = uuid::Uuid::parse_str(&self.id).map_err(|e| ArcError::Memory(e.to_string()))?;
        let scope = SqliteMemory::row_to_scope(&self.scope_kind, &self.scope_id);
        let metadata: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&self.metadata).map_err(|e| ArcError::Memory(e.to_string()))?;
        let tags: Vec<String> =
            serde_json::from_str(&self.tags).map_err(|e| ArcError::Memory(e.to_string()))?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| ArcError::Memory(e.to_string()))?
            .with_timezone(&chrono::Utc);

        Ok(Memory {
            id,
            scope,
            content: self.content,
            metadata,
            timestamp,
            tags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;
    use crate::thread::{ThreadId, UserId};

    #[tokio::test]
    async fn sqlite_insert_find_remove() {
        let store = SqliteMemory::in_memory().await.unwrap();

        let m = Memory::new_thread(ThreadId("s1".into()), "sqlite test");
        let id = store.insert(m.clone()).await.unwrap();

        let hits = store
            .find(MemoryFilter {
                scope: Some(MemoryScope::Thread {
                    thread_id: ThreadId("s1".into()),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "sqlite test");

        store.remove(id).await.unwrap();
        let hits2 = store.find(MemoryFilter::default()).await.unwrap();
        assert!(hits2.is_empty());
    }

    #[tokio::test]
    async fn sqlite_replace() {
        let store = SqliteMemory::in_memory().await.unwrap();
        let m = Memory::new_user(UserId("u1".into()), "original");
        let id = store.insert(m.clone()).await.unwrap();

        let updated = Memory {
            id,
            content: "updated".to_string(),
            ..m
        };
        store.replace(id, updated).await.unwrap();

        let hits = store.find(MemoryFilter::default()).await.unwrap();
        assert_eq!(hits[0].content, "updated");
    }

    #[tokio::test]
    async fn sqlite_cross_thread_scope_isolation() {
        let store = SqliteMemory::in_memory().await.unwrap();
        store
            .insert(Memory::new_thread(ThreadId("s1".into()), "s1 data"))
            .await
            .unwrap();
        store
            .insert(Memory::new_thread(ThreadId("s2".into()), "s2 data"))
            .await
            .unwrap();

        let hits = store
            .find(MemoryFilter {
                scope: Some(MemoryScope::Thread {
                    thread_id: ThreadId("s1".into()),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "s1 data");
    }
}
