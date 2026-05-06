pub mod client;
pub mod backend;
pub mod strategy;

pub use client::MemoryClient;
pub use backend::in_memory::InMemoryBackend;
pub use backend::sqlite::SqliteBackend;

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::ArcError;
use crate::thread::{ThreadId, UserId};

pub type MemoryId = Uuid;

/// 记忆的作用域：线程级（临时上下文）或用户级（长期偏好）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryScope {
    Thread { thread_id: ThreadId },
    User { user_id: UserId },
}

/// 单条记忆记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub scope: MemoryScope,
    /// 记忆的文本内容。
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub tags: Vec<String>,
}

impl Memory {
    pub fn new_thread(thread_id: ThreadId, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            scope: MemoryScope::Thread { thread_id },
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tags: Vec::new(),
        }
    }

    pub fn new_user(user_id: UserId, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            scope: MemoryScope::User { user_id },
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tags: Vec::new(),
        }
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into()).collect();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// 查询过滤条件；所有字段均可选，组合使用。
#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    /// 限定作用域；`None` 表示不过滤。
    pub scope: Option<MemoryScope>,
    /// 必须包含所有列出的标签（AND 语义）。
    pub tags: Vec<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    /// 最多返回条数。
    pub limit: Option<usize>,
}

/// 记忆提供器统一接口；通过 `MemoryScope` 区分会话/用户两个层级。
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    async fn insert(&self, memory: Memory) -> Result<MemoryId, ArcError>;
    async fn find(&self, filter: MemoryFilter) -> Result<Vec<Memory>, ArcError>;
    async fn replace(&self, id: MemoryId, memory: Memory) -> Result<(), ArcError>;
    async fn remove(&self, id: MemoryId) -> Result<(), ArcError>;
}
