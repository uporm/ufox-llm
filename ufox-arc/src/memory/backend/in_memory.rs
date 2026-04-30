use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;

use crate::error::ArcError;
use crate::memory::{Memory, MemoryFilter, MemoryId, MemoryStore};

/// 开发期内存后端；进程退出后数据不保留。
pub struct InMemoryStore {
    data: RwLock<HashMap<MemoryId, Memory>>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn insert(&self, memory: Memory) -> Result<MemoryId, ArcError> {
        let id = memory.id;
        self.data
            .write()
            .map_err(|e| ArcError::Memory(e.to_string()))?
            .insert(id, memory);
        Ok(id)
    }

    async fn find(&self, filter: MemoryFilter) -> Result<Vec<Memory>, ArcError> {
        let data = self
            .data
            .read()
            .map_err(|e| ArcError::Memory(e.to_string()))?;

        let mut results: Vec<Memory> = data
            .values()
            .filter(|m| {
                if let Some(ref scope) = filter.scope
                    && &m.scope != scope
                {
                    return false;
                }
                for tag in &filter.tags {
                    if !m.tags.contains(tag) {
                        return false;
                    }
                }
                if let Some(since) = filter.since
                    && m.timestamp < since
                {
                    return false;
                }
                if let Some(until) = filter.until
                    && m.timestamp > until
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        // 时间倒序
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    async fn replace(&self, id: MemoryId, memory: Memory) -> Result<(), ArcError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| ArcError::Memory(e.to_string()))?;
        if !data.contains_key(&id) {
            return Err(ArcError::Memory(format!("memory {id} not found")));
        }
        data.insert(id, memory);
        Ok(())
    }

    async fn remove(&self, id: MemoryId) -> Result<(), ArcError> {
        self.data
            .write()
            .map_err(|e| ArcError::Memory(e.to_string()))?
            .remove(&id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Memory, MemoryFilter, MemoryScope};
    use crate::session::{SessionId, UserId};

    fn session_memory(session_id: &str, content: &str) -> Memory {
        Memory::new_session(SessionId(session_id.to_string()), content)
    }

    fn user_memory(user_id: &str, content: &str) -> Memory {
        Memory::new_user(UserId(user_id.to_string()), content)
    }

    #[tokio::test]
    async fn insert_and_find_by_scope() {
        let store = InMemoryStore::new();
        let m = session_memory("s1", "hello session");
        store.insert(m.clone()).await.unwrap();

        let hits = store
            .find(MemoryFilter {
                scope: Some(MemoryScope::Session {
                    session_id: SessionId("s1".into()),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "hello session");
    }

    #[tokio::test]
    async fn scope_isolation() {
        let store = InMemoryStore::new();
        store
            .insert(session_memory("s1", "session data"))
            .await
            .unwrap();
        store.insert(user_memory("u1", "user data")).await.unwrap();

        let session_hits = store
            .find(MemoryFilter {
                scope: Some(MemoryScope::Session {
                    session_id: SessionId("s1".into()),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(session_hits.len(), 1);
        assert_eq!(session_hits[0].content, "session data");
    }

    #[tokio::test]
    async fn filter_by_tags() {
        let store = InMemoryStore::new();
        let m1 = session_memory("s1", "tagged").with_tags(["rust", "ai"]);
        let m2 = session_memory("s1", "not tagged");
        store.insert(m1).await.unwrap();
        store.insert(m2).await.unwrap();

        let hits = store
            .find(MemoryFilter {
                tags: vec!["rust".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "tagged");
    }

    #[tokio::test]
    async fn remove_memory() {
        let store = InMemoryStore::new();
        let m = session_memory("s1", "to be removed");
        let id = store.insert(m).await.unwrap();

        store.remove(id).await.unwrap();

        let hits = store.find(MemoryFilter::default()).await.unwrap();
        assert!(hits.is_empty());
    }
}
