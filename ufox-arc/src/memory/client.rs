use std::sync::Arc;

use crate::error::ArcError;
use crate::thread::{ThreadId, UserId};
use super::{Memory, MemoryFilter, MemoryId, MemoryProvider, MemoryScope};

/// 记忆访问门面，统一封装常见的用户级 / 线程级记忆操作。
#[derive(Clone)]
pub struct MemoryClient {
    provider: Arc<dyn MemoryProvider>,
}

impl MemoryClient {
    pub(crate) fn new(provider: Arc<dyn MemoryProvider>) -> Self {
        Self { provider }
    }

    pub async fn remember_user(
        &self,
        user_id: impl Into<UserId>,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, ArcError> {
        let memory = Memory::new_user(user_id.into(), content).with_tags(tags);
        self.provider.insert(memory).await
    }

    pub async fn remember_thread(
        &self,
        thread_id: impl Into<ThreadId>,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, ArcError> {
        let memory = Memory::new_thread(thread_id.into(), content).with_tags(tags);
        self.provider.insert(memory).await
    }

    pub async fn find(&self, filter: MemoryFilter) -> Result<Vec<Memory>, ArcError> {
        self.provider.find(filter).await
    }

    pub async fn user_memories(
        &self,
        user_id: impl Into<UserId>,
    ) -> Result<Vec<Memory>, ArcError> {
        self.find(MemoryFilter {
            scope: Some(MemoryScope::User {
                user_id: user_id.into(),
            }),
            ..Default::default()
        })
        .await
    }

    pub async fn thread_memories(
        &self,
        thread_id: impl Into<ThreadId>,
    ) -> Result<Vec<Memory>, ArcError> {
        self.find(MemoryFilter {
            scope: Some(MemoryScope::Thread {
                thread_id: thread_id.into(),
            }),
            ..Default::default()
        })
        .await
    }

}
