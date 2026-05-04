use async_trait::async_trait;

use crate::error::ArcError;

use super::{ThreadId, ThreadSnapshot};

/// 线程快照持久化接口。
#[async_trait]
pub trait ThreadStore: Send + Sync {
    /// 将线程完整快照持久化。
    async fn save(&self, snapshot: &ThreadSnapshot) -> Result<(), ArcError>;

    /// 读取线程快照；不存在时返回 `None`。
    async fn load(&self, thread_id: &ThreadId) -> Result<Option<ThreadSnapshot>, ArcError>;

    /// 删除线程快照。
    async fn delete(&self, thread_id: &ThreadId) -> Result<(), ArcError>;
}
