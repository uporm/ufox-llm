use async_trait::async_trait;
use ufox_llm::Message;

use crate::error::ArcError;
use crate::session::SessionId;

/// 会话历史持久化接口。
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// 将整个消息列表持久化（覆盖写）。
    async fn save(&self, session_id: &SessionId, messages: &[Message]) -> Result<(), ArcError>;
    /// 读取会话历史；不存在时返回空列表。
    async fn load(&self, session_id: &SessionId) -> Result<Vec<Message>, ArcError>;
    /// 删除会话历史。
    async fn delete(&self, session_id: &SessionId) -> Result<(), ArcError>;
}
