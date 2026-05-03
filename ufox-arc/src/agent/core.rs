use std::sync::Arc;

use ufox_llm::Client;

use super::builder::AgentBuilder;
use super::config::AgentConfig;
use crate::error::ArcError;
use crate::interrupt::InterruptHandler;
use crate::memory::MemoryStore;
use crate::session::{Session, SessionId, UserId};
use crate::tools::ToolManager;

/// AI Agent 核心结构体。
///
/// 持有 LLM 客户端、工具管理器和配置，并通过会话执行实际推理。
#[derive(Clone)]
pub struct Agent {
    pub(crate) llm: Arc<Client>,
    pub(crate) system: Option<String>,
    pub(crate) config: AgentConfig,
    pub(crate) tools: Option<Arc<ToolManager>>,
    pub(crate) memory: Option<Arc<dyn MemoryStore>>,
    pub(crate) interrupt_handler: Option<Arc<dyn InterruptHandler>>,
}

impl Agent {
    /// 返回 `Agent` 构建器。
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// 基于给定用户和会话标识创建会话句柄。
    pub async fn session(
        &self,
        user_id: impl Into<UserId>,
        session_id: impl Into<SessionId>,
    ) -> Result<Session, ArcError> {
        Ok(Session::new(
            user_id.into(),
            session_id.into(),
            Arc::new(self.clone()),
        ))
    }

    /// 为指定用户创建一个新的随机会话。
    pub async fn new_session(&self, user_id: impl Into<UserId>) -> Result<Session, ArcError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.session(user_id, session_id).await
    }
}
