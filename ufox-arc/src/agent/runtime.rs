use std::sync::Arc;

use ufox_llm::Client;

use super::builder::AgentBuilder;
use super::config::AgentConfig;
use crate::error::ArcError;
use crate::interrupt::InterruptHandler;
use crate::memory::{MemoryClient, MemoryProvider};
use crate::run::{RunEventStream, RunInput, RunResult, run_once, run_stream};
use crate::thread::{Thread, ThreadId, UserId};
use crate::tools::ToolManager;

/// AI Agent 核心结构体。
///
/// 持有 LLM 客户端、工具管理器和配置，并通过会话执行实际推理。
#[derive(Clone)]
pub struct Agent {
    pub(crate) llm: Arc<Client>,
    pub(crate) instructions: Option<String>,
    pub(crate) config: AgentConfig,
    pub(crate) tools: Option<Arc<ToolManager>>,
    pub(crate) memory: Option<Arc<dyn MemoryProvider>>,
    pub(crate) interrupt_handler: Option<Arc<dyn InterruptHandler>>,
}

impl Agent {
    /// 返回 `Agent` 构建器。
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// 基于给定用户和线程标识创建线程句柄。
    pub fn thread(&self, user_id: impl Into<UserId>, thread_id: impl Into<ThreadId>) -> Thread {
        Thread::new(user_id.into(), thread_id.into())
    }

    /// 为指定用户创建一个新的随机线程。
    pub fn new_thread(&self, user_id: impl Into<UserId>) -> Thread {
        self.thread(user_id, ThreadId::new())
    }

    /// 返回记忆门面；若未配置记忆提供器则返回错误。
    pub fn memory(&self) -> Result<MemoryClient, ArcError> {
        let provider = self
            .memory
            .as_ref()
            .cloned()
            .ok_or_else(|| ArcError::Memory("no memory provider configured".into()))?;
        Ok(MemoryClient::new(provider))
    }

    /// 在给定线程上执行一次运行。
    ///
    /// 本次运行使用当前 `Agent` 持有的 `AgentConfig`，不再支持请求级覆盖。
    pub async fn run<I>(&self, thread: &Thread, input: I) -> Result<RunResult, ArcError>
    where
        I: Into<RunInput>,
    {
        run_once(self, thread, input.into()).await
    }

    /// 在给定线程上执行一次流式运行。
    ///
    /// 本次运行使用当前 `Agent` 持有的 `AgentConfig`，不再支持请求级覆盖。
    pub async fn run_stream<I>(
        &self,
        thread: &Thread,
        input: I,
    ) -> Result<RunEventStream, ArcError>
    where
        I: Into<RunInput>,
    {
        run_stream(self, thread, input.into()).await
    }
}
