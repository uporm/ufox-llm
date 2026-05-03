mod config;
pub(crate) mod loop_;
pub mod step;

pub use config::AgentConfig;
pub use step::{ExecutionState, ExecutionStep, Memory, StepInput, StepKind, StepOutput};

use std::sync::Arc;

use ufox_llm::Client;

use crate::error::ArcError;
use crate::interrupt::InterruptHandler;
use crate::memory::MemoryStore;
use crate::session::{Session, SessionId, UserId};
use crate::tools::{Tool, ToolManager};

/// AI Agent 核心结构体。持有 LLM 客户端、工具管理器和配置，
/// 通过会话执行实际推理。多个会话通过 `Arc` 共享同一个 Agent。
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
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

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

    pub async fn new_session(&self, user_id: impl Into<UserId>) -> Result<Session, ArcError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.session(user_id, session_id).await
    }
}

/// `Agent` 构建器。
#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Client>,
    system: Option<String>,
    config: Option<AgentConfig>,
    max_iterations: Option<usize>,
    enable_perceive: Option<bool>,
    enable_observe: Option<bool>,
    enable_reflect: Option<bool>,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Arc<dyn MemoryStore>>,
    interrupt_handler: Option<Arc<dyn InterruptHandler>>,
}

impl AgentBuilder {
    pub fn llm(mut self, client: Client) -> Self {
        self.llm = Some(client);
        self
    }

    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    pub fn enable_perceive(mut self, v: bool) -> Self {
        self.enable_perceive = Some(v);
        self
    }

    pub fn enable_observe(mut self, v: bool) -> Self {
        self.enable_observe = Some(v);
        self
    }

    pub fn enable_reflect(mut self, v: bool) -> Self {
        self.enable_reflect = Some(v);
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn memory(mut self, store: impl MemoryStore + 'static) -> Self {
        self.memory = Some(Arc::new(store));
        self
    }

    /// 设置 HITL 中断处理器；工具确认策略要求人工确认时将调用此处理器。
    pub fn interrupt_handler(mut self, handler: impl InterruptHandler + 'static) -> Self {
        self.interrupt_handler = Some(Arc::new(handler));
        self
    }

    pub fn build(self) -> Result<Agent, ArcError> {
        let llm = self
            .llm
            .ok_or_else(|| ArcError::Config("llm client is required".into()))?;

        let mut config = self.config.unwrap_or_default();
        if let Some(n) = self.max_iterations {
            config.max_iterations = n;
        }
        if let Some(v) = self.enable_perceive {
            config.enable_perceive = v;
        }
        if let Some(v) = self.enable_observe {
            config.enable_observe = v;
        }
        if let Some(v) = self.enable_reflect {
            config.enable_reflect = v;
        }

        let tools = if self.tools.is_empty() {
            None
        } else {
            let mut manager = ToolManager::new();
            manager.register(self.tools)?;
            Some(Arc::new(manager))
        };

        Ok(Agent {
            llm: Arc::new(llm),
            system: self.system,
            config,
            tools,
            memory: self.memory,
            interrupt_handler: self.interrupt_handler,
        })
    }
}
