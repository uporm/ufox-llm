use std::sync::Arc;

use ufox_llm::Client;

use super::config::{AgentConfig};
use super::runtime::Agent;
use crate::error::ArcError;
use crate::interrupt::InterruptHandler;
use crate::memory::MemoryProvider;
use crate::tools::{Tool, ToolManager};

/// `Agent` 构建器。
pub struct AgentBuilder {
    llm: Option<Client>,
    instructions: Option<String>,
    config: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Arc<dyn MemoryProvider>>,
    interrupt_handler: Option<Arc<dyn InterruptHandler>>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            llm: None,
            instructions: None,
            config: AgentConfig::default(),
            tools: Vec::new(),
            memory: None,
            interrupt_handler: None,
        }
    }
}

impl AgentBuilder {
    /// 设置底层 LLM 客户端。
    pub fn llm(mut self, client: Client) -> Self {
        self.llm = Some(client);
        self
    }

    /// 设置 Agent 指令。
    pub fn instructions(mut self, prompt: impl Into<String>) -> Self {
        self.instructions = Some(prompt.into());
        self
    }

    /// 直接设置完整配置。
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// 注册一个工具实例。
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// 设置记忆提供器（同时隐式启用 Perceive 步骤）。
    pub fn memory(mut self, provider: impl MemoryProvider + 'static) -> Self {
        self.memory = Some(Arc::new(provider));
        self
    }

    /// 设置 HITL 中断处理器。
    pub fn interrupt_handler(mut self, handler: impl InterruptHandler + 'static) -> Self {
        self.interrupt_handler = Some(Arc::new(handler));
        self
    }

    /// 构建 `Agent` 实例。
    pub fn build(self) -> Result<Agent, ArcError> {
        let llm = self
            .llm
            .ok_or_else(|| ArcError::Config("llm client is required".into()))?;

        let tools = if self.tools.is_empty() {
            None
        } else {
            let mut manager = ToolManager::new();
            manager.register(self.tools);
            Some(Arc::new(manager))
        };

        Ok(Agent {
            llm: Arc::new(llm),
            instructions: self.instructions,
            config: self.config,
            tools,
            memory: self.memory,
            interrupt_handler: self.interrupt_handler,
        })
    }
}
