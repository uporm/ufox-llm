use std::sync::Arc;

use ufox_llm::Client;

use super::config::AgentConfig;
use super::core::Agent;
use crate::error::ArcError;
use crate::interrupt::InterruptHandler;
use crate::memory::MemoryStore;
use crate::tools::{Tool, ToolManager};

/// `Agent` 构建器。
#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Client>,
    instructions: Option<String>,
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
        self.config = Some(config);
        self
    }

    /// 覆盖最大推理轮数。
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// 启用或关闭 Perceive 步骤。
    pub fn enable_perceive(mut self, v: bool) -> Self {
        self.enable_perceive = Some(v);
        self
    }

    /// 启用或关闭 Observe 步骤。
    pub fn enable_observe(mut self, v: bool) -> Self {
        self.enable_observe = Some(v);
        self
    }

    /// 启用或关闭 Reflect 步骤。
    pub fn enable_reflect(mut self, v: bool) -> Self {
        self.enable_reflect = Some(v);
        self
    }

    /// 注册一个工具实例。
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// 设置记忆存储。
    pub fn memory(mut self, store: impl MemoryStore + 'static) -> Self {
        self.memory = Some(Arc::new(store));
        self
    }

    /// 设置 HITL 中断处理器。
    ///
    /// 工具确认策略要求人工确认时，将调用此处理器。
    pub fn interrupt_handler(mut self, handler: impl InterruptHandler + 'static) -> Self {
        self.interrupt_handler = Some(Arc::new(handler));
        self
    }

    /// 构建 `Agent` 实例。
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
            manager.register(self.tools);
            Some(Arc::new(manager))
        };

        Ok(Agent {
            llm: Arc::new(llm),
            instructions: self.instructions,
            config,
            tools,
            memory: self.memory,
            interrupt_handler: self.interrupt_handler,
        })
    }
}
