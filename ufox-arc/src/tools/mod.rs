pub mod builtin;
pub mod manager;

use std::time::Duration;

use async_trait::async_trait;
use ufox_llm::ToolResultPayload;

pub(crate) use manager::ToolManager;

/// 工具确认钩子的返回类型；`Some(reason)` 表示需要人工确认。
pub type Confirm = Result<Option<String>, ToolError>;

/// 工具执行过程中产生的错误。
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// 工具管理器中不存在指定名称的工具。
    #[error("tool '{name}' not found in manager")]
    NotFound { name: String },

    /// 工具参数不符合预期约束。
    #[error("invalid parameters for '{tool}': {message}")]
    InvalidParams { tool: String, message: String },

    /// 工具执行失败。
    #[error("execution failed for '{tool}': {message}")]
    ExecutionFailed { tool: String, message: String },

    /// 工具执行超时。
    #[error("tool '{tool}' timed out")]
    Timeout { tool: String },
}

/// 工具的静态规格；注册时确定，执行时不可变。
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// 工具的唯一名称。
    pub name: String,
    /// 展示给模型的工具描述。
    pub description: String,
    /// 工具参数的 JSON Schema。
    pub parameters_schema: serde_json::Value,
    /// 单次执行的超时时间。
    pub timeout: Duration,
}

/// 可供 Agent 调用的工具。
#[async_trait]
pub trait Tool: Send + Sync {
    /// 返回工具的静态规格。
    fn spec(&self) -> &ToolSpec;

    /// 根据本次参数决定是否需要人工确认，并可返回触发原因。
    fn confirm(&self, _params: &serde_json::Value) -> Confirm {
        Ok(None)
    }

    /// 执行工具并返回标准化结果载荷。
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError>;
}
