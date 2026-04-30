/// 工具执行过程中产生的错误。
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool '{name}' not found in registry")]
    NotFound { name: String },

    #[error("invalid parameters for '{tool}': {message}")]
    InvalidParams { tool: String, message: String },

    #[error("execution failed for '{tool}': {message}")]
    ExecutionFailed { tool: String, message: String },

    #[error("tool '{tool}' timed out")]
    Timeout { tool: String },
}
