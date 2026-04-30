use std::time::Duration;

/// ufox-arc 公开 API 返回的错误类型。
#[derive(Debug, thiserror::Error)]
pub enum ArcError {
    #[error("LLM error: {0}")]
    Llm(#[from] ufox_llm::LlmError),

    #[error("tool error: {tool} — {message}")]
    Tool { tool: String, message: String },

    /// 同一会话收到并发写入请求时触发。
    #[error("session is currently busy processing another request")]
    SessionBusy,

    #[error("session error: {0}")]
    Session(String),

    #[error("timeout after {0:?}")]
    Timeout(Duration),

    #[error("max iterations reached: {0}")]
    MaxIterations(usize),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("memory error: {0}")]
    Memory(String),
}
