pub mod cli;

pub use cli::{AutoApproveHandler, CliInterruptHandler};

use async_trait::async_trait;

use crate::error::ArcError;
use crate::thread::{ThreadId, UserId};

/// 触发人工中断的原因。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum InterruptReason {
    ToolConfirm {
        tool: String,
        params: serde_json::Value,
        reason: Option<String>,
    },
    ErrorRecovery {
        error: String,
        proposed_action: String,
    },
    UserBreakpoint {
        condition: String,
    },
}

/// 人工确认后的决策。
#[derive(Debug)]
pub enum InterruptDecision {
    /// 使用原参数继续执行。
    Continue,
    /// 中止当前工具调用，向 Agent 循环返回错误。
    Abort,
    /// 重试（与 `Continue` 等效；为语义完整性保留）。
    Retry,
    /// 使用修改后的参数继续执行。
    ModifyAndContinue(serde_json::Value),
}

/// 人机协同中断处理器。
#[async_trait]
pub trait InterruptHandler: Send + Sync {
    async fn handle_interrupt(
        &self,
        reason: InterruptReason,
        user_id: &UserId,
        thread_id: &ThreadId,
    ) -> Result<InterruptDecision, ArcError>;
}

/// 工具执行时携带的中断上下文；仅包含引用，构造开销极小。
pub struct InterruptCtx<'a> {
    pub handler: &'a dyn InterruptHandler,
    pub user_id: &'a UserId,
    pub thread_id: &'a ThreadId,
}
