pub mod builtin;
pub mod result;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ufox_llm::{ToolCall, ToolResult, ToolResultPayload};

use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptDecision, InterruptReason};
pub use result::ToolError;

/// 工具确认钩子的返回类型；`Some(reason)` 表示需要人工确认。
pub type Confirm = Result<Option<String>, ToolError>;

/// 工具的静态规格；注册时确定，执行时不可变。
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// 工具参数的 JSON Schema。
    pub parameters_schema: serde_json::Value,
    /// 单次执行的超时时间。
    pub timeout: Duration,
}

/// 可供 Agent 调用的工具。
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    /// 根据本次参数决定是否需要人工确认，并可返回触发原因。
    fn confirm(&self, _params: &serde_json::Value) -> Confirm {
        Ok(None)
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError>;
}

/// Agent 运行时使用的工具管理器；对外装配入口统一收敛到 `AgentBuilder`。
pub(crate) struct ToolManager {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl Default for ToolManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolManager {
    pub(crate) fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// 批量注册共享工具实例；重名时保留首个注册项并记录警告。
    pub(crate) fn register(
        &mut self,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Result<(), ArcError> {
        let mut pending = Vec::new();
        let mut names = HashSet::new();

        for tool in tools {
            let name = tool.spec().name.clone();
            if self.tools.contains_key(&name) || !names.insert(name.clone()) {
                // 保留首个同名工具，避免后注册项悄悄覆盖既有行为。
                tracing::warn!(tool = %name, "tool is already registered; skipping duplicate");
                continue;
            }
            pending.push((name, tool));
        }

        for (name, tool) in pending {
            self.tools.insert(name, tool);
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub(crate) fn to_llm_tools(&self) -> Vec<ufox_llm::Tool> {
        self.tools
            .values()
            .map(|t| {
                let m = t.spec();
                ufox_llm::Tool {
                    name: m.name.clone(),
                    description: m.description.clone(),
                    input_schema: m.parameters_schema.clone(),
                }
            })
            .collect()
    }

    /// 执行工具调用：参数校验 → HITL 确认（可选）→ 带超时执行 → 标准化返回。
    ///
    /// 当 `interrupt` 存在且工具返回需要确认的原因时：
    /// - `Continue` / `Retry` → 使用当前参数执行
    /// - `ModifyAndContinue(new_params)` → 使用修改后的参数重新评估是否需要确认
    /// - `Abort` → 返回 `ArcError::Tool { message: "aborted by user" }`
    #[tracing::instrument(name = "tool.execute", skip(self, interrupt), fields(tool = %tool_call.tool_name))]
    pub(crate) async fn execute<'a>(
        &self,
        tool_call: &ToolCall,
        interrupt: Option<InterruptCtx<'a>>,
    ) -> Result<ToolResult, ArcError> {
        let tool = self
            .get(&tool_call.tool_name)
            .ok_or_else(|| ArcError::Tool {
                tool: tool_call.tool_name.clone(),
                message: "not registered".into(),
            })?;

        // 允许工具根据本次参数动态决定是否需要 HITL，并在用户修改参数后重新评估。
        let mut args = tool_call.arguments.clone();
        loop {
            validate_required_params(&tool.spec().parameters_schema, &args, &tool_call.tool_name)?;

            let reason = tool
                .confirm(&args)
                .map_err(|e| ArcError::Tool {
                    tool: tool_call.tool_name.clone(),
                    message: e.to_string(),
                })?;

            if let Some(reason) = reason {
                if let Some(ref ctx) = interrupt {
                    let decision = ctx
                        .handler
                        .handle_interrupt(
                            InterruptReason::ToolConfirm {
                                tool: tool_call.tool_name.clone(),
                                params: args.clone(),
                                reason: Some(reason),
                            },
                            ctx.user_id,
                            ctx.session_id,
                        )
                        .await?;

                    match decision {
                        InterruptDecision::Continue | InterruptDecision::Retry => break,
                        InterruptDecision::ModifyAndContinue(new_args) => {
                            tracing::info!(
                                tool = %tool_call.tool_name,
                                "user modified tool arguments"
                            );
                            args = new_args;
                        }
                        InterruptDecision::Abort => {
                            return Err(ArcError::Tool {
                                tool: tool_call.tool_name.clone(),
                                message: "execution aborted by user".into(),
                            });
                        }
                    }
                } else {
                    tracing::warn!(
                        tool = %tool_call.tool_name,
                        reason,
                        "tool requires confirmation but no interrupt handler configured; proceeding"
                    );
                    break;
                }
            } else {
                break;
            }
        }

        let timeout = tool.spec().timeout;
        let payload = tokio::time::timeout(timeout, tool.execute(args))
            .await
            .map_err(|_| ArcError::Tool {
                tool: tool_call.tool_name.clone(),
                message: "execution timed out".into(),
            })?
            .map_err(|e| ArcError::Tool {
                tool: tool_call.tool_name.clone(),
                message: e.to_string(),
            })?;

        Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            tool_name: Some(tool_call.tool_name.clone()),
            payload,
            is_error: false,
        })
    }
}

fn validate_required_params(
    schema: &serde_json::Value,
    params: &serde_json::Value,
    tool: &str,
) -> Result<(), ArcError> {
    let Some(required) = schema.get("required").and_then(|r| r.as_array()) else {
        return Ok(());
    };
    for field in required {
        if let Some(field_name) = field.as_str()
            && params.get(field_name).is_none()
        {
            return Err(ArcError::Tool {
                tool: tool.to_string(),
                message: format!("missing required parameter: '{field_name}'"),
            });
        }
    }
    Ok(())
}
