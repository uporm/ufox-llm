pub mod builtin;
pub mod result;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ufox_llm::{ToolCall, ToolResult, ToolResultPayload};

use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptDecision, InterruptReason};
pub use result::ToolError;

/// 工具的静态描述信息；注册时确定，执行时不可变。
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    /// 工具参数的 JSON Schema。
    pub parameters_schema: serde_json::Value,
    /// 是否需要人工确认后才能执行（HITL Phase 6）。
    pub requires_confirmation: bool,
    /// 单次执行的超时时间。
    pub timeout: Duration,
}

/// 可供 Agent 调用的工具。
#[async_trait]
pub trait Tool: Send + Sync {
    fn metadata(&self) -> &ToolMetadata;
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError>;
}

/// 工具注册表；持有所有可用工具，并负责执行调度。
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) -> Result<(), ArcError> {
        let name = tool.metadata().name.clone();
        if self.tools.contains_key(&name) {
            return Err(ArcError::Config(format!(
                "tool '{name}' is already registered"
            )));
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    pub(crate) fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.metadata().name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn to_llm_tools(&self) -> Vec<ufox_llm::Tool> {
        self.tools
            .values()
            .map(|t| {
                let m = t.metadata();
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
    /// 当 `interrupt` 存在且工具 `requires_confirmation` 为 true 时：
    /// - `Continue` / `Retry` → 使用原参数执行
    /// - `ModifyAndContinue(new_params)` → 使用修改后的参数执行
    /// - `Abort` → 返回 `ArcError::Tool { message: "aborted by user" }`
    #[tracing::instrument(name = "tool.execute", skip(self, interrupt), fields(tool = %tool_call.tool_name))]
    pub async fn execute<'a>(
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

        validate_required_params(
            &tool.metadata().parameters_schema,
            &tool_call.arguments,
            &tool_call.tool_name,
        )?;

        // HITL 确认
        let args = if tool.metadata().requires_confirmation {
            if let Some(ctx) = interrupt {
                let decision = ctx
                    .handler
                    .handle_interrupt(
                        InterruptReason::ToolConfirmation {
                            tool: tool_call.tool_name.clone(),
                            params: tool_call.arguments.clone(),
                        },
                        ctx.user_id,
                        ctx.session_id,
                    )
                    .await?;

                match decision {
                    InterruptDecision::Continue | InterruptDecision::Retry => {
                        tool_call.arguments.clone()
                    }
                    InterruptDecision::ModifyAndContinue(new_args) => {
                        tracing::info!(
                            tool = %tool_call.tool_name,
                            "user modified tool arguments"
                        );
                        new_args
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
                    "requires_confirmation is set but no interrupt handler configured; proceeding"
                );
                tool_call.arguments.clone()
            }
        } else {
            tool_call.arguments.clone()
        };

        let timeout = tool.metadata().timeout;
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
