use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use ufox_llm::{ToolCall, ToolResult};

use crate::error::ArcError;
use crate::interrupt::{InterruptCtx, InterruptDecision, InterruptReason};
use crate::tools::Tool;

/// Agent 运行时使用的工具管理器；对外装配入口统一收敛到 `AgentBuilder`。
#[derive(Default)]
pub(crate) struct ToolManager {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolManager {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 批量注册共享工具实例；重名时保留首个注册项并记录警告。
    pub(crate) fn register(&mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) {
        for tool in tools {
            let name = tool.spec().name.clone();
            if let Entry::Vacant(e) = self.tools.entry(name.clone()) {
                e.insert(tool);
            } else {
                tracing::warn!(tool = %name, "tool is already registered; skipping duplicate");
            }
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
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
        let name = &tool_call.tool_name;
        let tool = self.get(name).ok_or_else(|| ArcError::Tool {
            tool: name.clone(),
            message: "not registered".into(),
        })?;

        let args = Self::resolve_args(&tool, name, tool_call.arguments.clone(), interrupt).await?;

        let timeout = tool.spec().timeout;
        let payload = tokio::time::timeout(timeout, tool.execute(args))
            .await
            .map_err(|_| ArcError::Tool {
                tool: name.clone(),
                message: "execution timed out".into(),
            })?
            .map_err(|e| ArcError::Tool {
                tool: name.clone(),
                message: e.to_string(),
            })?;

        Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            tool_name: Some(name.clone()),
            payload,
            is_error: false,
        })
    }

    /// 处理参数校验与人工确认 (HITL) 的循环逻辑
    async fn resolve_args<'a>(
        tool: &Arc<dyn Tool>,
        tool_name: &str,
        mut args: serde_json::Value,
        interrupt: Option<InterruptCtx<'a>>,
    ) -> Result<serde_json::Value, ArcError> {
        loop {
            validate_required_params(&tool.spec().parameters_schema, &args, tool_name)?;

            let Some(reason) = tool.confirm(&args).map_err(|e| ArcError::Tool {
                tool: tool_name.to_string(),
                message: e.to_string(),
            })?
            else {
                return Ok(args);
            };

            let Some(ref ctx) = interrupt else {
                tracing::warn!(
                    tool = %tool_name,
                    reason,
                    "tool requires confirmation but no interrupt handler configured; proceeding"
                );
                return Ok(args);
            };

            let decision = ctx
                .handler
                .handle_interrupt(
                    InterruptReason::ToolConfirm {
                        tool: tool_name.to_string(),
                        params: args.clone(),
                        reason: Some(reason),
                    },
                    ctx.user_id,
                    ctx.thread_id,
                )
                .await?;

            match decision {
                InterruptDecision::Continue | InterruptDecision::Retry => return Ok(args),
                InterruptDecision::ModifyAndContinue(new_args) => {
                    tracing::info!(tool = %tool_name, "user modified tool arguments");
                    args = new_args;
                }
                InterruptDecision::Abort => {
                    return Err(ArcError::Tool {
                        tool: tool_name.to_string(),
                        message: "execution aborted by user".into(),
                    });
                }
            }
        }
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

    for field in required.iter().filter_map(|f| f.as_str()) {
        if params.get(field).is_none() {
            return Err(ArcError::Tool {
                tool: tool.to_string(),
                message: format!("missing required parameter: '{field}'"),
            });
        }
    }

    Ok(())
}
