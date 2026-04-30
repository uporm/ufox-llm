use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use ufox_llm::ToolResultPayload;

use crate::tools::result::ToolError;
use crate::tools::{Tool, ToolMetadata};

/// 通过 `sh -c` 执行 Shell 命令。
///
/// 此工具具有破坏性潜力，默认 `requires_confirmation = true`，
/// 建议配合 HITL 中断处理器（Phase 6）一起使用。
pub struct ShellTool {
    metadata: ToolMetadata,
}

impl Default for ShellTool {
    fn default() -> Self {
        Self {
            metadata: ToolMetadata {
                name: "shell".to_string(),
                description: "执行 Shell 命令，返回 stdout 和 stderr 的合并输出。".to_string(),
                parameters_schema: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的 Shell 命令"
                        }
                    },
                    "required": ["command"]
                }),
                requires_confirmation: true,
                timeout: Duration::from_secs(30),
            },
        }
    }
}

impl ShellTool {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError> {
        let command = params["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "shell".into(),
                message: "missing 'command' parameter".into(),
            })?;

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "shell".into(),
                message: e.to_string(),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        let result = if stderr.is_empty() {
            format!("exit_code={exit_code}\n{stdout}")
        } else {
            format!("exit_code={exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}")
        };

        Ok(ToolResultPayload::text(result))
    }
}
