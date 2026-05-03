use std::path::{Component, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use ufox_llm::ToolResultPayload;

use crate::tools::{Confirm, Tool, ToolError, ToolSpec};

/// 以文本模式读取指定路径文件。
pub struct FileReadTool {
    spec: ToolSpec,
}

impl Default for FileReadTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "file_read".to_string(),
                description: "读取指定路径的文件内容，以文本形式返回。".to_string(),
                parameters_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要读取的文件路径"
                        }
                    },
                    "required": ["path"]
                }),
                timeout: Duration::from_secs(10),
            },
        }
    }
}

impl FileReadTool {
    /// 创建文件读取工具。
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError> {
        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "file_read".into(),
                message: "missing 'path' parameter".into(),
            })?;

        let path = safe_path(path_str, "file_read")?;
        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: "file_read".into(),
                    message: e.to_string(),
                })?;
        Ok(ToolResultPayload::text(content))
    }
}

/// 将文本内容写入指定路径文件（文件不存在时自动创建）。
pub struct FileWriteTool {
    spec: ToolSpec,
}

impl Default for FileWriteTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: "file_write".to_string(),
                description: "将给定文本内容写入指定路径文件，文件不存在时自动创建。".to_string(),
                parameters_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要写入的文件路径"
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入的文件内容"
                        }
                    },
                    "required": ["path", "content"]
                }),
                timeout: Duration::from_secs(10),
            },
        }
    }
}

impl FileWriteTool {
    /// 创建文件写入工具。
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn confirm(&self, _params: &serde_json::Value) -> Confirm {
        Ok(Some("写文件属于破坏性操作，默认需要人工确认".into()))
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResultPayload, ToolError> {
        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "file_write".into(),
                message: "missing 'path' parameter".into(),
            })?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "file_write".into(),
                message: "missing 'content' parameter".into(),
            })?;

        let path = safe_path(path_str, "file_write")?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: "file_write".into(),
                    message: e.to_string(),
                })?;
        }
        tokio::fs::write(&path, content)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "file_write".into(),
                message: e.to_string(),
            })?;
        Ok(ToolResultPayload::text(format!(
            "Successfully wrote {} bytes to '{}'.",
            content.len(),
            path.display()
        )))
    }
}

/// 拒绝含 `..` 分量的路径，防止目录穿越。
fn safe_path(path_str: &str, tool: &str) -> Result<PathBuf, ToolError> {
    let path = PathBuf::from(path_str);
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(ToolError::ExecutionFailed {
                tool: tool.to_string(),
                message: "path traversal ('..') is not allowed".into(),
            });
        }
    }
    Ok(path)
}
