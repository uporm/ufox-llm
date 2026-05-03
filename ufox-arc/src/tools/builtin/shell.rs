use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use ufox_llm::ToolResultPayload;

use crate::tools::result::ToolError;
use crate::tools::{Confirm, Tool, ToolSpec};

/// 通过 `sh -c` 执行 Shell 命令。
///
/// 此工具具有破坏性潜力；运行时会根据命令内容动态决定是否进入 HITL。
pub struct ShellTool {
    spec: ToolSpec,
}

impl Default for ShellTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
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
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn confirm(&self, params: &serde_json::Value) -> Confirm {
        let command = params["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "shell".into(),
                message: "missing 'command' parameter".into(),
            })?;
        classify_shell_command(command)
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

fn classify_shell_command(command: &str) -> Result<Option<String>, ToolError> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(ToolError::InvalidParams {
            tool: "shell".into(),
            message: "parameter 'command' must not be empty".into(),
        });
    }

    if let Some(reason) = detect_complex_shell_syntax(trimmed) {
        return Ok(Some(reason.to_string()));
    }

    let tokens = match shell_words::split(trimmed) {
        Ok(tokens) => tokens,
        Err(_) => {
            return Ok(Some("命令包含无法静态解析的 shell 语法".into()));
        }
    };

    if tokens.is_empty() {
        return Err(ToolError::InvalidParams {
            tool: "shell".into(),
            message: "parameter 'command' must not be empty".into(),
        });
    }

    if tokens.iter().any(|token| is_env_assignment_token(token)) {
        return Ok(Some("命令包含环境变量注入前缀，保守起见需要人工确认".into()));
    }

    let program = tokens[0].as_str();
    let reason = match program {
        "ls" | "pwd" | "whoami" | "uname" | "date" | "cat" | "head" | "tail" | "wc"
        | "which" | "echo" | "grep" | "du" | "df" | "ps" | "env" | "id" => None,
        "git" => classify_git_command(&tokens),
        "cargo" => classify_cargo_command(&tokens),
        "rm" => Some("命令包含删除操作".into()),
        "mv" | "cp" | "chmod" | "chown" | "dd" | "mkfs" | "diskutil" | "launchctl"
        | "kill" | "pkill" | "tee" | "touch" | "mkdir" | "rmdir" | "ln" | "install"
        | "sudo" => Some("命令可能修改文件、进程或系统状态".into()),
        _ => Some("未知 shell 命令，保守起见需要人工确认".into()),
    };
    Ok(reason)
}

fn classify_git_command(tokens: &[String]) -> Option<String> {
    let Some(subcommand) = tokens.get(1).map(String::as_str) else {
        return Some("git 子命令不明确，保守起见需要人工确认".into());
    };

    match subcommand {
        "status" | "diff" | "log" | "show" | "rev-parse" => None,
        _ => Some("git 子命令可能修改仓库状态".into()),
    }
}

fn classify_cargo_command(tokens: &[String]) -> Option<String> {
    let Some(subcommand) = tokens.get(1).map(String::as_str) else {
        return Some("cargo 子命令不明确，保守起见需要人工确认".into());
    };

    match subcommand {
        "check" | "test" | "clippy" | "metadata" | "tree" => None,
        _ => Some("cargo 子命令可能修改构建产物或发布状态".into()),
    }
}

fn is_env_assignment_token(token: &str) -> bool {
    let Some((name, _value)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(idx, ch)| ch == '_' || ch.is_ascii_alphanumeric() && (idx > 0 || !ch.is_ascii_digit()))
}

fn detect_complex_shell_syntax(command: &str) -> Option<&'static str> {
    let mut chars = command.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            escaped = false;
            continue;
        }

        match ch {
            '\\' if !in_single => {
                escaped = true;
            }
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            _ if in_single || in_double => {}
            ';' => return Some("命令包含多条子命令"),
            '&' => return Some("命令包含后台或条件执行符"),
            '|' => return Some("命令包含管道或条件执行符"),
            '>' | '<' => return Some("命令包含重定向"),
            '`' => return Some("命令包含命令替换"),
            '\n' | '\r' => return Some("命令包含多行 shell 片段"),
            '$' if chars.peek() == Some(&'(') => return Some("命令包含命令替换"),
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_readonly_commands_skip_confirmation() {
        assert_eq!(classify_shell_command("ls -la").unwrap(), None);
        assert_eq!(classify_shell_command("git status --short").unwrap(), None);
        assert_eq!(classify_shell_command("cargo test").unwrap(), None);
    }

    #[test]
    fn risky_commands_require_confirmation() {
        assert!(classify_shell_command("rm -rf tmp").unwrap().is_some());
        assert!(classify_shell_command("mv a b").unwrap().is_some());
        assert!(classify_shell_command("git reset --hard").unwrap().is_some());
    }

    #[test]
    fn complex_shell_syntax_requires_confirmation() {
        assert!(classify_shell_command("echo hi > out.txt").unwrap().is_some());
        assert!(classify_shell_command("ls | wc -l").unwrap().is_some());
        assert!(classify_shell_command("FOO=bar ls").unwrap().is_some());
        assert!(classify_shell_command("echo $(pwd)").unwrap().is_some());
    }

    #[test]
    fn unparsable_shell_syntax_requires_confirmation() {
        assert!(classify_shell_command("echo \"unterminated").unwrap().is_some());
    }
}
