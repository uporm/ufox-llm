use async_trait::async_trait;

use crate::error::ArcError;
use crate::interrupt::{InterruptDecision, InterruptHandler, InterruptReason};
use crate::session::{SessionId, UserId};

/// CLI 版中断处理器：将确认提示打印到 stdout，从 stdin 读取用户决策。
///
/// 支持以下输入（大小写不敏感）：
/// - `y` / `yes` / 直接回车 → `Continue`
/// - `n` / `no`              → `Abort`
/// - `m <JSON>`              → `ModifyAndContinue(JSON)`（修改参数后继续）
///
/// 在非交互终端（CI、重定向 stdin）中，空行与回车均视为 `Continue`。
#[derive(Debug, Default)]
pub struct CliInterruptHandler;

#[async_trait]
impl InterruptHandler for CliInterruptHandler {
    async fn handle_interrupt(
        &self,
        reason: InterruptReason,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<InterruptDecision, ArcError> {
        let prompt = format_prompt(&reason, user_id, session_id);

        let input = tokio::task::spawn_blocking(move || {
            use std::io::{BufRead, Write};
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            let _ = writeln!(out, "\n{prompt}");
            let _ = out.flush();

            let stdin = std::io::stdin();
            let mut line = String::new();
            stdin.lock().read_line(&mut line).ok();
            line.trim().to_string()
        })
        .await
        .map_err(|e| ArcError::Session(format!("CLI interrupt handler error: {e}")))?;

        Ok(parse_decision(&input))
    }
}

/// 自动批准处理器：始终返回 `Continue`，适用于测试与非交互场景。
#[derive(Debug, Default)]
pub struct AutoApproveHandler;

#[async_trait]
impl InterruptHandler for AutoApproveHandler {
    async fn handle_interrupt(
        &self,
        _reason: InterruptReason,
        _user_id: &UserId,
        _session_id: &SessionId,
    ) -> Result<InterruptDecision, ArcError> {
        Ok(InterruptDecision::Continue)
    }
}

fn format_prompt(reason: &InterruptReason, user_id: &UserId, session_id: &SessionId) -> String {
    let header = format!("[HITL] user={} session={}", user_id, session_id);
    let body = match reason {
        InterruptReason::ToolConfirmation { tool, params } => {
            format!(
                "Tool confirmation required\n  tool    : {tool}\n  params  : {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            )
        }
        InterruptReason::ErrorRecovery {
            error,
            proposed_action,
        } => {
            format!("Error recovery\n  error   : {error}\n  proposed: {proposed_action}")
        }
        InterruptReason::UserBreakpoint { condition } => {
            format!("User breakpoint\n  condition: {condition}")
        }
    };
    format!("{header}\n{body}\nContinue? [Y/n/m <json>]: ")
}

fn parse_decision(input: &str) -> InterruptDecision {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    match lower.as_str() {
        "" | "y" | "yes" => InterruptDecision::Continue,
        "n" | "no" => InterruptDecision::Abort,
        s if s.starts_with("m ") || s.starts_with("modify ") => {
            // Extract JSON from the *original* trimmed input to preserve casing
            let json_part = trimmed.split_once(' ').map(|x| x.1).unwrap_or("{}");
            match serde_json::from_str(json_part) {
                Ok(v) => InterruptDecision::ModifyAndContinue(v),
                Err(_) => {
                    eprintln!("[HITL] invalid JSON, proceeding with original params");
                    InterruptDecision::Continue
                }
            }
        }
        _ => InterruptDecision::Continue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_yes_variants() {
        assert!(matches!(parse_decision(""), InterruptDecision::Continue));
        assert!(matches!(parse_decision("y"), InterruptDecision::Continue));
        assert!(matches!(parse_decision("Y"), InterruptDecision::Continue));
        assert!(matches!(parse_decision("yes"), InterruptDecision::Continue));
    }

    #[test]
    fn parse_abort() {
        assert!(matches!(parse_decision("n"), InterruptDecision::Abort));
        assert!(matches!(parse_decision("no"), InterruptDecision::Abort));
    }

    #[test]
    fn parse_modify() {
        let input = r#"m {"city": "Shanghai"}"#;
        if let InterruptDecision::ModifyAndContinue(v) = parse_decision(input) {
            assert_eq!(v["city"], "Shanghai");
        } else {
            panic!("expected ModifyAndContinue");
        }
    }

    #[tokio::test]
    async fn auto_approve_always_continues() {
        let handler = AutoApproveHandler;
        let user_id = UserId("u1".into());
        let session_id = SessionId("s1".into());
        let decision = handler
            .handle_interrupt(
                InterruptReason::ToolConfirmation {
                    tool: "shell".into(),
                    params: serde_json::json!({"command": "rm -rf /"}),
                },
                &user_id,
                &session_id,
            )
            .await
            .unwrap();
        assert!(matches!(decision, InterruptDecision::Continue));
    }
}
