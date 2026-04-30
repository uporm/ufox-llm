/// 演示 Phase 6 HITL：ShellTool 触发人工确认，CliInterruptHandler 从 stdin 读取决策。
///
/// 运行方式：
///   cargo run -p ufox-arc --example hitl_agent
///
/// 提示出现时可输入：
///   y / 回车    → 继续执行原命令
///   n           → 中止
///   m {"cmd":"echo safe"} → 用修改后的参数执行
use anyhow::Result;
use ufox_arc::tools::builtin::ShellTool;
use ufox_arc::{Agent, AgentConfig, AutoApproveHandler, CliInterruptHandler};
use ufox_llm::Client;

fn make_agent_with_handler<H>(handler: H) -> Result<Agent>
where
    H: ufox_arc::InterruptHandler + 'static,
{
    let llm = ufox_arc::tools::builtin::ShellTool::new(); // just to confirm import
    let _ = llm;

    // 从环境变量读取 LLM 配置
    let llm = Client::from_env()?;

    let agent = Agent::builder()
        .llm(llm)
        .system(
            "You are a helpful assistant. When the user asks to run a command, use the shell tool.",
        )
        .config(AgentConfig {
            max_iterations: 5,
            ..Default::default()
        })
        .tool(ShellTool::new())
        .interrupt_handler(handler)
        .build()?;

    Ok(agent)
}

#[tokio::main]
async fn main() -> Result<()> {
    // 非交互演示：AutoApproveHandler 自动批准，无需终端输入
    demo_auto_approve().await?;

    println!("\n--- 如需交互式 HITL，取消注释下方代码 ---");
    // demo_cli_interrupt().await?;

    Ok(())
}

async fn demo_auto_approve() -> Result<()> {
    println!("=== HITL Demo: AutoApproveHandler (自动批准) ===\n");

    let agent = make_agent_with_handler(AutoApproveHandler)?;
    let mut session = agent.new_session("user-demo").await?;

    let result = session
        .chat("请用 shell 工具执行命令 `echo hello from hitl`")
        .await?;

    println!("响应：{}", result.response.text);
    println!("\n执行步骤：");
    for step in &result.trace.steps {
        println!("  [{:?}] {:?}", step.kind, step.duration);
    }

    Ok(())
}

#[allow(dead_code)]
async fn demo_cli_interrupt() -> Result<()> {
    println!("=== HITL Demo: CliInterruptHandler (交互式确认) ===\n");
    println!("提示出现时：y=继续  n=中止  m {{\"cmd\":\"echo safe\"}}=修改参数\n");

    let agent = make_agent_with_handler(CliInterruptHandler)?;
    let mut session = agent.new_session("user-cli").await?;

    let result = session
        .chat("请用 shell 工具执行命令 `rm -rf /tmp/test_hitl`")
        .await;

    match result {
        Ok(r) => println!("执行完成：{}", r.response.text),
        Err(ufox_arc::ArcError::Tool { message, .. }) if message.contains("aborted") => {
            println!("用户已中止操作。");
        }
        Err(e) => return Err(e.into()),
    }

    Ok(())
}
