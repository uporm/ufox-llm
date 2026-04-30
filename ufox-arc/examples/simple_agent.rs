use anyhow::Result;
use futures::StreamExt;
use std::io::Write;
use ufox_arc::{Agent, AgentConfig};
use ufox_llm::Client;
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let client = Client::from_env()?;
    let agent = Agent::builder()
        .llm(client)
        .system("你是一位 Rust 专家，回答简洁准确。")
        .config(AgentConfig::default())
        .build()?;

    // 普通对话
    let mut session = agent.session("user_123", "intro").await?;

    let result = session
        .chat("解释 Rust 中所有权规则的核心要点，三句话以内。")
        .await?;
    println!("=== 普通对话 ===");
    println!("{}", result.response.text);
    println!(
        "[session={}, duration={:?}, tokens={}]",
        result.session_id, result.trace.total_duration, result.trace.total_usage.total_tokens
    );

    // 多轮对话（同一 session 继续）
    let result2 = session.chat("所有权和借用检查器的关系是什么？").await?;
    println!("\n=== 多轮对话 ===");
    println!("{}", result2.response.text);

    // 流式输出（新会话）
    let mut stream_session = agent.session("user_123", "stream-demo").await?;
    let mut stream = stream_session
        .chat_stream("用一段话描述 Rust 的生命周期机制。")
        .await?;

    println!("\n=== 流式输出 ===");
    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(text) = event.chunk.and_then(|c| c.text_delta) {
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
    }
    println!();

    Ok(())
}
