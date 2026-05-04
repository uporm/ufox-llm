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
        .instructions("你是一位 Rust 专家，回答简洁准确。")
        .config(AgentConfig::default())
        .build()?;

    // 普通对话
    let thread = agent.thread("user_123", "intro");

    let result = agent.run(&thread, "解释 Rust 中所有权规则的核心要点，三句话以内。").await?;
    println!("=== 普通对话 ===");
    println!("{}", result.response.text);
    println!(
        "[thread={}, duration={:?}, tokens={}]",
        result.thread_id, result.trace.total_duration, result.trace.total_usage.total_tokens
    );

    // 多轮对话（同一 thread 继续）
    let result2 = agent.run(&thread, "所有权和借用检查器的关系是什么？").await?;
    println!("\n=== 多轮对话 ===");
    println!("{}", result2.response.text);

    // 流式输出（新线程）
    let stream_thread = agent.thread("user_123", "stream-demo");
    let mut stream = agent
        .run_stream(&stream_thread, "用一段话描述 Rust 的生命周期机制。")
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
