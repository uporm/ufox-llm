use ufox_llm::{ChatRequest, Client};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    // 显式关闭 thinking，便于对比基础示例和关闭推理后的响应差异。
    let client = Client::from_env()?;

    let output = client
        .chat(
            ChatRequest::builder()
                .user_text("1+1 等于几？只回答数字。")
                .thinking(false)
                .max_tokens(1024)
                .build(),
        )
        .await?;

    // 无论是否返回 thinking 都打印出来，方便确认关闭后服务端是否仍然泄露推理内容。
    println!("=== 思考过程 ===\n");
    match output.thinking.as_deref() {
        Some(thinking) if !thinking.is_empty() => println!("{thinking}\n"),
        _ => println!("(未返回 thinking)\n"),
    }

    println!("=== 最终回答 ===\n");
    println!("{}", output.text);
    Ok(())
}
