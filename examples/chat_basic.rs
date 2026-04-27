use ufox_llm::{ChatRequest, Client};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    // 默认走环境变量，避免维护两份仅初始化方式不同的示例。
    let client = Client::from_env()?;
    // 也可以显式构建客户端：
    // let client = Client::builder()
    //     .provider(Provider::OpenAI)
    //     .api_key("sk-xxx")
    //     .model("gpt-4o")
    //     .build()?;

    let output = client
        .chat(
            ChatRequest::builder()
                .user_text("解释一下 Rust 的所有权模型")
                .max_tokens(1024)
                .thinking(true)
                .build(),
        )
        .await?;

    if let Some(thinking) = output.thinking.as_deref()
        && !thinking.is_empty()
    {
        println!("=== 思考过程 ===\n");
        println!("{thinking}\n");
    }

    println!("=== 最终回答 ===\n");
    println!("{}", output.text);
    Ok(())
}
