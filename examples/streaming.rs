use futures::StreamExt;
use ufox_llm::{ChatRequest, Client};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    init_tracing();

    // 默认走环境变量，避免维护两份仅初始化方式不同的示例。
    let client = Client::from_env()?;
    // 也可以显式构建客户端：
    // let client = Client::builder()
    //     .provider(Provider::OpenAI)
    //     .api_key("sk-xxx")
    //     .model("gpt-4o")
    //     .build()?;

    let mut stream = client
        .chat_stream(
            ChatRequest::builder()
                .user_text("Rust 是什么？")
                .thinking(true)
                .build(),
        )
        .await?;

    let mut started_thinking = false;
    let mut started_answer = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(thinking) = &chunk.thinking_delta {
            // 只在首个 thinking chunk 时打印标题，避免流式增量把前缀刷满整屏。
            if !started_thinking {
                println!("\n=== 思考过程 ===\n");
                started_thinking = true;
            }
            print!("{thinking}");
        }
        if let Some(text) = &chunk.text_delta {
            if !started_answer {
                if started_thinking {
                    println!("\n\n=== 最终回答 ===\n");
                }
                started_answer = true;
            }
            print!("{text}");
        }
        if chunk.is_finished() {
            break;
        }
    }

    if started_thinking || started_answer {
        println!();
    }

    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    // 默认输出 info，排查更细粒度问题时可通过 RUST_LOG=debug 覆盖。
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
