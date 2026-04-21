//! 流式聊天示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 构建一个 [`ufox_llm::Client`]；
//! 3. 构造 `ChatRequest` 并调用 `chat_stream()` 获取流式响应；
//! 4. 使用 `futures_util::StreamExt` 逐段消费文本增量，并在结束时打印结束原因与用量。
//!
//! 运行前请至少设置：
//! - `UFOX_LLM_PROVIDER`：`openai` / `qwen` / `compatible`
//! - `UFOX_LLM_API_KEY`：对应 Provider 的 API Key
//! - `UFOX_LLM_BASE_URL`：当 `provider=compatible` 时必须设置
//!
//! 可选环境变量：
//! - `UFOX_LLM_MODEL`

use std::env;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use tracing_subscriber::EnvFilter;
use ufox_llm::{ChatRequest, Client, Message, Provider};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let _ = dotenvy::dotenv();
    let provider = read_provider()?;
    let api_key = required_env("UFOX_LLM_API_KEY")?;
    let model = env::var("UFOX_LLM_MODEL").unwrap_or_else(|_| default_model(provider).to_string());

    let mut builder = Client::builder()
        .provider(provider)
        .api_key(api_key)
        .model(model);

    if let Some(base_url) = optional_env("UFOX_LLM_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = builder.build().context("构建客户端失败")?;
    let messages = vec![
        Message::system("你是一位简洁且专业的中文助手。"),
        Message::user("请分三行简要介绍 Rust 的优势。"),
    ];

    let request = ChatRequest::new(&messages).build();
    let mut stream = client
        .chat_stream(&request)
        .await
        .context("发起流式聊天失败")?;

    println!("开始接收流式输出：\n");

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("读取流式分片失败")?;

        if !chunk.delta().is_empty() {
            print!("{}", chunk.delta());
        }

        if let Some(reason) = chunk.finish_reason() {
            println!("\n\n结束原因：{}", reason.as_str());
        }

        if let Some(usage) = chunk.usage() {
            println!(
                "Token 用量：prompt={}, completion={}, total={}",
                usage.prompt_tokens(),
                usage.completion_tokens(),
                usage.total_tokens()
            );
        }
    }

    Ok(())
}

fn init_tracing() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("ufox_llm=debug"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .try_init();
}

fn read_provider() -> Result<Provider> {
    let raw = env::var("UFOX_LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());

    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Ok(Provider::OpenAI),
        "qwen" => Ok(Provider::Qwen),
        "compatible" => Ok(Provider::Compatible),
        other => bail!("不支持的 Provider：{other}，可选值为 openai、qwen、compatible"),
    }
}

fn required_env(key: &str) -> Result<String> {
    env::var(key).with_context(|| format!("缺少必需环境变量：{key}"))
}

fn optional_env(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.trim().is_empty())
}

fn default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "gpt-4o",
        Provider::Qwen => "qwen-max",
        Provider::Compatible => "deepseek-chat",
    }
}
